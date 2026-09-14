"""Train the CRNN/CTC marking recognizer.

    python -m icinspect.ocr.train
    python -m icinspect.ocr.train --labels synth/labels.txt --epochs 20
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from icinspect.config import load_yaml, resolve_output, resolve_paths
from icinspect.ocr.charset import Charset, cer
from icinspect.ocr.dataset import (
    MarkingDataset,
    collate,
    filter_encodable,
    read_label_file,
    split_samples,
)
from icinspect.ocr.model import build_model


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def pick_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@torch.no_grad()
def evaluate(model, loader, charset: Charset, device) -> dict[str, float]:
    model.eval()
    total_cer, exact, n = 0.0, 0, 0
    for images, _, _, texts in loader:
        logits = model(images.to(device))                    # (T, N, C)
        best = logits.argmax(dim=2).permute(1, 0).cpu().tolist()
        for frames, target in zip(best, texts):
            pred = charset.ctc_greedy_decode(frames)
            gold = charset.normalize(target)
            total_cer += cer(pred, gold)
            exact += int(pred == gold)
            n += 1
    return {
        "char_accuracy": 1.0 - (total_cer / max(n, 1)),
        "seq_accuracy": exact / max(n, 1),
        "samples": float(n),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Train the CRNN OCR model on IC markings.")
    ap.add_argument("--config", default="ocr.yaml")
    ap.add_argument("--paths", default="paths.yaml")
    ap.add_argument("--labels", default=None, help="label file, relative to the OCR root (default: train.txt)")
    ap.add_argument("--val-labels", default=None, help="explicit validation label file")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    ap.add_argument("--resume", default=None, help="checkpoint to warm-start from")
    args = ap.parse_args(argv)

    cfg = load_yaml(args.config)
    paths = resolve_paths(args.paths)
    tcfg = cfg.setdefault("train", {})
    if args.epochs is not None:
        tcfg["epochs"] = args.epochs
    if args.batch_size is not None:
        tcfg["batch_size"] = args.batch_size
    if args.lr is not None:
        tcfg["lr"] = args.lr

    set_seed(int(tcfg.get("seed", 42)))
    device = pick_device(args.device)
    charset = Charset.from_config(cfg)

    # --- data -------------------------------------------------------------
    label_path = Path(args.labels) if args.labels else paths.labels_file("train")
    if not label_path.is_absolute():
        label_path = paths.ocr_root / label_path
    train_samples = filter_encodable(read_label_file(label_path, [paths.ocr_root]), charset)

    if args.val_labels:
        vpath = Path(args.val_labels)
        if not vpath.is_absolute():
            vpath = paths.ocr_root / vpath
        val_samples = filter_encodable(read_label_file(vpath, [paths.ocr_root]), charset)
    elif paths.labels_file("val").exists():
        val_samples = filter_encodable(read_label_file(paths.labels_file("val"), [paths.ocr_root]), charset)
    else:
        train_samples, val_samples = split_samples(
            train_samples, float(tcfg.get("val_split", 0.1)), int(tcfg.get("seed", 42))
        )

    if not train_samples:
        raise SystemExit("No usable training samples. Check your label file and charset.")

    width = int(cfg.get("image", {}).get("width", 160))
    max_len = max(len(charset.normalize(s.text)) for s in train_samples)

    train_ds = MarkingDataset(train_samples, charset, cfg, train=True, seed=int(tcfg.get("seed", 42)))
    val_ds = MarkingDataset(val_samples, charset, cfg, train=False) if val_samples else None
    workers = int(tcfg.get("num_workers", 0))
    train_loader = DataLoader(
        train_ds, batch_size=int(tcfg.get("batch_size", 32)), shuffle=True,
        collate_fn=collate, num_workers=workers, drop_last=False,
    )
    val_loader = (
        DataLoader(val_ds, batch_size=int(tcfg.get("batch_size", 32)), shuffle=False,
                   collate_fn=collate, num_workers=workers)
        if val_ds else None
    )

    # --- model ------------------------------------------------------------
    model = build_model(cfg, charset.num_classes).to(device)
    time_steps = model.output_length(width)
    if time_steps < max_len:
        raise SystemExit(
            f"image.width={width} yields only {time_steps} CTC frames but the longest label "
            f"has {max_len} characters. Increase image.width in configs/ocr.yaml to at least "
            f"{4 * (max_len + 1) + 4}."
        )

    if args.resume:
        state = torch.load(args.resume, map_location=device)
        model.load_state_dict(state["model"])
        print(f"[ocr] resumed from {args.resume}")

    criterion = nn.CTCLoss(blank=0, zero_infinity=True)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=float(tcfg.get("lr", 1e-3)),
        weight_decay=float(tcfg.get("weight_decay", 0.0)),
    )
    epochs = int(tcfg.get("epochs", 40))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(epochs, 1))
    use_amp = bool(tcfg.get("amp", True)) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    out_dir = resolve_output(tcfg.get("output_dir", "results/ocr"))
    out_dir.mkdir(parents=True, exist_ok=True)
    best_path = out_dir / "crnn_best.pt"
    last_path = out_dir / "crnn_last.pt"
    history_path = out_dir / "history.csv"

    print(f"[ocr] device={device} train={len(train_ds)} val={len(val_ds) if val_ds else 0} "
          f"classes={charset.num_classes} T={time_steps}")

    def save(path: Path, epoch: int, metrics: dict) -> None:
        torch.save(
            {
                "model": model.state_dict(),
                "charset": {"symbols": charset.symbols, "uppercase": charset.uppercase},
                "config": cfg,
                "epoch": epoch,
                "metrics": metrics,
            },
            path,
        )

    patience = int(tcfg.get("early_stop_patience", 8))
    best_score, patience_left = -1.0, patience
    history: list[dict] = []
    grad_clip = float(tcfg.get("grad_clip", 5.0))

    for epoch in range(1, epochs + 1):
        model.train()
        started, running, batches = time.time(), 0.0, 0
        for images, targets, target_lengths, _ in train_loader:
            images = images.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
                logits = model(images)                                # (T, N, C)
            # CTC is numerically fragile in fp16, so the loss runs in fp32.
            log_probs = logits.float().log_softmax(2)
            input_lengths = torch.full(
                (images.size(0),), log_probs.size(0), dtype=torch.long, device=log_probs.device
            )
            loss = criterion(log_probs, targets.to(device), input_lengths, target_lengths.to(device))
            scaler.scale(loss).backward()
            if grad_clip > 0:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
            running += float(loss.detach())
            batches += 1
        scheduler.step()

        row = {"epoch": epoch, "loss": running / max(batches, 1), "seconds": time.time() - started}
        if val_loader is not None:
            row.update(evaluate(model, val_loader, charset, device))
        history.append(row)

        msg = f"[ocr] epoch {epoch:3d}/{epochs}  loss {row['loss']:.4f}"
        if "seq_accuracy" in row:
            msg += f"  char_acc {row['char_accuracy']:.4f}  seq_acc {row['seq_accuracy']:.4f}"
        print(msg + f"  ({row['seconds']:.1f}s)")

        save(last_path, epoch, row)
        score = row.get("seq_accuracy", -row["loss"])
        if score > best_score:
            best_score, patience_left = score, patience
            save(best_path, epoch, row)
        else:
            patience_left -= 1
            if patience_left <= 0:
                print(f"[ocr] early stop at epoch {epoch} (no improvement)")
                break

    with history_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=sorted({k for r in history for k in r}))
        writer.writeheader()
        writer.writerows(history)
    (out_dir / "metrics.json").write_text(
        json.dumps({"best_score": best_score, "history": history}, indent=2), encoding="utf-8"
    )

    print(f"[ocr] best model : {best_path}")
    print(f"[ocr] history    : {history_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
