"""Read IC markings with a trained CRNN checkpoint.

Pre-cut single-line crops:

    python -m icinspect.ocr.infer --input "%USERPROFILE%/Desktop/ic/ic-dataset/ocr/images"

Whole photos of a chip -- cropped to the package, then split into text lines:

    python -m icinspect.ocr.infer --input chip.png --split auto
    python -m icinspect.ocr.infer --input chip.png --split half        # top / bottom
    python -m icinspect.ocr.infer --input crop.png --labels test.txt   # scored run
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from icinspect.config import load_yaml, resolve_output, resolve_paths
from icinspect.ocr.charset import Charset, cer
from icinspect.ocr.dataset import fit_image, list_images, load_image
from icinspect.ocr.model import build_model
from icinspect.ocr.segment import MODES, save_chip, save_crops, segment_config, split_image


def load_checkpoint(ckpt_path: Path, device: torch.device):
    """Restore model + charset + image config from a training checkpoint."""
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = state.get("config", {})
    cs = state.get("charset")
    if cs is None:
        raise ValueError(f"{ckpt_path} has no charset; retrain with the current train.py")
    charset = Charset(symbols=cs["symbols"], uppercase=cs.get("uppercase", True))
    model = build_model(cfg, charset.num_classes).to(device)
    model.load_state_dict(state["model"])
    model.eval()
    return model, charset, cfg


def _image_shape(cfg: dict) -> tuple[int, int, bool]:
    img_cfg = cfg.get("image", {})
    return (
        int(img_cfg.get("height", 32)),
        int(img_cfg.get("width", 160)),
        bool(img_cfg.get("grayscale", True)),
    )


def _to_tensor(img: Image.Image) -> torch.Tensor:
    arr = np.asarray(img, dtype=np.float32) / 255.0
    arr = arr[None, ...] if arr.ndim == 2 else arr.transpose(2, 0, 1)
    return torch.from_numpy((arr - 0.5) / 0.5)


@torch.no_grad()
def _decode_batch(
    model,
    charset: Charset,
    tensors: list[torch.Tensor],
    device,
    min_char_confidence: float = 0.0,
) -> list[dict]:
    """Decode a batch. Characters below `min_char_confidence` are left out.

    Dropping beats guessing on a marking: the TI logo sits in the same band as
    the date code and decodes to junk characters the model is visibly unsure of,
    while the real characters come back at ~1.0. `raw` keeps the unfiltered read
    so nothing disappears silently.
    """
    logits = model(torch.stack(tensors).to(device))          # (T, N, C)
    probs = logits.softmax(dim=2)
    conf, best = probs.max(dim=2)                            # (T, N)
    best = best.permute(1, 0).cpu().tolist()
    conf = conf.permute(1, 0).cpu().tolist()

    out = []
    for frames, scores in zip(best, conf):
        chars = charset.ctc_greedy_decode_scored(frames, scores)
        kept = [(ch, sc) for ch, sc in chars if sc >= min_char_confidence]
        # Confidence is the mean over the characters actually reported, so it is
        # on the same scale as the per-character threshold above it.
        score = sum(sc for _, sc in kept) / len(kept) if kept else 0.0
        out.append(
            {
                "text": "".join(ch for ch, _ in kept),
                "confidence": round(score, 4),
                "raw_text": "".join(ch for ch, _ in chars),
                "dropped": len(chars) - len(kept),
            }
        )
    return out


def min_char_confidence(cfg: dict) -> float:
    """The `decode.min_char_confidence` setting; 0 keeps every character."""
    return float((cfg.get("decode") or {}).get("min_char_confidence", 0.0))


def max_unsure_frac(cfg: dict) -> float:
    """The `decode.max_unsure_frac` setting; 1.0 keeps every non-empty segment."""
    return float((cfg.get("decode") or {}).get("max_unsure_frac", 1.0))


def is_unreadable(read: dict, max_unsure: float) -> bool:
    """True when a segment is a mark rather than text.

    Two ways to fail: nothing survived the per-character filter at all, or most
    of what the model saw was guesswork. The second is what catches a logo --
    it decodes to a couple of characters, one confident by luck, and reporting
    that one would be worse than reporting none.
    """
    if not read["raw_text"]:
        return False                     # nothing was there to begin with
    if not read["text"]:
        return True
    return read["dropped"] / len(read["raw_text"]) >= max_unsure


def predict_images(model, charset: Charset, cfg: dict, images: list[Image.Image], device, batch_size: int = 32):
    """Read already-open crops. Returns [{text, confidence, ...}] in input order."""
    height, width, grayscale = _image_shape(cfg)
    floor = min_char_confidence(cfg)
    results: list[dict] = []
    for start in range(0, len(images), batch_size):
        chunk = images[start : start + batch_size]
        tensors = [_to_tensor(fit_image(im, height, width, grayscale)) for im in chunk]
        results.extend(_decode_batch(model, charset, tensors, device, floor))
    return results


def predict_paths(model, charset: Charset, cfg: dict, image_paths: list[Path], device, batch_size: int = 32):
    """Read one text line per file (the pre-cut-crop path)."""
    height, width, grayscale = _image_shape(cfg)
    floor = min_char_confidence(cfg)
    results: list[dict] = []
    for start in range(0, len(image_paths), batch_size):
        chunk = image_paths[start : start + batch_size]
        tensors = [_to_tensor(load_image(p, height, width, grayscale)) for p in chunk]
        for path, read in zip(chunk, _decode_batch(model, charset, tensors, device, floor)):
            results.append({"image": str(path), **read})
    return results


def predict_split(
    model,
    charset: Charset,
    cfg: dict,
    image_paths: list[Path],
    device,
    mode: str,
    crops_dir: Path | None = None,
    detect_chip: bool | None = None,
    drop_unreadable: bool = True,
    batch_size: int = 32,
    chip_box: tuple[int, int, int, int] | None = None,
):
    """Crop to the package, segment it into text lines, then read every line.

    Returns (rows, ignored). One row per readable segment, top to bottom and
    left to right, with the box it was cut from -- in source-photo pixels, so
    the boxes can be drawn back onto the original. `ignored` holds the segments
    that read as nothing but the model's guesswork. `chip_box` pins the package
    (source pixels, the same for every photo in the call) instead of finding it.
    """
    rows: list[dict] = []
    ignored: list[dict] = []
    max_unsure = max_unsure_frac(cfg)
    for index, path in enumerate(image_paths):
        lines, chip = split_image(path, mode=mode, cfg=cfg, detect_chip=detect_chip, chip_box=chip_box)
        saved = save_crops(path, lines, crops_dir, index) if crops_dir else [None] * len(lines)
        if crops_dir and chip:
            save_chip(path, Image.open(path), chip, crops_dir, index)
        reads = predict_images(model, charset, cfg, [ln.image for ln in lines], device, batch_size)
        for line, crop_path, read in zip(lines, saved, reads):
            # A segment nothing survived in is a logo or a mark the model has no
            # business guessing at -- report it as ignored rather than as a read.
            if drop_unreadable and is_unreadable(read, max_unsure):
                ignored.append({"image": str(path), "box": line.box_str, "raw_text": read["raw_text"]})
                continue
            rows.append(
                {
                    "image": str(path),
                    "line": line.index,
                    "part": line.part,
                    "chip": "{},{},{},{}".format(*chip) if chip else "",
                    "box": line.box_str,
                    "crop": str(crop_path) if crop_path else "",
                    **read,
                }
            )
    return rows, ignored


def group_by_image(rows: list[dict]) -> dict[str, list[dict]]:
    """Line rows keyed by the source photo, order preserved."""
    grouped: dict[str, list[dict]] = {}
    for r in rows:
        grouped.setdefault(r["image"], []).append(r)
    return grouped


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Run OCR on IC marking crops or whole chip photos.")
    ap.add_argument("--input", required=True, help="image file or folder")
    ap.add_argument("--ckpt", default=None, help="checkpoint (default: results/ocr/crnn_best.pt)")
    ap.add_argument("--config", default="ocr.yaml")
    ap.add_argument("--paths", default="paths.yaml")
    ap.add_argument("--labels", default=None, help="TSV ground truth; enables accuracy reporting")
    ap.add_argument("--out", default=None, help="CSV output path")
    ap.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    ap.add_argument(
        "--split",
        default="none",
        choices=list(MODES),
        help="none: one line per file (default) | auto: find the text lines | half: fixed top/bottom",
    )
    ap.add_argument(
        "--max-lines",
        type=int,
        default=None,
        help="auto: keep only the N strongest lines | half: number of equal strips (default 2)",
    )
    ap.add_argument("--crops-dir", default=None, help="where line crops are written (default: <out_dir>/crops)")
    ap.add_argument("--no-crops", action="store_true", help="do not write the line crops to disk")
    ap.add_argument(
        "--no-chip-crop",
        action="store_true",
        help="read the whole frame instead of first cropping to the package body",
    )
    ap.add_argument(
        "--min-char-confidence",
        type=float,
        default=None,
        help="leave out characters the model scores below this (0-1; default from ocr.yaml)",
    )
    ap.add_argument(
        "--keep-unreadable",
        action="store_true",
        help="report segments that decoded to nothing (logos, marks) instead of ignoring them",
    )
    args = ap.parse_args(argv)

    cfg = load_yaml(args.config)
    paths = resolve_paths(args.paths)
    out_dir = resolve_output(cfg.get("train", {}).get("output_dir", "results/ocr"))

    ckpt = Path(args.ckpt) if args.ckpt else out_dir / "crnn_best.pt"
    if not ckpt.exists():
        raise SystemExit(f"Checkpoint not found: {ckpt}\nTrain first: python -m icinspect.ocr.train")

    device = torch.device(
        args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    model, charset, ckpt_cfg = load_checkpoint(ckpt, device)
    # The checkpoint pins the input geometry; the segmentation knobs come from the
    # live config instead, so they can be retuned without retraining.
    run_cfg = dict(ckpt_cfg or cfg)
    run_cfg["segment"] = segment_config(cfg)
    if args.max_lines is not None:
        run_cfg["segment"]["max_lines"] = args.max_lines
    run_cfg["decode"] = dict(cfg.get("decode") or {})
    if args.min_char_confidence is not None:
        run_cfg["decode"]["min_char_confidence"] = args.min_char_confidence
    char_floor = min_char_confidence(run_cfg)

    images = list_images(Path(args.input))
    if not images:
        raise SystemExit(f"No images found under {args.input}")

    splitting = args.split != "none"
    if splitting:
        crops_dir = None
        if not args.no_crops:
            crops_dir = Path(args.crops_dir) if args.crops_dir else out_dir / "crops"
        results, ignored = predict_split(
            model, charset, run_cfg, images, device, args.split, crops_dir,
            detect_chip=not args.no_chip_crop,
            drop_unreadable=not args.keep_unreadable,
        )
    else:
        results, ignored = predict_paths(model, charset, run_cfg, images, device), []

    grouped = group_by_image(results)

    if args.labels:
        from icinspect.ocr.dataset import read_label_file

        lpath = Path(args.labels)
        if not lpath.is_absolute():
            lpath = paths.ocr_root / lpath
        truth = {str(s.image_path.resolve()): charset.normalize(s.text) for s in read_label_file(lpath, [paths.ocr_root])}
        scored, total_cer, exact = 0, 0.0, 0
        for image, rows in grouped.items():
            gold = truth.get(str(Path(image).resolve()))
            # Under --split the label covers the whole marking, so score the joined lines.
            read = " ".join(r["text"] for r in rows if r["text"]) if splitting else rows[0]["text"]
            for r in rows:
                r["truth"] = gold or ""
            if gold is None:
                continue
            scored += 1
            total_cer += cer(read, gold)
            exact += int(read == gold)
        if scored:
            unit = "photo(s)" if splitting else "image(s)"
            print(f"[ocr] char_accuracy {1 - total_cer / scored:.4f}  "
                  f"seq_accuracy {exact / scored:.4f}  over {scored} labelled {unit}")
        else:
            print("[ocr] no predictions matched the label file -- check the paths inside it")

    out_csv = Path(args.out) if args.out else out_dir / "ocr_predictions.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fields = ["image"] + (["line", "part", "chip", "box", "crop"] if splitting else []) + ["text", "confidence"]
    if char_floor > 0:
        fields += ["raw_text", "dropped"]
    if args.labels:
        fields.append("truth")
    with out_csv.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)

    print(f"[ocr] checkpoint : {ckpt}")
    if char_floor > 0:
        dropped = sum(r["dropped"] for r in results) + sum(len(i["raw_text"]) for i in ignored)
        print(f"[ocr] min char confidence {char_floor:g}: left out {dropped} unsure character(s)")
    if ignored:
        shown = ", ".join(sorted({i["raw_text"] for i in ignored})[:5])
        print(f"[ocr] ignored {len(ignored)} unreadable segment(s) -- logos and marks (read as: {shown})")
    if splitting:
        print(f"[ocr] split '{args.split}': {len(images)} photo(s) -> {len(results)} line(s) -> {out_csv}")
        for image, rows in list(grouped.items())[:10]:
            chip = rows[0]["chip"]
            print(f"  {Path(image).name}" + (f"   chip [{chip}]" if chip else "   chip not found"))
            for r in rows:
                unsure = f"  (raw {r['raw_text']})" if char_floor > 0 and r["dropped"] else ""
                where = f"{r['line']}.{r['part']}"
                print(f"    line {where:<5} {r['confidence']:.3f}  {r['text']:<16}  [{r['box']}]{unsure}")
    else:
        print(f"[ocr] read {len(results)} image(s) -> {out_csv}")
        for r in results[:10]:
            unsure = f"  (raw {r['raw_text']})" if char_floor > 0 and r["dropped"] else ""
            print(f"  {r['confidence']:.3f}  {r['text']:<16}  {Path(r['image']).name}{unsure}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
