"""Score new IC images with a trained PatchCore checkpoint.

    python -m icinspect.anomaly.infer_patchcore --ckpt results/patchcore/.../model.ckpt \
        --input "%USERPROFILE%/Desktop/ic-dataset/raw"
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

from icinspect.anomaly.common import build_engine, load_patchcore
from icinspect.config import load_yaml, resolve_output, resolve_paths

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def find_checkpoint(explicit: str | None, search_root: Path) -> Path:
    if explicit:
        p = Path(explicit)
        if not p.exists():
            raise FileNotFoundError(f"Checkpoint not found: {p}")
        return p
    ckpts = sorted(search_root.rglob("*.ckpt"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not ckpts:
        raise FileNotFoundError(
            f"No .ckpt under {search_root}. Train first: "
            "python -m icinspect.anomaly.train_patchcore"
        )
    return ckpts[0]


def _as_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        try:
            return float(value.item())  # 0-d tensor
        except Exception:
            return None


def _rows_from_batch(batch: Any) -> list[dict[str, Any]]:
    """anomalib returns a Batch dataclass (2.x) or a dict (1.x); handle both."""
    get = batch.get if isinstance(batch, dict) else lambda k, d=None: getattr(batch, k, d)
    paths = get("image_path") or []
    scores = get("pred_score")
    labels = get("pred_label")

    if isinstance(paths, (str, Path)):
        paths = [paths]

    rows: list[dict[str, Any]] = []
    for i, img_path in enumerate(paths):
        score = _as_float(scores[i]) if scores is not None and len(scores) > i else None
        label = labels[i] if labels is not None and len(labels) > i else None
        rows.append(
            {
                "image": str(img_path),
                "score": score,
                "pred_label": int(_as_float(label)) if _as_float(label) is not None else None,
            }
        )
    return rows


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Run PatchCore inference on images.")
    ap.add_argument("--ckpt", default=None, help="checkpoint path (default: newest under results)")
    ap.add_argument("--input", required=True, help="image file or folder of images")
    ap.add_argument("--config", default="patchcore.yaml")
    ap.add_argument("--paths", default="paths.yaml")
    ap.add_argument("--out", default=None, help="CSV output path")
    ap.add_argument("--threshold", type=float, default=None, help="override image score cutoff")
    args = ap.parse_args(argv)

    cfg = load_yaml(args.config)
    _ = resolve_paths(args.paths)
    out_dir = resolve_output(cfg.get("trainer", {}).get("default_root_dir", "results/patchcore"))

    ckpt = find_checkpoint(args.ckpt, out_dir)
    source = Path(args.input)
    if not source.exists():
        raise FileNotFoundError(f"Input not found: {source}")

    model = load_patchcore(ckpt, cfg)

    engine = build_engine(cfg, out_dir)
    predictions = engine.predict(model=model, data_path=str(source)) or []

    rows: list[dict[str, Any]] = []
    for batch in predictions:
        rows.extend(_rows_from_batch(batch))

    thr = args.threshold
    if thr is None:
        thr = cfg.get("threshold", {}).get("image_score")
    if thr is not None:
        for r in rows:
            if r["score"] is not None:
                r["pred_label"] = int(r["score"] >= float(thr))

    out_csv = Path(args.out) if args.out else out_dir / "predictions.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["image", "score", "pred_label"])
        writer.writeheader()
        writer.writerows(rows)

    n_bad = sum(1 for r in rows if r["pred_label"] == 1)
    print(f"[patchcore] checkpoint : {ckpt}")
    print(f"[patchcore] scored {len(rows)} image(s), {n_bad} flagged as anomalous")
    print(f"[patchcore] wrote      : {out_csv}")
    for r in rows[:10]:
        score = f"{r['score']:.4f}" if r["score"] is not None else "n/a"
        print(f"  {score}  label={r['pred_label']}  {Path(r['image']).name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
