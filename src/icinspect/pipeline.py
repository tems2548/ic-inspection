"""End-to-end IC test: anomaly verdict + marking readout in one pass.

For each image it reports PASS/FAIL from two independent checks:

  1. PatchCore image-level anomaly score vs a threshold  -> surface/package defects
  2. CRNN OCR of the marking vs an expected part number -> wrong/missing part

The marking is found automatically: the package is located, split into text
lines, and each line read (`--split auto`, the default). Pass `--roi` to pin the
region by hand, or `--split none` when the input is already a single-line crop.

Either check can be disabled, so this also works as OCR-only or anomaly-only.

    python -m icinspect.pipeline --input "%USERPROFILE%/Desktop/ic/ic-dataset/raw"
    python -m icinspect.pipeline --input board.png --expect NE555P
    python -m icinspect.pipeline --input board.png --expect STM32F407VGT6 --roi 0.15 0.35 0.85 0.6
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from icinspect.config import load_yaml, resolve_output, resolve_paths
from icinspect.ocr.charset import cer
from icinspect.ocr.dataset import list_images
from icinspect.ocr.segment import MODES

ROI_HELP = "marking region as fractions of the image: x1 y1 x2 y2 (0-1)"


def crop_roi(
    image_path: Path,
    roi: tuple[float, float, float, float],
    out_dir: Path,
    index: int = 0,
) -> Path:
    """Write the marking crop next to the report so failures are inspectable.

    The filename is prefixed with the source index: inputs from different folders
    routinely share a basename (test/good/ic_000.png vs test/defect/ic_000.png),
    and without the prefix one crop would silently overwrite the other.
    """
    from PIL import Image

    img = Image.open(image_path)
    w, h = img.size
    x1, y1, x2, y2 = roi
    box = (int(x1 * w), int(y1 * h), int(x2 * w), int(y2 * h))
    if box[2] <= box[0] or box[3] <= box[1]:
        raise ValueError(f"Invalid ROI {roi}: x2/y2 must exceed x1/y1")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{index:04d}_{image_path.stem}_roi.png"
    img.crop(box).save(out_path)
    return out_path


def run_anomaly(images: list[Path], cfg_name: str, paths_name: str, ckpt: str | None) -> dict[str, dict]:
    """Score images with PatchCore. Returns {image_path: {score, label}}."""
    import torch

    from icinspect.anomaly.common import build_engine, load_patchcore
    from icinspect.anomaly.infer_patchcore import _rows_from_batch, find_checkpoint

    cfg = load_yaml(cfg_name)
    out_dir = resolve_output(cfg.get("trainer", {}).get("default_root_dir", "results/patchcore"))
    ckpt_path = find_checkpoint(ckpt, out_dir)

    model = load_patchcore(ckpt_path, cfg)

    engine = build_engine(cfg, out_dir)
    # Predict per parent folder so anomalib walks a directory, not N single files.
    folders = sorted({p.parent for p in images})
    scored: dict[str, dict] = {}
    with torch.no_grad():
        for folder in folders:
            for batch in engine.predict(model=model, data_path=str(folder)) or []:
                for row in _rows_from_batch(batch):
                    scored[str(Path(row["image"]).resolve())] = row

    thr = cfg.get("threshold", {}).get("image_score")
    if thr is not None:
        for row in scored.values():
            if row.get("score") is not None:
                row["pred_label"] = int(row["score"] >= float(thr))
    return scored


def run_ocr(
    images: list[Path],
    cfg_name: str,
    ckpt: str | None,
    split: str = "auto",
    crops_dir: Path | None = None,
) -> dict[str, dict]:
    """Read the marking on each photo. Returns {image_path: {text, confidence, lines}}.

    A whole chip photo is not a single text line. By default the package is
    located and split into lines first -- the same path as `ocr.infer --split`.
    Feeding the full frame straight to the CRNN instead squashes a 4K photo into
    a 160x32 strip and decodes to junk, so `split="none"` is only for pre-cut
    crops (and for the --roi path, which has already cut one).
    """
    import torch

    from icinspect.ocr.infer import group_by_image, load_checkpoint, predict_paths, predict_split
    from icinspect.ocr.segment import segment_config

    cfg = load_yaml(cfg_name)
    out_dir = resolve_output(cfg.get("train", {}).get("output_dir", "results/ocr"))
    ckpt_path = Path(ckpt) if ckpt else out_dir / "crnn_best.pt"
    if not ckpt_path.exists():
        raise SystemExit(f"OCR checkpoint not found: {ckpt_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, charset, ckpt_cfg = load_checkpoint(ckpt_path, device)
    # The checkpoint pins the input geometry; decoding and segmentation options
    # come from the live config, so they can be retuned without retraining.
    run_cfg = dict(ckpt_cfg or cfg)
    run_cfg["decode"] = cfg.get("decode") or {}
    run_cfg["segment"] = segment_config(cfg)

    if split == "none":
        # Pair by position: the caller may have passed ROI crops whose paths
        # differ from the source photos.
        return {
            str(src.resolve()): {"text": r["text"], "confidence": r["confidence"], "lines": [r["text"]]}
            for src, r in zip(images, predict_paths(model, charset, run_cfg, images, device))
        }

    rows, _ignored = predict_split(model, charset, run_cfg, images, device, split, crops_dir)
    grouped = group_by_image(rows)
    reads: dict[str, dict] = {}
    for image in images:
        lines = [r for r in grouped.get(str(image), []) if r["text"]]
        # Mean over the lines actually reported, matching how _decode_batch
        # averages over the characters it kept.
        confidence = round(sum(r["confidence"] for r in lines) / len(lines), 4) if lines else 0.0
        reads[str(image.resolve())] = {
            "text": " ".join(r["text"] for r in lines),
            "confidence": confidence,
            "lines": [r["text"] for r in lines],
        }
    return reads


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Run the full IC inspection on images.")
    ap.add_argument("--input", required=True, help="image file or folder of IC photos")
    ap.add_argument("--expect", default=None, help="expected marking text (enables a part-number check)")
    ap.add_argument("--roi", nargs=4, type=float, default=None, metavar=("X1", "Y1", "X2", "Y2"), help=ROI_HELP)
    ap.add_argument("--anomaly-ckpt", default=None)
    ap.add_argument("--ocr-ckpt", default=None)
    ap.add_argument(
        "--split",
        default="auto",
        choices=list(MODES),
        help="how to find the marking: auto (default) | half | none (input is already a crop)",
    )
    ap.add_argument("--no-crops", action="store_true", help="do not write the line crops to disk")
    ap.add_argument("--skip-anomaly", action="store_true")
    ap.add_argument("--skip-ocr", action="store_true")
    ap.add_argument("--max-cer", type=float, default=0.0, help="tolerated character error rate vs --expect")
    ap.add_argument("--min-confidence", type=float, default=0.0, help="fail reads below this OCR confidence")
    ap.add_argument("--patchcore-config", default="patchcore.yaml")
    ap.add_argument("--ocr-config", default="ocr.yaml")
    ap.add_argument("--paths", default="paths.yaml")
    ap.add_argument("--out", default=None, help="report CSV path")
    args = ap.parse_args(argv)

    if args.skip_anomaly and args.skip_ocr:
        raise SystemExit("Nothing to do: both checks are skipped.")

    paths = resolve_paths(args.paths)
    images = list_images(Path(args.input))
    if not images:
        raise SystemExit(f"No images found under {args.input}")

    report_dir = paths.results_dir / "inspect"
    report_dir.mkdir(parents=True, exist_ok=True)

    anomaly = {} if args.skip_anomaly else run_anomaly(
        images, args.patchcore_config, args.paths, args.anomaly_ckpt
    )

    ocr: dict[str, dict] = {}
    if not args.skip_ocr:
        # An explicit ROI is already a single line, so it goes straight to the
        # reader; anything else is a whole photo and has to be segmented first.
        if args.roi:
            crops = [
                crop_roi(p, tuple(args.roi), report_dir / "roi", index=i)
                for i, p in enumerate(images)
            ]
            split = "none"
        else:
            crops = images
            split = args.split
        reads = run_ocr(
            crops,
            args.ocr_config,
            args.ocr_ckpt,
            split=split,
            crops_dir=None if args.no_crops else report_dir / "lines",
        )
        # run_ocr keys on the path it was handed; under --roi that is the crop,
        # so map the results back onto the source photos by position.
        ocr = {
            str(src.resolve()): reads.get(str(crop.resolve()), {})
            for src, crop in zip(images, crops)
        }

    rows: list[dict] = []
    expected = args.expect.upper() if args.expect else None
    for image in images:
        key = str(image.resolve())
        a = anomaly.get(key, {})
        o = ocr.get(key, {})
        reasons: list[str] = []

        if not args.skip_anomaly:
            if a.get("pred_label") == 1:
                reasons.append("anomaly")
            elif not a:
                reasons.append("anomaly-missing")

        text = o.get("text", "")
        confidence = o.get("confidence")
        if not args.skip_ocr:
            if expected is not None:
                if cer(text, expected) > args.max_cer:
                    reasons.append("marking-mismatch")
            elif not text:
                reasons.append("marking-unreadable")
            if args.min_confidence and (confidence or 0.0) < args.min_confidence:
                reasons.append("low-confidence")

        rows.append(
            {
                "image": str(image),
                "verdict": "FAIL" if reasons else "PASS",
                "reasons": ";".join(reasons),
                "anomaly_score": a.get("score"),
                "anomaly_label": a.get("pred_label"),
                "marking": text,
                "marking_lines": " | ".join(o.get("lines") or []),
                "ocr_confidence": confidence,
                "expected": expected or "",
            }
        )

    out_csv = Path(args.out) if args.out else report_dir / "inspection_report.csv"
    with out_csv.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    failed = sum(1 for r in rows if r["verdict"] == "FAIL")
    print(f"[inspect] {len(rows)} image(s): {len(rows) - failed} PASS / {failed} FAIL")
    print(f"[inspect] report: {out_csv}")
    for r in rows[:15]:
        score = f"{r['anomaly_score']:.4f}" if isinstance(r["anomaly_score"], float) else "  n/a "
        print(f"  {r['verdict']:<4} score={score}  '{r['marking']}'  {Path(r['image']).name}"
              + (f"  [{r['reasons']}]" if r["reasons"] else ""))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
