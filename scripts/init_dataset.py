"""Create the dataset tree on the Desktop, outside the code repo.

    python scripts/init_dataset.py
    python scripts/init_dataset.py --root "D:/data/ic-dataset"

Safe to re-run: it only creates what is missing and never deletes images.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from icinspect.config import load_yaml, resolve_paths  # noqa: E402

DATASET_README = """\
# IC inspection dataset

This folder is deliberately OUTSIDE the code repo. Point the project at it via
`configs/paths.yaml` (`dataset_root`) or the `IC_DATASET_ROOT` environment variable.

    anomaly/ic/
      train/good/            defect-free IC images -- PatchCore memory bank (30-300 is plenty)
      test/good/             defect-free images held out for testing
      test/defect/           defective images (scratches, chips, bent pins, missing marking)
      ground_truth/defect/   OPTIONAL pixel masks, same filename as the defect image,
                             white = defect. Enables pixel-level AUROC.

    ocr/
      images/                cropped marking images, ONE text line per crop
      train.txt              "<relative/path.png><TAB><LABEL>" per line
      val.txt                optional; without it train.py splits train.txt
      test.txt               optional held-out set
      synth/                 generated crops (scripts/gen_synth_ocr.py) + synth/labels.txt

    raw/                     unlabelled captures, staging area for new inspections

Shooting notes that matter more than model tuning:
  * Fix the camera, lens, working distance and lighting. PatchCore compares against
    the training images -- if illumination drifts, everything looks anomalous.
  * Keep the IC in the same orientation and roughly the same position.
  * Marking crops should be tight on the text, upright, with a little margin.
"""

ANOMALY_DIRS = [
    "anomaly/ic/train/good",
    "anomaly/ic/test/good",
    "anomaly/ic/test/defect",
    "anomaly/ic/ground_truth/defect",
]
OCR_DIRS = ["ocr/images", "ocr/synth"]
OTHER_DIRS = ["raw"]

LABEL_TEMPLATE = """\
# One sample per line:  <relative/path.png><TAB><LABEL>
# Paths are relative to this file. Labels are upper-cased before training.
# Example:
# images/u1_top.png\tSTM32F407VGT6
"""


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Create the IC dataset folder tree.")
    ap.add_argument("--root", default=None, help="dataset root (default: configs/paths.yaml)")
    ap.add_argument("--paths", default="paths.yaml")
    args = ap.parse_args(argv)

    if args.root:
        root = Path(args.root).expanduser()
        anomaly_sub = load_yaml(args.paths).get("anomaly_subdir", "anomaly/ic")
    else:
        paths = resolve_paths(args.paths)
        root = paths.dataset_root
        anomaly_sub = None

    created: list[Path] = []
    for rel in ANOMALY_DIRS + OCR_DIRS + OTHER_DIRS:
        target = root / rel
        if not target.exists():
            target.mkdir(parents=True, exist_ok=True)
            created.append(target)

    readme = root / "README.md"
    if not readme.exists():
        readme.write_text(DATASET_README, encoding="utf-8")
        created.append(readme)

    for split in ("train", "val", "test"):
        label_file = root / "ocr" / f"{split}.txt"
        if not label_file.exists():
            label_file.write_text(LABEL_TEMPLATE, encoding="utf-8")
            created.append(label_file)

    print(f"[dataset] root: {root}")
    if created:
        for path in created:
            print(f"  + {path.relative_to(root)}")
    else:
        print("  (already complete -- nothing to create)")
    if anomaly_sub:
        print(f"[dataset] note: configs/paths.yaml expects anomaly data under {anomaly_sub}")
    print("\nNext:")
    print(f"  1. Copy defect-free IC photos into {root / 'anomaly/ic/train/good'}")
    print(f"  2. Copy defective photos into      {root / 'anomaly/ic/test/defect'}")
    print("  3. python scripts/gen_synth_ocr.py     (bootstrap OCR data)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
