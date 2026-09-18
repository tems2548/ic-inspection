"""Turn photos of a known marking into labelled line crops for training.

The reader fails on lighting it has not seen -- a batch shot with a fainter
etch reads as `NED55` where the earlier batch read `NE555P` -- and no
inference-time trick fixes that: the model has to see such crops. Labelling
them by hand is the slow part, and it is unnecessary when every photo in a
folder carries the same marking. Then the label of a segment follows from
*where* it is: first line, second segment. So this takes the marking as
lines of segments (`NE555P | 93M | DNJW26 G4`), segments each photo with the
same code the window uses, assigns the texts by position, and writes the
crops plus a `train.txt`-style label line for each.

The reads are used only to tell a logo from text where a line has more
segments than the marking says (the TI logo sits beside `93M`); a photo
whose lines do not come out as the marking says is skipped, not guessed at.

    python -m icinspect.ocr.harvest --input raw --expect "NE555P | 93M | DNJW26 G4"
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path

from icinspect.ocr.charset import cer

LINE_SEP = "|"


@dataclass(frozen=True)
class Pick:
    """One crop to keep: which segment row it came from and what it says."""

    row: dict
    text: str


def parse_expected(text: str) -> list[list[str]]:
    """`"NE555P | 93M | DNJW26 G4"` -> [["NE555P"], ["93M"], ["DNJW26", "G4"]]."""
    lines = [part.split() for part in text.upper().split(LINE_SEP)]
    lines = [line for line in lines if line]
    if not lines:
        raise ValueError("Give the marking as lines separated by '|', segments by spaces.")
    return lines


def assign(rows: list[dict], expected: list[list[str]]) -> tuple[list[Pick], str | None]:
    """Match one photo's segment rows to the expected texts by position.

    Returns (picks, problem). `problem` says why the photo was skipped; the
    picks are then empty. A line with exactly the expected number of segments
    is taken as is; a line with more has the extras (logos, marks) dropped by
    keeping the subset whose raw reads sit closest to the expected texts; a
    line with fewer is a segmentation miss and the whole photo is skipped,
    because a wrong label is worse than no crop.
    """
    by_line: dict[int, list[dict]] = {}
    for row in rows:
        by_line.setdefault(int(row["line"]), []).append(row)
    lines = [sorted(by_line[key], key=lambda r: int(r["box"].split(",")[0]))
             for key in sorted(by_line)]
    if len(lines) != len(expected):
        return [], f"{len(lines)} line(s) found, marking has {len(expected)}"

    picks: list[Pick] = []
    for index, (found, wanted) in enumerate(zip(lines, expected)):
        if len(found) < len(wanted):
            return [], f"line {index}: {len(found)} segment(s) found, marking has {len(wanted)}"
        best: tuple[float, tuple[dict, ...]] | None = None
        for subset in combinations(found, len(wanted)):
            cost = sum(cer(row.get("raw_text") or row.get("text") or "", text)
                       for row, text in zip(subset, wanted))
            if best is None or cost < best[0]:
                best = (cost, subset)
        assert best is not None
        picks.extend(Pick(row, text) for row, text in zip(best[1], wanted))
    return picks, None


def crop_name(image: Path, row: dict) -> str:
    """`train_good_P00001_l0p0.png`: the two folders above the photo lead the
    name, because capture sessions reuse photo numbers -- `train/good/P00001`
    and `test/good/P00001` are different parts -- and a crop must not overwrite
    another folder's."""
    parents = [parent.name for parent in (image.parent.parent, image.parent) if parent.name]
    return "_".join([*parents, image.stem]) + f"_l{int(row['line'])}p{int(row['part'])}.png"


def harvest(
    images: list[Path],
    expected: list[list[str]],
    out_dir: Path,
    labels: Path,
    predict,
    log=print,
) -> tuple[int, int]:
    """Crop and label every photo. Returns (crops written, photos skipped).

    `predict(images)` returns the segment rows for those photos, as
    `icinspect.ocr.infer.predict_split` does with `drop_unreadable=False`:
    every segment, the unreadable ones included, since a faint line that the
    reader gave up on is the very crop the training set is missing.
    """
    from PIL import Image

    out_dir.mkdir(parents=True, exist_ok=True)
    labels.parent.mkdir(parents=True, exist_ok=True)
    known = set()
    if labels.exists():
        for line in labels.read_text(encoding="utf-8").splitlines():
            if line.strip() and not line.lstrip().startswith("#"):
                known.add(line.split("\t", 1)[0].strip())

    rows = predict(images)
    by_image: dict[str, list[dict]] = {}
    for row in rows:
        by_image.setdefault(str(Path(row["image"]).resolve()), []).append(row)

    written = skipped = 0
    new_lines: list[str] = []
    for image in images:
        picks, problem = assign(by_image.get(str(image.resolve()), []), expected)
        if problem:
            log(f"[harvest] skip {image.name}: {problem}")
            skipped += 1
            continue
        with Image.open(image) as photo:
            photo.load()
            for pick in picks:
                name = crop_name(image, pick.row)
                target = out_dir / name
                rel = target.relative_to(labels.parent).as_posix() \
                    if _is_under(target, labels.parent) else str(target)
                x1, y1, x2, y2 = (int(v) for v in pick.row["box"].split(","))
                photo.crop((x1, y1, x2, y2)).save(target)
                if rel not in known:
                    new_lines.append(f"{rel}\t{pick.text}")
                    known.add(rel)
                written += 1
    if new_lines:
        existing = labels.read_text(encoding="utf-8") if labels.exists() else ""
        with labels.open("a", encoding="utf-8") as fh:
            if existing and not existing.endswith("\n"):
                fh.write("\n")
            fh.write(f"# harvested from {images[0].parent} ({len(new_lines)} crops)\n")
            fh.write("\n".join(new_lines) + "\n")
    return written, skipped


def _is_under(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Harvest labelled line crops from photos of a known marking.")
    ap.add_argument("--input", required=True, help="photo or folder of photos, all with the same marking")
    ap.add_argument("--expect", required=True,
                    help="the marking, lines separated by '|', segments by spaces: \"NE555P | 93M | DNJW26 G4\"")
    ap.add_argument("--out", default=None, help="crop folder (default: <ocr root>/images/harvest)")
    ap.add_argument("--labels", default="train.txt", help="label file to append to, under the OCR root")
    ap.add_argument("--ckpt", default=None, help="reader checkpoint (default: results/ocr/crnn_best.pt)")
    ap.add_argument("--config", default="ocr.yaml")
    ap.add_argument("--paths", default="paths.yaml")
    ap.add_argument("--no-chip-crop", action="store_true", help="the photos are already package crops")
    args = ap.parse_args(argv)

    import torch

    from icinspect.config import load_yaml, resolve_output, resolve_paths
    from icinspect.ocr.dataset import list_images
    from icinspect.ocr.infer import load_checkpoint, predict_split
    from icinspect.ocr.segment import segment_config

    expected = parse_expected(args.expect)
    paths = resolve_paths(args.paths)
    cfg = load_yaml(args.config)
    images = list_images(Path(args.input))
    if not images:
        raise SystemExit(f"No images found under {args.input}")

    out_dir = resolve_output(cfg.get("train", {}).get("output_dir", "results/ocr"))
    ckpt = Path(args.ckpt) if args.ckpt else out_dir / "crnn_best.pt"
    if not ckpt.exists():
        raise SystemExit(f"Reader checkpoint not found: {ckpt}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, charset, ckpt_cfg = load_checkpoint(ckpt, device)
    run_cfg = dict(ckpt_cfg or cfg)
    run_cfg["segment"] = segment_config(cfg)
    run_cfg["decode"] = dict(cfg.get("decode") or {})

    def predict(batch: list[Path]) -> list[dict]:
        rows, _ = predict_split(model, charset, run_cfg, batch, device, "auto", None,
                                detect_chip=not args.no_chip_crop, drop_unreadable=False)
        return rows

    crops = Path(args.out) if args.out else paths.ocr_root / "images" / "harvest"
    labels = Path(args.labels)
    if not labels.is_absolute():
        labels = paths.ocr_root / labels
    print(f"[harvest] {len(images)} photo(s), marking {' | '.join(' '.join(l) for l in expected)}")
    written, skipped = harvest(images, expected, crops, labels, predict)
    print(f"[harvest] {written} crop(s) -> {crops}")
    print(f"[harvest] labels appended to {labels}; {skipped} photo(s) skipped")
    if written:
        print("[harvest] next: Train tab -> tick fine-tune -> Train reader")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
