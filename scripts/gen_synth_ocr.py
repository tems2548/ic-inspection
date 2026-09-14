"""Generate synthetic IC-marking crops so OCR training can start immediately.

    python scripts/gen_synth_ocr.py                 # uses configs/ocr.yaml -> synth.count
    python scripts/gen_synth_ocr.py --count 20000 --split

`--split` also writes train.txt / val.txt under the OCR root so
`python -m icinspect.ocr.train` works with no further setup.
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from icinspect.config import load_yaml, resolve_paths  # noqa: E402
from icinspect.ocr.synth import generate, load_fonts  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Generate synthetic IC marking crops.")
    ap.add_argument("--config", default="ocr.yaml")
    ap.add_argument("--paths", default="paths.yaml")
    ap.add_argument("--count", type=int, default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--split", action="store_true", help="also write train.txt / val.txt")
    ap.add_argument("--val-ratio", type=float, default=0.1)
    args = ap.parse_args(argv)

    cfg = load_yaml(args.config)
    paths = resolve_paths(args.paths)
    scfg = cfg.get("synth", {})

    count = args.count if args.count is not None else int(scfg.get("count", 5000))
    seed = args.seed if args.seed is not None else int(scfg.get("seed", 1234))
    height = int(cfg.get("image", {}).get("height", 32))

    fonts = load_fonts(scfg.get("fonts_dir", "C:/Windows/Fonts"), list(scfg.get("fonts", [])))
    if not fonts:
        print("[synth] warning: no TrueType fonts found; falling back to PIL's bitmap font "
              "(low variety -- set synth.fonts_dir in configs/ocr.yaml)")
    else:
        print(f"[synth] fonts: {', '.join(f.name for f in fonts)}")

    out_dir = paths.ocr_synth
    label_file = out_dir / "labels.txt"
    written = generate(
        out_dir=out_dir,
        label_file=label_file,
        count=count,
        fonts=fonts,
        height=height,
        min_len=int(scfg.get("min_len", 3)),
        max_len=int(scfg.get("max_len", 12)),
        seed=seed,
        relative_to=paths.ocr_root,   # paths stay valid from the OCR root
    )
    print(f"[synth] wrote {written} crops -> {out_dir}")
    print(f"[synth] labels: {label_file}")

    if args.split:
        lines = [ln for ln in label_file.read_text(encoding="utf-8").splitlines() if ln.strip()]
        random.Random(seed).shuffle(lines)
        n_val = max(1, int(len(lines) * args.val_ratio))
        paths.labels_file("val").write_text("\n".join(lines[:n_val]) + "\n", encoding="utf-8")
        paths.labels_file("train").write_text("\n".join(lines[n_val:]) + "\n", encoding="utf-8")
        print(f"[synth] split -> train.txt ({len(lines) - n_val})  val.txt ({n_val})")
        print("\nNext: python -m icinspect.ocr.train")
    else:
        print("\nNext: python -m icinspect.ocr.train --labels synth/labels.txt")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
