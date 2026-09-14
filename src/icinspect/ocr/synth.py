"""Synthetic IC top-marking generator.

Lets you train and sanity-check the recognizer before enough real crops are
labelled. PIL + numpy only, so it runs without torch installed.
"""

from __future__ import annotations

import random
import string
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

# Shapes that show up on real IC tops: part numbers, date/lot codes, logos-as-text.
_PREFIXES = ["STM32F", "LM", "NE", "TL", "MAX", "AT", "PIC", "ESP", "CD", "SN", "74HC", "74LS", "MC", "LT", "ADS"]
_SUFFIXES = ["N", "P", "D", "DR", "T6", "VGT6", "RBT6", "CN", "ID", "PWR", "AN", "M", ""]


def random_marking(rng: random.Random, min_len: int = 3, max_len: int = 12) -> str:
    """A plausible marking string, clipped to [min_len, max_len]."""
    kind = rng.random()
    if kind < 0.45:                                   # part number
        text = (
            rng.choice(_PREFIXES)
            + "".join(rng.choice(string.digits) for _ in range(rng.randint(2, 4)))
            + rng.choice(_SUFFIXES)
        )
    elif kind < 0.65:                                 # date code, e.g. 2412 / K 2338
        text = f"{rng.randint(15, 26):02d}{rng.randint(1, 52):02d}"
        if rng.random() < 0.4:
            text = rng.choice(string.ascii_uppercase) + " " + text
    elif kind < 0.85:                                 # lot code
        text = "".join(rng.choice(string.ascii_uppercase + string.digits) for _ in range(rng.randint(4, 9)))
    else:                                             # mixed with separators
        text = "".join(
            rng.choice(string.ascii_uppercase + string.digits + "-./+") for _ in range(rng.randint(4, 10))
        ).strip("-./+ ")

    text = text.upper()
    if len(text) < min_len:
        text += "".join(rng.choice(string.digits) for _ in range(min_len - len(text)))
    return text[:max_len]


def load_fonts(fonts_dir: str | Path, names: list[str]) -> list[Path]:
    """Resolve configured font names; fall back to anything TTF in the folder."""
    base = Path(fonts_dir)
    found = [base / n for n in names if (base / n).exists()]
    if not found and base.is_dir():
        found = sorted(base.glob("*.ttf"))[:8]
    return found


def render(text: str, font_path: Path | None, rng: random.Random, height: int = 32) -> Image.Image:
    """Render one marking as a grayscale crop with camera-like degradation."""
    font_size = rng.randint(26, 40)
    try:
        font = ImageFont.truetype(str(font_path), font_size) if font_path else ImageFont.load_default()
    except OSError:
        font = ImageFont.load_default()

    # Laser marking is usually light text on dark package; occasionally inverted.
    inverted = rng.random() < 0.25
    bg = rng.randint(180, 240) if inverted else rng.randint(20, 70)
    fg = rng.randint(20, 80) if inverted else rng.randint(170, 235)

    pad_x, pad_y = rng.randint(6, 18), rng.randint(4, 10)
    probe = Image.new("L", (8, 8), bg)
    box = ImageDraw.Draw(probe).textbbox((0, 0), text, font=font)
    tw, th = max(1, box[2] - box[0]), max(1, box[3] - box[1])

    img = Image.new("L", (tw + 2 * pad_x, th + 2 * pad_y), bg)
    draw = ImageDraw.Draw(img)
    if rng.random() < 0.5:  # spacing varies a lot between marking machines
        x = pad_x - box[0]
        for ch in text:
            draw.text((x, pad_y - box[1]), ch, font=font, fill=fg)
            x += draw.textlength(ch, font=font) + rng.uniform(-0.5, 2.0)
    else:
        draw.text((pad_x - box[0], pad_y - box[1]), text, font=font, fill=fg)

    # Package texture + uneven lighting + defocus + sensor noise.
    arr = np.asarray(img, dtype=np.float32)
    h, w = arr.shape
    grain = np.random.normal(0.0, rng.uniform(2.0, 7.0), (h, w)).astype(np.float32)
    gx = np.linspace(rng.uniform(-25, 0), rng.uniform(0, 25), w, dtype=np.float32)[None, :]
    gy = np.linspace(rng.uniform(-15, 0), rng.uniform(0, 15), h, dtype=np.float32)[:, None]
    img = Image.fromarray(np.clip(arr + grain + gx + gy, 0, 255).astype(np.uint8), mode="L")

    if rng.random() < 0.8:
        img = img.filter(ImageFilter.GaussianBlur(rng.uniform(0.2, 1.2)))
    img = img.rotate(rng.uniform(-2.5, 2.5), resample=Image.BILINEAR, fillcolor=bg, expand=False)

    scale = height / img.height
    return img.resize((max(1, int(img.width * scale)), height), Image.BILINEAR)


def generate(
    out_dir: Path,
    label_file: Path,
    count: int,
    fonts: list[Path],
    height: int = 32,
    min_len: int = 3,
    max_len: int = 12,
    seed: int = 1234,
    relative_to: Path | None = None,
) -> int:
    """Write `count` crops plus a TSV label file. Returns the number written."""
    rng = random.Random(seed)
    np.random.seed(seed)
    out_dir.mkdir(parents=True, exist_ok=True)
    label_file.parent.mkdir(parents=True, exist_ok=True)
    base = relative_to or label_file.parent

    lines: list[str] = []
    for i in range(count):
        text = random_marking(rng, min_len, max_len)
        img = render(text, rng.choice(fonts) if fonts else None, rng, height)
        name = f"synth_{i:06d}.png"
        img.save(out_dir / name)
        try:
            rel = (out_dir / name).relative_to(base)
        except ValueError:
            rel = out_dir / name
        lines.append(f"{str(rel).replace(chr(92), '/')}\t{text}")

    label_file.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return len(lines)
