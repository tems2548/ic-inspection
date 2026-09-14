"""Label-file dataset for single-line marking crops.

Label file format (UTF-8, one sample per line):

    images/u17_top.png<TAB>STM32F407VGT6

The image path is relative to the OCR root (the label file's own folder is also
tried), so the dataset stays movable on the Desktop.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageFilter
from torch.utils.data import Dataset

from icinspect.ocr.charset import Charset

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


@dataclass(frozen=True)
class Sample:
    image_path: Path
    text: str


def read_label_file(label_file: Path, roots: list[Path] | None = None) -> list[Sample]:
    """Parse a TSV label file. Missing images and blank lines are skipped."""
    label_file = Path(label_file)
    if not label_file.exists():
        raise FileNotFoundError(f"Label file not found: {label_file}")

    search_roots = [label_file.parent, *(roots or [])]
    samples: list[Sample] = []
    skipped: list[str] = []

    for lineno, raw in enumerate(label_file.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw.rstrip("\n")
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        # Tab is the documented separator; a single space is tolerated as a fallback.
        # Split on the FIRST space: labels can contain spaces (date codes), paths rarely do.
        parts = line.split("\t", 1) if "\t" in line else line.split(" ", 1)
        if len(parts) != 2:
            skipped.append(f"line {lineno}: no <path>TAB<label> separator")
            continue
        rel, text = parts[0].strip(), parts[1].strip()
        if not text:
            skipped.append(f"line {lineno}: empty label")
            continue

        candidate = Path(rel)
        resolved = candidate if candidate.is_absolute() else None
        if resolved is None:
            for root in search_roots:
                if (root / candidate).exists():
                    resolved = root / candidate
                    break
        if resolved is None or not resolved.exists():
            skipped.append(f"line {lineno}: image not found ({rel})")
            continue
        samples.append(Sample(image_path=resolved, text=text))

    if skipped:
        print(f"[ocr] {label_file.name}: skipped {len(skipped)} line(s)")
        for msg in skipped[:5]:
            print(f"      {msg}")
    if not samples:
        raise ValueError(f"No usable samples in {label_file}")
    return samples


def filter_encodable(samples: list[Sample], charset: Charset) -> list[Sample]:
    """Drop labels containing characters the model cannot output."""
    keep, bad_chars = [], set()
    for s in samples:
        if charset.is_encodable(s.text):
            keep.append(s)
        else:
            bad_chars |= charset.unknown_chars(s.text)
    if bad_chars:
        print(
            f"[ocr] dropped {len(samples) - len(keep)} sample(s) with out-of-charset "
            f"characters {sorted(bad_chars)!r} -- add them to configs/ocr.yaml if needed"
        )
    return keep


def fit_image(img: Image.Image, height: int, width: int, grayscale: bool = True) -> Image.Image:
    """Resize preserving aspect ratio to `height`, then pad/crop to `width`.

    Takes an already-open image so in-memory crops (see `ocr.segment`) go
    through exactly the same normalisation as files on disk.
    """
    img = img.convert("L" if grayscale else "RGB")
    w, h = img.size
    new_w = max(1, min(width, round(w * height / max(h, 1))))
    img = img.resize((new_w, height), Image.BILINEAR)
    if new_w == width:
        return img
    # Pad on the right with the image's own border tone so the model does not
    # learn a hard black edge as an end-of-text cue.
    arr = np.asarray(img)
    if grayscale:
        pad_value: int | tuple[int, ...] = int(np.median(arr[:, -1]))
    else:
        pad_value = tuple(int(v) for v in np.median(arr[:, -1], axis=0))
    canvas = Image.new(img.mode, (width, height), pad_value)
    canvas.paste(img, (0, 0))
    return canvas


def load_image(path: Path, height: int, width: int, grayscale: bool = True) -> Image.Image:
    """`fit_image` for a file on disk."""
    return fit_image(Image.open(path), height, width, grayscale)


def augment(img: Image.Image, cfg: dict, rng: random.Random) -> Image.Image:
    a = cfg.get("augment", {})
    if not a.get("enabled", True):
        return img

    rot = float(a.get("rotate_deg", 0.0))
    if rot > 0:
        px = np.asarray(img)
        fill = int(np.median(px)) if img.mode == "L" else tuple(int(v) for v in np.median(px, axis=(0, 1)))
        img = img.rotate(rng.uniform(-rot, rot), resample=Image.BILINEAR, fillcolor=fill)

    if rng.random() < float(a.get("blur_prob", 0.0)):
        img = img.filter(ImageFilter.GaussianBlur(radius=rng.uniform(0.3, 1.1)))

    arr = np.asarray(img, dtype=np.float32) / 255.0
    b, c = float(a.get("brightness", 0.0)), float(a.get("contrast", 0.0))
    if c > 0:
        factor = 1.0 + rng.uniform(-c, c)
        arr = (arr - arr.mean()) * factor + arr.mean()
    if b > 0:
        arr = arr + rng.uniform(-b, b)
    noise = float(a.get("noise_std", 0.0))
    if noise > 0:
        arr = arr + np.random.normal(0.0, noise, arr.shape).astype(np.float32)
    return Image.fromarray((np.clip(arr, 0.0, 1.0) * 255).astype(np.uint8), mode=img.mode)


class MarkingDataset(Dataset):
    """Yields (image_tensor, encoded_label, raw_text)."""

    def __init__(
        self,
        samples: list[Sample],
        charset: Charset,
        cfg: dict,
        train: bool = False,
        seed: int = 42,
    ) -> None:
        self.samples = samples
        self.charset = charset
        self.cfg = cfg
        self.train = train
        img_cfg = cfg.get("image", {})
        self.height = int(img_cfg.get("height", 32))
        self.width = int(img_cfg.get("width", 160))
        self.grayscale = bool(img_cfg.get("grayscale", True))
        self._rng = random.Random(seed)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        s = self.samples[idx]
        img = load_image(s.image_path, self.height, self.width, self.grayscale)
        if self.train:
            img = augment(img, self.cfg, self._rng)
        arr = np.asarray(img, dtype=np.float32) / 255.0
        if arr.ndim == 2:
            arr = arr[None, ...]                    # (1, H, W)
        else:
            arr = arr.transpose(2, 0, 1)            # (3, H, W)
        tensor = torch.from_numpy((arr - 0.5) / 0.5)
        target = torch.tensor(self.charset.encode(s.text), dtype=torch.long)
        return tensor, target, s.text


def collate(batch):
    """Pack variable-length CTC targets into one flat tensor."""
    images, targets, texts = zip(*batch)
    lengths = torch.tensor([t.numel() for t in targets], dtype=torch.long)
    return torch.stack(images, dim=0), torch.cat(targets), lengths, list(texts)


def split_samples(samples: list[Sample], val_ratio: float, seed: int) -> tuple[list[Sample], list[Sample]]:
    if val_ratio <= 0 or len(samples) < 2:
        return samples, []
    shuffled = list(samples)
    random.Random(seed).shuffle(shuffled)
    n_val = max(1, int(round(len(shuffled) * val_ratio)))
    n_val = min(n_val, len(shuffled) - 1)
    return shuffled[n_val:], shuffled[:n_val]


def list_images(path: Path) -> list[Path]:
    path = Path(path)
    if path.is_file():
        return [path]
    return sorted(p for p in path.rglob("*") if p.suffix.lower() in IMAGE_EXTS)
