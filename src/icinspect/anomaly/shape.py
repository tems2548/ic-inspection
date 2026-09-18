"""What kind of defect is it? Read off the shape of the damaged area.

PatchCore says *whether* a part is defective and *where*; it says nothing
about *what*. But the damage map already carries the answer for the kinds
that matter on a moulded package: a scratch is long and thin, a droplet or
a dent is a compact spot, wear covers most of it, and dust is many tiny
specks. So the map is cut into connected blobs and each blob is measured --
how big, how elongated, how solid, whether it touches the part's edge -- and
a few rules name the kind.

One kind needs more than the map: a chunk broken off the package. Its blob
is not distinctive in shape -- a broken corner reads as a fat diagonal band,
a broken side as an irregular patch -- so two things that only happen when
material is gone are measured instead. Through a broken corner the *fixture
backdrop shows inside the part*; and a broken side, where what shows is the
fracture face rather than the backdrop, makes the *package outline smaller*
than a normal part's, which the package finder reports directly.

Rules rather than a classifier because there are no labels to train one on,
and because a rule can be read: "scratch = elongation over 3" is something
an operator can check against the photo and argue with.

Everything here is numpy; scipy's labeller is used when it is around and a
small flood fill otherwise, so this stays testable on any machine.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# Kinds, in the order they are tried. The label is what the window shows.
KIND_NONE = "none"
KIND_BROKEN = "broken"      # material missing: chipped corner, broken side
KIND_WEAR = "wear"          # most of the part is damaged: abrasion, heavy scratching
KIND_SCRATCH = "scratch"    # one long thin mark
KIND_SCRATCHES = "scratches"
KIND_SPOT = "spot"          # a compact blob: droplet, dent, contamination
KIND_PATCH = "patch"        # an irregular area: dirt, discolouration, abrasion
KIND_SPECKS = "specks"      # many tiny marks: dust, pitting

LABELS = {
    KIND_NONE: "no damage",
    KIND_BROKEN: "chunk missing",
    KIND_WEAR: "surface wear",
    KIND_SCRATCH: "scratch",
    KIND_SCRATCHES: "scratches",
    KIND_SPOT: "spot",
    KIND_PATCH: "patch",
    KIND_SPECKS: "specks",
}

# The numbers behind the rules. Areas are shares of the part, not of the frame.
WEAR_FRAC = 0.30            # this much of the part damaged is wear, whatever its shape
BROKEN_FRAC = 0.02          # a blob this big showing background through it is a chunk gone
BROKEN_BACKGROUND = 0.20    # ... when at least this share of the blob is background
BODY_INTACT = 0.85          # a package outline under this share of a normal one has lost a piece
SCRATCH_ELONGATION = 3.0    # major / minor axis of a scratch (the maps are blurry: real ones read ~4)
SPOT_ELONGATION = 2.0       # a spot is rounder than this ...
SPOT_EXTENT = 0.45          # ... and fills its box at least this well
SPOT_FRAC = 0.05            # ... and is smaller than this
SPECK_FRAC = 0.004          # a blob under this is a speck
PATCH_FRAC = 0.01           # a blob over this is a patch, whatever its shape
IGNORE_FRAC = 0.0005        # blobs under this are map noise and are not counted
RIM_FRAC = 0.10             # a strip along the part's border no deeper than this is the bevel


@dataclass
class Blob:
    """One connected piece of the damaged area, measured."""

    area: int                                   # pixels of the map
    area_frac: float                            # share of the part
    bbox: tuple[int, int, int, int]             # x1, y1, x2, y2 in map pixels
    centroid: tuple[float, float]               # x, y in map pixels
    elongation: float                           # major / minor axis, frame-corrected
    extent: float                               # area / bounding-box area
    length_frac: float                          # major axis as a share of the part's diagonal
    on_edge: bool                               # touches the part's border
    peak: float                                 # hottest score inside, 0..1
    mean: float                                 # mean score inside, 0..1
    background: float = 0.0                     # share of the blob that shows the backdrop
    rim: bool = False                           # a strip along the border: the lit bevel, not damage

    @property
    def elongated(self) -> bool:
        return self.elongation >= SCRATCH_ELONGATION

    @property
    def compact(self) -> bool:
        return self.elongation < SPOT_ELONGATION and self.extent >= SPOT_EXTENT

    @property
    def see_through(self) -> bool:
        """Background shows where package should be: material is missing."""
        return self.background >= BROKEN_BACKGROUND and self.area_frac >= BROKEN_FRAC


@dataclass
class Shape:
    """The answer: a kind, the reasons for it, and the blobs it came from."""

    kind: str
    damage_frac: float                          # share of the part damaged
    blobs: list[Blob] = field(default_factory=list)
    reason: str = ""
    body_frac: float | None = None              # package outline vs a normal part's

    @property
    def label(self) -> str:
        return LABELS.get(self.kind, self.kind)

    @property
    def count(self) -> int:
        return len(self.blobs)

    @property
    def largest(self) -> Blob | None:
        return max(self.blobs, key=lambda b: b.area) if self.blobs else None

    def describe(self) -> str:
        """One line for the window: kind, how much, and why."""
        if self.kind == KIND_NONE:
            return "no damage"
        parts = [self.label, f"{self.damage_frac * 100:.1f}% of the part"]
        if self.count > 1:
            parts.append(f"{self.count} blobs")
        if self.reason:
            parts.append(self.reason)
        return ", ".join(parts)


# -- connected components -------------------------------------------------------


def label_blobs(mask: np.ndarray) -> tuple[np.ndarray, int]:
    """Label 8-connected regions of a boolean mask: (labels, count)."""
    mask = np.asarray(mask, dtype=bool)
    try:
        from scipy import ndimage

        labels, count = ndimage.label(mask, structure=np.ones((3, 3), dtype=int))
        return labels.astype(np.int32), int(count)
    except ImportError:
        return _label_flood(mask)


def _label_flood(mask: np.ndarray) -> tuple[np.ndarray, int]:
    """The fallback labeller: an explicit-stack flood fill, 8-connected."""
    h, w = mask.shape
    labels = np.zeros((h, w), dtype=np.int32)
    count = 0
    for sy, sx in zip(*np.nonzero(mask)):
        if labels[sy, sx]:
            continue
        count += 1
        stack = [(int(sy), int(sx))]
        labels[sy, sx] = count
        while stack:
            y, x = stack.pop()
            for ny in (y - 1, y, y + 1):
                if ny < 0 or ny >= h:
                    continue
                for nx in (x - 1, x, x + 1):
                    if 0 <= nx < w and mask[ny, nx] and not labels[ny, nx]:
                        labels[ny, nx] = count
                        stack.append((ny, nx))
    return labels, count


# -- measuring -------------------------------------------------------------------


def _region_box(shape: tuple[int, int], region: tuple[float, float, float, float] | None):
    """The part's box in map pixels (x1, y1, x2, y2), the whole map if no region."""
    h, w = shape[:2]
    if region is None:
        return 0, 0, w, h
    x1, y1, x2, y2 = region
    return (max(int(np.floor(x1 * w)), 0), max(int(np.floor(y1 * h)), 0),
            min(int(np.ceil(x2 * w)), w), min(int(np.ceil(y2 * h)), h))


def measure(
    labels: np.ndarray,
    count: int,
    heat: np.ndarray | None,
    part_box: tuple[int, int, int, int],
    aspect: float,
    background: np.ndarray | None = None,
) -> list[Blob]:
    """Measure every labelled blob. `aspect` corrects the map's pixels to the
    frame's: the map is square while the photo is not, so a round drop is an
    ellipse in map pixels and would read as elongated without it. `background`
    marks, at map resolution, where the photo shows the fixture backdrop."""
    px1, py1, px2, py2 = part_box
    part_area = max((px2 - px1) * (py2 - py1), 1)
    diagonal = float(np.hypot((px2 - px1) * aspect, py2 - py1)) or 1.0
    scores = None if heat is None else np.asarray(heat, dtype=np.float32) / 255.0
    blobs: list[Blob] = []
    for index in range(1, count + 1):
        ys, xs = np.nonzero(labels == index)
        area = int(xs.size)
        if not area:
            continue
        x1, x2 = int(xs.min()), int(xs.max()) + 1
        y1, y2 = int(ys.min()), int(ys.max()) + 1
        # Principal axes of the pixel cloud, in frame-shaped coordinates. The
        # covariance of a filled shape gives its axis lengths up to a constant,
        # so the ratio is the elongation and the major axis its length.
        fx = xs.astype(np.float64) * aspect
        fy = ys.astype(np.float64)
        if area > 2:
            cov = np.cov(np.stack([fx, fy]))
            eig = np.sort(np.linalg.eigvalsh(cov))[::-1]
            major = 4.0 * float(np.sqrt(max(eig[0], 0.0)))
            minor = 4.0 * float(np.sqrt(max(eig[1], 0.0)))
        else:
            major = minor = 1.0
        # A one-pixel-wide line has zero minor variance; a pixel is a unit wide.
        elongation = major / max(minor, 1.0)
        extent = area / float((x2 - x1) * (y2 - y1))
        edge = x1 <= px1 + 1 or y1 <= py1 + 1 or x2 >= px2 - 1 or y2 >= py2 - 1
        # The lit bevel of the package scores as anomalous too, as a thin strip
        # hugging the border of the region. It is not damage and is not counted.
        strip = ((y1 <= py1 + 1 or y2 >= py2 - 1) and (y2 - y1) <= RIM_FRAC * (py2 - py1)) or \
                ((x1 <= px1 + 1 or x2 >= px2 - 1) and (x2 - x1) <= RIM_FRAC * (px2 - px1))
        rim = strip and elongation >= SCRATCH_ELONGATION
        inside = None if scores is None else scores[ys, xs]
        blobs.append(Blob(
            area=area,
            area_frac=round(area / part_area, 5),
            bbox=(x1, y1, x2, y2),
            centroid=(float(xs.mean()), float(ys.mean())),
            elongation=round(elongation, 2),
            extent=round(extent, 3),
            length_frac=round(max(major, 1.0) / diagonal, 4),
            on_edge=bool(edge),
            peak=0.0 if inside is None else round(float(inside.max()), 4),
            mean=0.0 if inside is None else round(float(inside.mean()), 4),
            background=0.0 if background is None else round(float(background[ys, xs].mean()), 4),
            rim=bool(rim),
        ))
    return blobs


def background_mask(photo, shape: tuple[int, int], min_contrast: float = 0.05) -> np.ndarray:
    """Where the photo shows the fixture backdrop, at the map's resolution.

    The same reading of the frame as the package finder: the border of the
    frame is backdrop, and anything within `min_contrast` of its brightness
    is backdrop too. Inside the part that should be nowhere; where it is, the
    part is not there any more.
    """
    from PIL import Image

    gray = photo.convert("L").resize((shape[1], shape[0]), Image.BILINEAR)
    arr = np.asarray(gray, dtype=np.float32)
    h, w = arr.shape
    edge = max(1, int(0.04 * min(h, w)))
    border = np.concatenate(
        [arr[:edge].ravel(), arr[-edge:].ravel(), arr[:, :edge].ravel(), arr[:, -edge:].ravel()]
    )
    return np.abs(arr - float(np.median(border))) / 255.0 < float(min_contrast)


# -- naming ------------------------------------------------------------------------


def classify(
    blobs: list[Blob], damage_frac: float, body_frac: float | None = None
) -> tuple[str, str]:
    """Name the kind from the measured blobs: (kind, reason).

    `body_frac` is the package outline's area as a share of a normal part's,
    when the finder measured one; under `BODY_INTACT`, with the damage on the
    part's edge, a piece of it is gone even if nothing shows through.
    """
    counted = [b for b in blobs if b.area_frac >= IGNORE_FRAC and not b.rim]
    if not counted:
        return KIND_NONE, ""
    largest = max(counted, key=lambda b: b.area)
    gone = [b for b in counted if b.see_through]
    if gone:
        missing = sum(b.area_frac * b.background for b in gone)
        return KIND_BROKEN, f"backdrop shows through {missing * 100:.0f}% of the part"
    if body_frac is not None and body_frac < BODY_INTACT and largest.on_edge:
        return KIND_BROKEN, f"the package outline is {body_frac * 100:.0f}% of a normal part's"
    if damage_frac >= WEAR_FRAC:
        return KIND_WEAR, f"{damage_frac * 100:.0f}% of the part is over the cutoff"

    long = [b for b in counted if b.elongated and b.area_frac >= SPECK_FRAC]

    if len(long) >= 2:
        return KIND_SCRATCHES, f"{len(long)} long thin marks"
    if len(long) == 1 and long[0] is largest:
        return KIND_SCRATCH, (f"{long[0].elongation:.1f}x longer than wide, "
                              f"{long[0].length_frac * 100:.0f}% of the part's diagonal")
    if largest.area_frac >= PATCH_FRAC and not largest.compact:
        return KIND_PATCH, f"irregular area, {largest.area_frac * 100:.1f}% of the part"
    if largest.area_frac >= SPECK_FRAC and largest.compact and largest.area_frac < SPOT_FRAC:
        return KIND_SPOT, f"compact, {largest.area_frac * 100:.1f}% of the part"
    if largest.area_frac >= SPECK_FRAC:
        return KIND_PATCH, f"{largest.area_frac * 100:.1f}% of the part"
    return KIND_SPECKS, f"{len(counted)} tiny mark(s), largest {largest.area_frac * 100:.2f}% of the part"


def body_fraction(photo, region: tuple[float, float, float, float] | None, cfg: dict | None = None) -> float | None:
    """The package outline's area as a share of the normal part's `region`.

    The finder measures what is actually there; on a part with a piece gone it
    reports a smaller body. None when there is no region to compare against
    or no body to be found.
    """
    if region is None:
        return None
    from icinspect.ocr.segment import find_chip

    box = find_chip(photo, cfg)
    if box is None:
        return None
    width, height = photo.size
    normal = (region[2] - region[0]) * (region[3] - region[1]) * width * height
    if normal <= 0:
        return None
    return round((box[2] - box[0]) * (box[3] - box[1]) / normal, 4)


def analyse(
    damage: np.ndarray | None,
    heat: np.ndarray | None,
    region: tuple[float, float, float, float] | None,
    frame_aspect: float | None = None,
    photo=None,
    body_frac: float | None = None,
) -> Shape:
    """Name the defect on one photo from its damage mask.

    `damage` is the boolean map of the area that failed (`heatmap.damage_mask`),
    `heat` the score map it came from (uint8), `region` the part as fractions
    of the frame, and `frame_aspect` the photo's width / height, which corrects
    the map's square pixels to the photo's real proportions. `photo` (a PIL
    image) and `body_frac` (the package outline's area as a share of a normal
    part's) are what tell a chunk broken off from a dirty patch; without them
    that kind is never reported.
    """
    if damage is None:
        return Shape(KIND_NONE, 0.0)
    mask = np.asarray(damage, dtype=bool)
    if mask.ndim != 2 or not mask.any():
        return Shape(KIND_NONE, 0.0)
    part_box = _region_box(mask.shape, region)
    px1, py1, px2, py2 = part_box
    part_area = max((px2 - px1) * (py2 - py1), 1)
    damage_frac = round(float(mask.sum()) / part_area, 4)

    map_aspect = mask.shape[1] / mask.shape[0]
    aspect = 1.0 if frame_aspect is None else float(frame_aspect) / map_aspect

    if photo is not None and frame_aspect is None:
        aspect = (photo.size[0] / photo.size[1]) / map_aspect
    backdrop = None if photo is None else background_mask(photo, mask.shape)
    labels, count = label_blobs(mask)
    blobs = measure(labels, count, heat, part_box, aspect, backdrop)
    blobs.sort(key=lambda b: -b.area)
    kind, reason = classify(blobs, damage_frac, body_frac)
    return Shape(kind, damage_frac,
                 [b for b in blobs if b.area_frac >= IGNORE_FRAC and not b.rim], reason, body_frac)
