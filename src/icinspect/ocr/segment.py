"""Split a whole IC photo into single-line marking crops.

The CRNN reads one text line at a time, but a real top marking is two or three
stacked lines (part number / date code / lot code). This module finds those
lines automatically so `icinspect.ocr.infer` can take a photo of the chip
instead of pre-cut crops.

A photo of a chip on the fixture is mostly background, so `split_image` first
locates the package itself (`find_chip`) and works inside that crop; the boxes
it reports are still in source-image coordinates.

Two modes:

* ``auto`` -- horizontal projection profile over an illumination-corrected ink
  map. Handles light-on-dark (laser-etched) and dark-on-light markings alike,
  because "ink" is measured as deviation from the local background rather than
  as darkness.
* ``half`` -- a fixed top/bottom split, each strip then tightened onto its own
  ink. Use it when you know the marking is exactly two lines, or when package
  texture makes the projection profile find lines that are not there.

Nothing here needs torch; it is PIL + numpy only, so it stays testable on any
machine.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

MODES = ("auto", "half", "none")

# The outermost rows/columns of a crop carry its boundary -- the moulding rim,
# or whatever the crop cut through. They are still searched for text, but they
# do not get a vote on where the threshold sits, because one saturated edge
# streak would otherwise push every real line below the cut.
EDGE_STAT_FRAC = 0.02

# A lit bevel of the moulding shows up as a solid streak flush against the crop
# border, far brighter than any text row. It can reach this deep into the crop
# (fraction of the dimension) and still be treated as border, and anything in
# that zone brighter than the interior's brightest row by this factor is one.
EDGE_STREAK_FRAC = 0.15
EDGE_STREAK_FACTOR = 1.2

# Defaults for the `segment:` block of configs/ocr.yaml.
DEFAULTS = {
    "mode": "auto",
    "max_lines": 0,          # 0 = keep every line found
    "ink_frac": 0.02,        # minimum coverage that counts as content, and the flat-profile floor
    "line_threshold": 0.80,  # fraction of the automatic (Otsu) profile level a line row must reach
    "line_hysteresis": 0.85, # a found line grows while rows stay above this fraction of the threshold
    "smooth_frac": 0.01,     # projection-profile smoothing window, fraction of the dimension
    "min_contrast": 0.05,    # a stroke must beat the local background by this much of full scale
    "stroke_frac": 0.01,     # ink-density window radius, fraction of H (0 = use the raw ink pixels)
    "stroke_density": 0.15,  # the least share of ink in that window that still counts as a stroke
    "min_line_frac": 0.05,   # bands thinner than this fraction of H are noise
    "thin_line_frac": 0.70,  # a line under this share of a full line's height is grown to one
    "merge_gap_frac": 0.05,  # blank runs shorter than this join two bands (faint text profiles are patchy)
    "split_gap_frac": 0.35,  # cut a line at gaps wider than this many line heights (0 = never)
    "min_part_frac": 0.50,   # a segment narrower than this x its height is a blob, not a character
    "edge_mark_frac": 0.75,  # ...and on the crop border, narrower than this is the dimple or the bevel
    "pad_frac": 0.25,        # vertical padding, as a fraction of band height
    "background_blur": 0.06, # illumination-estimate blur radius, fraction of H
    "trim_columns": True,    # also cut leading/trailing blank columns
    "chip_detect": True,     # crop to the package before looking for text lines
    "chip_body_frac": 0.75,  # a body row/column is this full of foreground (~0.97); a lead row is ~0.55
    "chip_inset": 0.08,      # pull the crop in off the moulding edge, fraction of the box
}


@dataclass(frozen=True)
class Line:
    """One detected text line: its box in the source image, plus the crop."""

    index: int
    box: tuple[int, int, int, int]   # (x1, y1, x2, y2) in source pixels
    image: Image.Image
    part: int = 0                    # segment within the line, split at wide gaps

    @property
    def box_str(self) -> str:
        return "{},{},{},{}".format(*self.box)

    @property
    def name(self) -> str:
        return f"l{self.index}p{self.part}"


def segment_config(cfg: dict | None) -> dict:
    """Merge the `segment:` block of an ocr.yaml over the defaults."""
    merged = dict(DEFAULTS)
    merged.update((cfg or {}).get("segment", {}) or {})
    return merged


def ink_map(img: Image.Image, background_blur: float = DEFAULTS["background_blur"]) -> np.ndarray:
    """Deviation from the local background, as a fraction of full scale (0..1).

    Subtracting a heavily blurred copy removes vignetting and the uneven
    illumination that ring lights put on a shiny package, and it makes the
    result polarity-agnostic: white-on-black text scores exactly like
    black-on-white. Scaling by 255 rather than by the image's own peak keeps
    the numbers absolute, so a blank package stays near zero instead of having
    its noise stretched to look like ink.
    """
    gray = img.convert("L")
    arr = np.asarray(gray, dtype=np.float32)
    radius = max(1.0, float(background_blur) * arr.shape[0])
    bg = np.asarray(gray.filter(ImageFilter.GaussianBlur(radius)), dtype=np.float32)
    return np.abs(arr - bg) / 255.0


def _otsu(values: np.ndarray, bins: int = 64) -> float:
    """Otsu's threshold over 0..1 data, histogram-based (no scipy/skimage)."""
    hist, edges = np.histogram(values, bins=bins, range=(0.0, 1.0))
    total = int(hist.sum())
    if total == 0:
        return 0.0
    centers = (edges[:-1] + edges[1:]) / 2.0
    prob = hist / total
    w0 = np.cumsum(prob)                    # class weights, as probabilities
    w1 = 1.0 - w0
    m0 = np.cumsum(prob * centers)
    m_total = float(m0[-1])

    usable = (w0 > 0) & (w1 > 0)
    if not usable.any():                    # single-valued image: no split exists
        return 0.0
    between = np.zeros(bins, dtype=np.float64)
    between[usable] = (m_total * w0[usable] - m0[usable]) ** 2 / (w0[usable] * w1[usable])
    return float(centers[int(np.argmax(between))])


def text_mask(ink: np.ndarray, min_contrast: float = DEFAULTS["min_contrast"]) -> np.ndarray:
    """Binarise the ink map at Otsu's threshold, floored at `min_contrast`.

    Counting ink *pixels* per row, rather than summing ink *intensity*, is what
    keeps a bold part-number line from drowning out the thinner date-code line
    below it, and it puts blank package on a hard zero -- so a row is text at an
    absolute coverage, with no dependence on how bold the boldest line is.

    Otsu always finds a split, even in an unmarked package's sensor noise. The
    contrast floor is what says "there is nothing here to read".
    """
    return ink >= max(_otsu(ink), float(min_contrast))


def stroke_mask(img: Image.Image, cfg: dict | None = None) -> np.ndarray:
    """Where the strokes are: the text mask, filled in at stroke scale.

    A laser-etched marking is not solid ink -- it is a roughened patch of the
    moulding, and the pixel mask inside a stroke is as speckled as the blank
    package around it, only denser. Counting those pixels row by row leaves a
    faint marking barely above the speckle floor, and two lines with a shallow
    valley between them merge. So the mask is turned into a local ink density
    (a box blur about one stroke wide) and thresholded again: a stroke reads
    as solid, isolated speckle drops out, and the projection profile gets
    valleys near zero. Otsu picks the split, floored at `stroke_density` so an
    unmarked package -- whose density map has nothing to separate -- stays
    empty. `stroke_frac: 0` skips the filling and returns the raw text mask.
    """
    s = segment_config(cfg)
    mask = text_mask(ink_map(img, s["background_blur"]), s["min_contrast"])
    radius = float(s["stroke_frac"]) * mask.shape[0]
    if radius <= 0:
        return mask
    density = np.asarray(
        Image.fromarray(mask.astype(np.uint8) * 255).filter(ImageFilter.BoxBlur(radius)),
        dtype=np.float32,
    ) / 255.0
    return density >= max(_otsu(density), float(s["stroke_density"]))


def _bands(
    profile: np.ndarray,
    threshold: float,
    min_len: int,
    merge_gap: int,
    low: float | None = None,
) -> list[tuple[int, int]]:
    """Contiguous runs at or above `threshold`, gap-merged then length-filtered.

    With `low`, a run is seeded at `threshold` and then grown outward for as
    long as the profile stays at or above `low` (hysteresis). A faint line next
    to a bold one only clears the threshold at its densest rows -- the middle
    bar of an E, the waist of a 5 -- and without the growth it comes back as a
    thin slice through the characters instead of the whole line.
    """
    hot = profile >= threshold
    warm = hot if low is None else (profile >= min(low, threshold))

    runs: list[list[int]] = []
    start: int | None = None
    for i, on in enumerate(warm):
        if on and start is None:
            start = i
        elif not on and start is not None:
            if hot[start:i].any():
                runs.append([start, i])
            start = None
    if start is not None and hot[start:].any():
        runs.append([start, len(hot)])

    merged: list[list[int]] = []
    for run in runs:
        if merged and run[0] - merged[-1][1] <= merge_gap:
            merged[-1][1] = run[1]
        else:
            merged.append(run)
    return [(a, b) for a, b in merged if b - a >= min_len]


def _stat_interior(profile: np.ndarray) -> np.ndarray:
    """The part of a profile that gets a vote on where the threshold sits.

    The fixed edge margin goes first. Then, within the outer `EDGE_STREAK_FRAC`
    of either end, everything from the border up to the innermost row that is
    brighter than any row in the middle of the profile is a streak, not text,
    and goes too -- the whole run, so a streak with a dip in it does not leave
    its inner half behind. A text line that sits in that zone and beats the
    middle is excluded from the statistics as well; that only lowers the
    threshold a little, and the line itself still gets thresholded.
    """
    n = len(profile)
    edge = int(EDGE_STAT_FRAC * n)
    zone = int(EDGE_STREAK_FRAC * n)
    if not zone or n <= 2 * zone:
        return profile[edge : n - edge] if edge and n > 2 * edge else profile
    bright = float(profile[zone : n - zone].max()) * EDGE_STREAK_FACTOR
    start, stop = edge, n - edge
    hot = np.flatnonzero(profile[:zone] > bright)
    if hot.size:
        start = max(start, int(hot[-1]) + 1)
    hot = np.flatnonzero(profile[n - zone :] > bright)
    if hot.size:
        stop = min(stop, n - zone + int(hot[0]))
    return profile[start:stop] if stop > start else profile[zone : n - zone]


def _profile_bands(
    profile: np.ndarray,
    bias: float,
    ink_frac: float,
    min_len: int,
    merge_gap: int,
    smooth: int = 1,
    hysteresis: float = 1.0,
) -> list[tuple[int, int]]:
    """Bands of a projection profile, thresholded relative to the profile itself.

    A moulded package is speckled, so on a real photo *every* row carries some
    ink -- text rows just carry more (~0.30 coverage against ~0.15 for blank
    package). A fixed threshold cannot serve both that and a synthetic crop
    whose background is a hard zero, so the level comes from Otsu over the
    profile's own range.

    Otsu sits a little high for this: text rows are a large minority class and
    the faint last line (date/lot code) falls under the cut. `bias` scales the
    level down to take the fainter line's shoulders with it, and `hysteresis`
    (a fraction of the biased level, 1.0 = off) lets a band that did clear the
    cut grow out to where the profile really falls away.

    A bright, near-solid streak flush against the crop border -- the lit bevel
    of the package, which `chip_inset` does not always clear -- has higher row
    coverage than any text line and would pull the threshold above the real
    lines. Such streaks are kept out of the statistics (see `_stat_interior`),
    and the band they still form at the border is dropped for the same reason:
    it is brighter than any row that was counted.
    """
    if smooth > 1:
        # Package speckle makes single rows jump around; without this the noise
        # bridges the shallow valley between two lines and merges them.
        pad = smooth // 2
        window = np.ones(smooth, dtype=np.float32) / smooth
        profile = np.convolve(np.pad(profile, pad, mode="edge"), window, mode="valid")[: len(profile)]
    interior = _stat_interior(profile)
    lo, hi = float(interior.min()), float(interior.max())
    if hi - lo < ink_frac:          # a profile this flat has no line structure
        return []
    threshold = _otsu((interior - lo) / (hi - lo)) * float(bias)
    bands = _bands((profile - lo) / (hi - lo), threshold, min_len, merge_gap,
                   low=threshold * float(hysteresis))
    # The streak that was kept out of the statistics still clears the cut, as
    # a band flush against the border. It is the one band brighter than any
    # row that had a vote, which is how it is told from a text line there.
    streak = EDGE_STREAK_FACTOR * hi
    return [
        (a, b) for a, b in bands
        if not ((a == 0 or b == len(profile)) and float(profile[a:b].mean()) > streak)
    ]


def _column_segments(cols: list[tuple[int, int]], gap: int) -> list[tuple[int, int]]:
    """Group column runs into segments, cutting where a gap is wider than `gap`.

    A logo sits in the same row band as the text beside it and decodes to junk.
    Cutting the band at the wide gap that separates them gives the logo its own
    crop, which the reader can then reject whole instead of interleaving its
    characters with the real ones -- and it stops the logo stealing width from
    the text when the crop is squeezed to the model's fixed input.

    `gap <= 0` disables the split and returns one span, as before.
    """
    if not cols:
        return []
    if gap <= 0:
        return [(cols[0][0], cols[-1][1])]

    segments = [[cols[0][0], cols[0][1]]]
    for start, end in cols[1:]:
        if start - segments[-1][1] > gap:
            segments.append([start, end])
        else:
            segments[-1][1] = end
    return [(a, b) for a, b in segments]


def auto_lines(img: Image.Image, cfg: dict | None = None) -> list[Line]:
    """Find text lines by horizontal projection. Empty list if none stand out."""
    s = segment_config(cfg)
    ink = stroke_mask(img, cfg).astype(np.float32)
    h, w = ink.shape
    ink_frac, bias = float(s["ink_frac"]), float(s["line_threshold"])
    rows = _profile_bands(
        ink.mean(axis=1),
        bias=bias,
        ink_frac=ink_frac,
        min_len=max(4, int(float(s["min_line_frac"]) * h)),
        merge_gap=max(1, int(float(s["merge_gap_frac"]) * h)),
        smooth=max(1, int(float(s["smooth_frac"]) * h)),
        hysteresis=float(s["line_hysteresis"]),
    )
    # The moulding rim survives cropping as a thin streak flush against the crop
    # border. Drop that, but only when it is much thinner than the real lines --
    # a text line can legitimately reach the border on a tightly cropped chip.
    if len(rows) > 1:
        typical = float(np.median([b[1] - b[0] for b in rows]))
        rows = [
            b for b in rows
            if not ((b[0] == 0 or b[1] == h) and (b[1] - b[0]) < 0.5 * typical)
        ] or rows
    rows = _grow_thin_bands(rows, h, float(s["thin_line_frac"]))
    if not rows:
        return []

    max_lines = int(s["max_lines"] or 0)
    if max_lines > 0 and len(rows) > max_lines:
        # Keep the strongest bands (most ink), then restore top-to-bottom order.
        rows = sorted(sorted(rows, key=lambda b: -float(ink[b[0]:b[1]].sum()))[:max_lines])

    lines: list[Line] = []
    for i, (y1, y2) in enumerate(rows):
        pad = int(float(s["pad_frac"]) * (y2 - y1))
        top, bottom = max(0, y1 - pad), min(h, y2 + pad)

        spans: list[tuple[int, int]] = [(0, w)]
        if s["trim_columns"]:
            # A column only has to be ink or blank here, but its coverage runs
            # from a bowl's two thin arcs (~0.2 of the line height) to a
            # full-height stem (~1.0), and Otsu over that range lands above the
            # bowls and cuts a D or an O off its word. The square root pulls
            # the stems down towards the bowls and leaves blank at zero.
            cols = _profile_bands(
                np.sqrt(ink[y1:y2].mean(axis=0)),
                bias=bias,
                ink_frac=ink_frac,
                min_len=1,
                merge_gap=max(2, int(0.05 * w)),   # keep gaps between characters
                smooth=max(1, int(float(s["smooth_frac"]) * w)),
            )
            if cols:
                spans = _column_segments(cols, gap=int(float(s["split_gap_frac"]) * (y2 - y1)))

        boxes = [(max(0, x1 - pad), min(w, x2 + pad)) for x1, x2 in spans]
        # Moulding gets cut off as segments of its own: the pin-1 dimple and
        # the bevel reaching in from the crop border, and the odd blob of
        # texture beside a word. None of it is as wide as a character. Judged
        # on the padded box, a lone "1" comes out at about half its height; a
        # blob is well under that, and a dimple on the border -- where no
        # character of a centred marking sits -- under three quarters. Left
        # in, one decodes to a confident digit often enough to matter.
        if len(boxes) > 1:
            height = bottom - top
            blob = float(s["min_part_frac"]) * height
            edge_mark = float(s["edge_mark_frac"]) * height
            kept = [
                (left, right) for left, right in boxes
                if right - left >= blob
                and not ((left == 0 or right == w) and right - left < edge_mark)
            ]
            boxes = kept or boxes
        for part, (left, right) in enumerate(boxes):
            lines.append(
                Line(i, (left, top, right, bottom), img.crop((left, top, right, bottom)), part=part)
            )
    return lines


def _grow_thin_bands(rows: list[tuple[int, int]], h: int, frac: float) -> list[tuple[int, int]]:
    """Give a band that came out much thinner than the others a full line's height.

    The lines of a marking share one font, so they share one height. A band
    under `frac` of it is a faint line seen only at its densest rows -- the
    middle bar of the E, the waist of the 5 -- and a crop of that reads as
    nothing. It is widened about its centre to the height of a full line,
    stopping short of the crop's ends and of its neighbours. The full height
    is the median of the taller half of the bands, so a merged pair does not
    set it and a lone thin band beside one full line takes that line's.
    """
    if len(rows) < 2 or frac <= 0:
        return rows
    heights = sorted(b - a for a, b in rows)
    full = float(np.median(heights[len(heights) // 2:]))
    grown: list[tuple[int, int]] = []
    for index, (a, b) in enumerate(rows):
        if b - a >= frac * full:
            grown.append((a, b))
            continue
        centre = (a + b) / 2.0
        top, bottom = int(round(centre - full / 2)), int(round(centre + full / 2))
        floor = rows[index - 1][1] if index else 0
        ceiling = rows[index + 1][0] if index + 1 < len(rows) else h
        grown.append((max(top, floor, 0), min(bottom, ceiling, h)))
    return grown


def ink_box(img: Image.Image, cfg: dict | None = None) -> tuple[int, int, int, int] | None:
    """Bounding box of the ink in `img`, padded. None when there is none."""
    s = segment_config(cfg)
    ink = stroke_mask(img, cfg).astype(np.float32)
    h, w = ink.shape
    ink_frac, bias = float(s["ink_frac"]), float(s["line_threshold"])
    rows = _profile_bands(ink.mean(axis=1), bias, ink_frac, min_len=1, merge_gap=max(1, int(0.05 * h)))
    cols = _profile_bands(ink.mean(axis=0), bias, ink_frac, min_len=1, merge_gap=max(2, int(0.05 * w)))
    if not rows or not cols:
        return None
    pad = int(float(s["pad_frac"]) * (rows[-1][1] - rows[0][0]))
    return (
        max(0, cols[0][0] - pad),
        max(0, rows[0][0] - pad),
        min(w, cols[-1][1] + pad),
        min(h, rows[-1][1] + pad),
    )


def _widest_band(profile: np.ndarray, threshold: float) -> tuple[int, int] | None:
    """The longest contiguous run at or above `threshold`."""
    runs = _bands(profile, threshold, min_len=1, merge_gap=0)
    return max(runs, key=lambda r: r[1] - r[0]) if runs else None


def find_chip(img: Image.Image, cfg: dict | None = None, work_width: int = 512) -> tuple[int, int, int, int] | None:
    """Locate the package body in a photo of a chip on a plain background.

    Two passes, both on a downscaled copy:

    1. Everything that differs from the *frame border* (which the fixture keeps
       empty) is foreground. Its widest row/column runs bound the whole part --
       package plus the leads sticking out top and bottom.
    2. Inside that box, only rows/columns the foreground nearly fills are the
       moulded body: a row of four leads covers ~40% of the width, the body
       covers all of it. That drops the leads without any shape modelling.

    Returns a box in source-image pixels, or None when the result is not
    plausibly a chip (under 2% or over 98% of the frame -- no part in view, or
    an image that is already a tight crop).
    """
    s = segment_config(cfg)
    gray = img.convert("L")
    src_w, src_h = gray.size
    scale = max(1.0, src_w / float(work_width))
    if scale > 1.0:
        gray = gray.resize((max(1, round(src_w / scale)), max(1, round(src_h / scale))), Image.BILINEAR)

    arr = np.asarray(gray, dtype=np.float32)
    h, w = arr.shape
    edge = max(1, int(0.04 * min(h, w)))
    border = np.concatenate(
        [arr[:edge].ravel(), arr[-edge:].ravel(), arr[:, :edge].ravel(), arr[:, -edge:].ravel()]
    )
    dev = np.abs(arr - float(np.median(border))) / 255.0
    mask = (dev >= max(_otsu(dev), float(s["min_contrast"]))).astype(np.float32)

    ink_frac = float(s["ink_frac"])
    rough_y = _widest_band(mask.mean(axis=1), ink_frac)
    rough_x = _widest_band(mask.mean(axis=0), ink_frac)
    if rough_y is None or rough_x is None:
        return None

    inner = mask[rough_y[0]:rough_y[1], rough_x[0]:rough_x[1]]
    body_frac = float(s["chip_body_frac"])
    body_y = _widest_band(inner.mean(axis=1), body_frac) or (0, inner.shape[0])
    body_x = _widest_band(inner.mean(axis=0), body_frac) or (0, inner.shape[1])

    y1, y2 = rough_y[0] + body_y[0], rough_y[0] + body_y[1]
    x1, x2 = rough_x[0] + body_x[0], rough_x[0] + body_x[1]

    # Pull in off the moulding edge: against a bright background that edge is the
    # strongest gradient in the frame and would read as a text line of its own.
    inset = float(s["chip_inset"])
    x1 += int(inset * (x2 - x1))
    x2 -= int(inset * (x2 - x1))
    y1 += int(inset * (y2 - y1))
    y2 -= int(inset * (y2 - y1))
    if x2 <= x1 or y2 <= y1:
        return None

    area = ((x2 - x1) * (y2 - y1)) / float(w * h)
    if not 0.02 <= area <= 0.98:
        return None

    box = (
        max(0, int(x1 * scale)),
        max(0, int(y1 * scale)),
        min(src_w, int(round(x2 * scale))),
        min(src_h, int(round(y2 * scale))),
    )
    return box if box[2] > box[0] and box[3] > box[1] else None


def half_lines(img: Image.Image, count: int = 2, cfg: dict | None = None) -> list[Line]:
    """Cut the image into `count` equal horizontal strips, top first.

    Each strip is then tightened onto its own ink. Without that the CRNN gets a
    strip that is mostly blank package, and squeezing it to 32 px high leaves
    the text too small to read -- the fixed cut says *where* to split, not how
    much blank to keep.
    """
    w, h = img.size
    count = max(1, int(count))
    edges = [round(i * h / count) for i in range(count + 1)]

    lines: list[Line] = []
    for i in range(count):
        top, bottom = edges[i], edges[i + 1]
        strip = img.crop((0, top, w, bottom))
        box = ink_box(strip, cfg)
        if box is not None:
            x1, y1, x2, y2 = box
            top, bottom = top + y1, top + y2
        else:
            x1, x2 = 0, w                        # blank strip: keep it whole
        lines.append(Line(i, (x1, top, x2, bottom), img.crop((x1, top, x2, bottom))))
    return lines


def _offset(lines: list[Line], dx: int, dy: int) -> list[Line]:
    """Move line boxes back into source-image coordinates."""
    if not (dx or dy):
        return lines
    return [
        Line(
            ln.index,
            (ln.box[0] + dx, ln.box[1] + dy, ln.box[2] + dx, ln.box[3] + dy),
            ln.image,
            ln.part,
        )
        for ln in lines
    ]


def clamp_box(box: tuple[int, int, int, int], size: tuple[int, int]) -> tuple[int, int, int, int] | None:
    """`box` cut down to the image, or None when nothing of it is inside."""
    w, h = size
    x1, y1 = max(0, int(box[0])), max(0, int(box[1]))
    x2, y2 = min(w, int(box[2])), min(h, int(box[3]))
    return (x1, y1, x2, y2) if x2 > x1 and y2 > y1 else None


def split_image(
    image_path: Path,
    mode: str = "auto",
    cfg: dict | None = None,
    detect_chip: bool | None = None,
    chip_box: tuple[int, int, int, int] | None = None,
) -> tuple[list[Line], tuple[int, int, int, int] | None]:
    """Segment one file into text lines, cropping to the chip package first.

    Returns (lines, chip_box). There is always at least one line -- a photo with
    nothing readable comes back as the whole image, for the model to reject on
    confidence rather than for the caller to special-case. `chip_box` is None
    when no package was found, or when detection is off.

    A `chip_box` passed in (source pixels) is used as the package instead of
    searching for one -- a fixture holds every part in the same place, and a
    box drawn once by hand beats a detector on a part too damaged to find.
    """
    if mode not in MODES:
        raise ValueError(f"Unknown split mode {mode!r}; expected one of {MODES}")
    img = Image.open(image_path)
    img.load()

    s = segment_config(cfg)
    want_chip = bool(s["chip_detect"]) if detect_chip is None else detect_chip
    if chip_box is not None and mode != "none":
        chip_box = clamp_box(chip_box, img.size)
    else:
        chip_box = find_chip(img, cfg) if (want_chip and mode != "none") else None
    target = img.crop(chip_box) if chip_box else img
    dx, dy = (chip_box[0], chip_box[1]) if chip_box else (0, 0)

    if mode == "none":
        lines: list[Line] = []
    elif mode == "half":
        lines = half_lines(target, int(s["max_lines"] or 2), cfg)
    else:
        lines = auto_lines(target, cfg)

    if not lines:
        lines = [Line(0, (0, 0, *target.size), target)]
    return _offset(lines, dx, dy), chip_box


def save_crops(image_path: Path, lines: list[Line], out_dir: Path, index: int = 0) -> list[Path]:
    """Write each line crop so a bad read can be inspected afterwards.

    The source index prefixes the name: photos from different folders routinely
    share a basename, and the crops would otherwise overwrite each other.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for line in lines:
        path = out_dir / f"{index:04d}_{Path(image_path).stem}_{line.name}.png"
        line.image.save(path)
        written.append(path)
    return written


def save_chip(image_path: Path, img: Image.Image, box: tuple[int, int, int, int], out_dir: Path, index: int = 0) -> Path:
    """Write the detected package crop next to the line crops."""
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{index:04d}_{Path(image_path).stem}_chip.png"
    img.crop(box).save(path)
    return path
