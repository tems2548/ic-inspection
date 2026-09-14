"""Turning an anomalib anomaly map into something you can look at.

PatchCore returns a per-pixel score map alongside the image score. anomalib
normalises both against the threshold it calibrated, so **0.5 is the threshold**
on every image and the maps are comparable between photos -- which is the whole
reason the colour scale here is absolute rather than stretched per image. A
per-image stretch would paint a spotless part in the same fierce colours as a
cracked one.

The ramp is a semantic heat ramp -- yellow to orange to red, analogous
neighbours, hot means defective -- and it always ships with the scale legend in
`scale_strip`, so a colour can be read back to a number. Low scores fade out
instead of tinting the whole package: below `FLOOR` the photo shows through
untouched, because on a good part every pixel carries some small score and a
flat wash over the whole frame says nothing.
"""

from __future__ import annotations

import numpy as np
from PIL import Image

# Sequential heat: one analogous run (yellow -> orange -> red), no rainbow.
# Stops are (position, RGB) with position on the normalised 0..1 score scale.
RAMP: tuple[tuple[float, tuple[int, int, int]], ...] = (
    (0.00, (237, 161, 0)),     # yellow  - just above the floor
    (0.50, (235, 104, 52)),    # orange  - the threshold
    (1.00, (208, 59, 59)),     # red     - well past it
)

THRESHOLD = 0.5     # where anomalib's calibrated threshold always lands
FLOOR = 0.35        # below this the photo is left alone
FULL = 0.60         # at and above this the tint is at full strength
MASK_COLOR = (208, 59, 59)

# The floor sits just under the threshold on purpose. Every pixel of a moulded
# package carries some score -- on a real capture the body reads 0.2-0.4 all
# over -- so a floor near zero paints the whole part and says nothing. Starting
# just below the decision boundary shows the region that actually drove it,
# and "sensitivity" in the window walks this value down for a closer look.
SENSITIVITY = {"defects only": 0.42, "near threshold": 0.35, "all activity": 0.12}


def _ramp_lut() -> np.ndarray:
    """256-entry RGB lookup table for the ramp, built once per call site."""
    positions = np.array([p for p, _ in RAMP], dtype=np.float32)
    colors = np.array([c for _, c in RAMP], dtype=np.float32)
    x = np.linspace(0.0, 1.0, 256, dtype=np.float32)
    lut = np.stack([np.interp(x, positions, colors[:, i]) for i in range(3)], axis=1)
    return lut.astype(np.uint8)


LUT = _ramp_lut()


def to_uint8(anomaly_map: np.ndarray) -> np.ndarray:
    """Store a map compactly: 0..1 normalised score -> 0..255.

    A folder of 200 captures is 200 of these; float32 at full resolution would
    be a third of a gigabyte of nothing but display data.
    """
    clipped = np.clip(np.asarray(anomaly_map, dtype=np.float32), 0.0, 1.0)
    return (clipped * 255).round().astype(np.uint8)


def limit_mask(shape: tuple[int, int], region: tuple[float, float, float, float]) -> np.ndarray:
    """Boolean mask of `region` (fractions of the frame) over a map of `shape`."""
    h, w = shape[:2]
    x1, y1, x2, y2 = region
    mask = np.zeros((h, w), dtype=bool)
    cx1, cx2 = int(np.floor(x1 * w)), int(np.ceil(x2 * w))
    cy1, cy2 = int(np.floor(y1 * h)), int(np.ceil(y2 * h))
    mask[max(cy1, 0):max(cy2, 0), max(cx1, 0):max(cx2, 0)] = True
    return mask


def restrict(scores: np.ndarray, region: tuple[float, float, float, float] | None) -> np.ndarray:
    """Zero everything outside `region`, so the background neither counts nor shows.

    PatchCore scores every patch of the frame, the fixture backdrop included,
    and the image score is driven by the hottest patch anywhere in it. On a real
    capture the brightest thing in the map is often the halo around the part --
    the backdrop's own texture and the moulding edge -- which has nothing to do
    with whether the chip is defective. Scoring only the part means one number
    for one question.

    The same function feeds the score and the picture, so what the window draws
    is exactly what was counted.
    """
    if region is None:
        return scores
    return np.where(limit_mask(np.asarray(scores).shape, region), scores, 0).astype(np.uint8)


def peak(scores: np.ndarray, region: tuple[float, float, float, float] | None = None) -> float:
    """The hottest pixel, on anomalib's 0..1 scale, inside `region` if given."""
    values = np.asarray(scores)
    if region is not None:
        mask = limit_mask(values.shape, region)
        values = values[mask] if mask.any() else values
    return round(float(values.max()) / 255.0, 4) if values.size else 0.0


def damage_mask(
    scores: np.ndarray,
    region: tuple[float, float, float, float] | None,
    cutoff: float,
) -> np.ndarray:
    """Everything inside `region` at or above `cutoff` -- the area that failed.

    Deliberately the same cutoff the verdict uses, rather than anomalib's own
    pixel mask: that mask is thresholded at the pixel line calibrated over whole
    frames, and on a part that fails at 0.46 it is empty, so a failing part
    would report nothing damaged.
    """
    values = np.asarray(scores)
    hot = values >= int(round(cutoff * 255))
    if region is not None:
        hot = hot & limit_mask(values.shape, region)
    return hot


def damage_ratio(
    scores: np.ndarray,
    region: tuple[float, float, float, float] | None,
    cutoff: float,
) -> float:
    """Share of the *part* that is damaged (0..1), not share of the frame.

    The frame is mostly fixture, so a share of it understates the damage by
    about five to one -- a broken corner reads 5% of the frame and 27% of the
    chip. The chip is what is being judged.
    """
    values = np.asarray(scores)
    inside = limit_mask(values.shape, region) if region is not None else np.ones(
        values.shape[:2], dtype=bool)
    pixels = int(inside.sum())
    if not pixels:
        return 0.0
    return round(float(damage_mask(values, region, cutoff).sum()) / pixels, 4)


def colorize(
    scores: np.ndarray,
    opacity: float = 0.75,
    floor: float = FLOOR,
    limit: tuple[float, float, float, float] | None = None,
) -> Image.Image:
    """Score map (uint8, 0..255 for 0..1) -> RGBA tint, transparent where quiet.

    `limit` is the part: outside it nothing is painted at all, so an excluded
    background reads as excluded rather than as a quiet one.
    """
    values = np.asarray(scores, dtype=np.float32) / 255.0
    rgb = LUT[np.clip(scores, 0, 255).astype(np.uint8)]
    alpha = np.clip((values - floor) / max(FULL - floor, 1e-6), 0.0, 1.0)
    alpha = alpha * np.clip(opacity, 0.0, 1.0)
    if limit is not None:
        alpha = np.where(limit_mask(values.shape, limit), alpha, 0.0)
    return Image.fromarray(
        np.dstack([rgb, (alpha * 255).astype(np.uint8)[..., None]]).astype(np.uint8), mode="RGBA")


def outline(mask: np.ndarray, width: int = 2) -> Image.Image:
    """The thresholded region as an outline, not a fill.

    The tint already carries magnitude; what the outline adds is the model's
    actual decision -- this is the part it calls defective -- and an outline
    leaves the pixels underneath visible to judge it by.
    """
    solid = np.asarray(mask).astype(bool)
    if not solid.any():
        return Image.fromarray(np.zeros((*solid.shape, 4), dtype=np.uint8), mode="RGBA")
    inner = solid.copy()
    for _ in range(max(width, 1)):
        shrunk = inner.copy()
        shrunk[1:, :] &= inner[:-1, :]
        shrunk[:-1, :] &= inner[1:, :]
        shrunk[:, 1:] &= inner[:, :-1]
        shrunk[:, :-1] &= inner[:, 1:]
        inner = shrunk
    edge = solid & ~inner
    rgba = np.zeros((*solid.shape, 4), dtype=np.uint8)
    rgba[edge] = (*MASK_COLOR, 255)
    return Image.fromarray(rgba, mode="RGBA")


def overlay_region(
    scores: np.ndarray,
    mask: np.ndarray | None,
    region: tuple[float, float, float, float],
    size: tuple[int, int],
    opacity: float = 0.75,
    show_mask: bool = True,
    floor: float = FLOOR,
    limit: tuple[float, float, float, float] | None = None,
) -> Image.Image:
    """The overlay for one visible slice of the photo, at screen resolution.

    `region` is that slice in fractions of the source image (x1, y1, x2, y2).
    The map is small (the model's input, e.g. 384x384) and the photo is 4K, so
    the crop happens in map space and only the part on screen is ever scaled.
    """
    h, w = np.asarray(scores).shape[:2]
    x1, y1, x2, y2 = region
    box = (
        max(0.0, x1 * w), max(0.0, y1 * h),
        min(float(w), x2 * w), min(float(h), y2 * h),
    )
    if box[2] - box[0] < 1e-3 or box[3] - box[1] < 1e-3:
        return Image.new("RGBA", size, (0, 0, 0, 0))

    target = (max(size[0], 1), max(size[1], 1))
    # BILINEAR, not NEAREST: the map is a smoothed field, and blocky steps would
    # invite reading a pixel edge as a defect boundary.
    tint = colorize(scores, opacity, floor, limit).resize(target, Image.BILINEAR, box=box)
    if show_mask and mask is not None and np.asarray(mask).any():
        edge = outline(mask).resize(target, Image.NEAREST, box=box)
        tint = Image.alpha_composite(tint, edge)
    return tint


def scale_strip(width: int = 200, height: int = 12, floor: float = FLOOR) -> Image.Image:
    """The legend: the ramp itself, left (quiet) to right (well past threshold).

    Drawn over the same score range the strip is labelled with, so a colour on
    the photo can be read back to a number instead of guessed at.
    """
    x = np.linspace(0.0, 1.0, max(width, 2), dtype=np.float32)
    rgb = LUT[(x * 255).astype(np.uint8)]
    alpha = np.clip((x - floor) / max(FULL - floor, 1e-6), 0.0, 1.0)
    row = np.concatenate([rgb, (alpha * 255).astype(np.uint8)[:, None]], axis=1)
    return Image.fromarray(np.tile(row, (max(height, 1), 1, 1)).astype(np.uint8), mode="RGBA")


# -- anomalib's own look -------------------------------------------------
#
# anomalib's visualizer paints the map with OpenCV's JET colormap over the full
# 0..1 range and blends it across the whole frame, so cool blue means "nothing
# here" rather than "nothing painted". It is a rainbow, which is the wrong
# default for reading magnitude -- neighbouring hues are not ordered, so a
# reader cannot tell 0.3 from 0.5 without the scale -- and that is exactly why
# the window's own ramp above is the default. This mode exists so a screenshot
# from here matches the figures anomalib and the PatchCore papers produce, and
# so the whole field is visible at once. It always ships with the scale legend.

def _jet_lut() -> np.ndarray:
    """OpenCV's JET, computed rather than table-copied: the classic piecewise ramp."""
    x = np.linspace(0.0, 1.0, 256, dtype=np.float32)

    def band(center: float) -> np.ndarray:
        # Each channel is a trapezoid of half-width 0.25 around its centre.
        return np.clip(1.5 - np.abs(4.0 * (x - center)), 0.0, 1.0)

    rgb = np.stack([band(0.75), band(0.5), band(0.25)], axis=1)
    return (rgb * 255).astype(np.uint8)


JET = _jet_lut()


def jet_overlay(
    scores: np.ndarray,
    opacity: float = 0.5,
    limit: tuple[float, float, float, float] | None = None,
) -> Image.Image:
    """The whole map in JET at a flat opacity -- anomalib's blend, not a threshold cut.

    JET paints zero dark blue, so without `limit` an excluded background would
    come out tinted as if it had been scored and found quiet.
    """
    rgb = JET[np.clip(scores, 0, 255).astype(np.uint8)]
    alpha = np.full(rgb.shape[:2], np.clip(opacity, 0.0, 1.0), dtype=np.float32)
    if limit is not None:
        alpha = np.where(limit_mask(np.asarray(scores).shape, limit), alpha, 0.0)
    return Image.fromarray(
        np.dstack([rgb, (alpha * 255).astype(np.uint8)[..., None]]).astype(np.uint8), mode="RGBA")


def jet_strip(width: int = 200, height: int = 12) -> Image.Image:
    """Legend for the JET view: the full 0..1 range, opaque."""
    x = np.linspace(0.0, 1.0, max(width, 2), dtype=np.float32)
    rgb = JET[(x * 255).astype(np.uint8)]
    row = np.concatenate([rgb, np.full((rgb.shape[0], 1), 255, dtype=np.uint8)], axis=1)
    return Image.fromarray(np.tile(row, (max(height, 1), 1, 1)).astype(np.uint8), mode="RGBA")


def region_overlay(
    scores: np.ndarray,
    mask: np.ndarray | None,
    region: tuple[float, float, float, float],
    size: tuple[int, int],
    style: str = "threshold",
    opacity: float = 0.75,
    show_mask: bool = True,
    floor: float = FLOOR,
    limit: tuple[float, float, float, float] | None = None,
) -> Image.Image:
    """`overlay_region`, with a choice of look: this window's ramp, or anomalib's JET."""
    if style != "jet":
        return overlay_region(scores, mask, region, size, opacity, show_mask, floor, limit)

    h, w = np.asarray(scores).shape[:2]
    x1, y1, x2, y2 = region
    box = (max(0.0, x1 * w), max(0.0, y1 * h), min(float(w), x2 * w), min(float(h), y2 * h))
    if box[2] - box[0] < 1e-3 or box[3] - box[1] < 1e-3:
        return Image.new("RGBA", size, (0, 0, 0, 0))
    target = (max(size[0], 1), max(size[1], 1))
    tint = jet_overlay(scores, opacity, limit).resize(target, Image.BILINEAR, box=box)
    if show_mask and mask is not None and np.asarray(mask).any():
        edge = outline(mask).resize(target, Image.NEAREST, box=box)
        tint = Image.alpha_composite(tint, edge)
    return tint


def caption(panel: Image.Image, text: str) -> Image.Image:
    """anomalib captions each panel in a band across the top; so does this."""
    from PIL import ImageDraw, ImageFont

    out = panel.convert("RGBA")
    draw = ImageDraw.Draw(out, "RGBA")
    size = max(11, out.width // 26)
    try:
        font = ImageFont.truetype("segoeui.ttf", size)
    except OSError:                      # no Windows fonts: PIL's built-in still reads
        font = ImageFont.load_default()
    box = draw.textbbox((0, 0), text, font=font)
    height = box[3] - box[1] + 8
    draw.rectangle((0, 0, out.width, height), fill=(13, 17, 23, 190))
    draw.text((6, 3), text, fill=(255, 255, 255, 255), font=font)
    return out


def panel_view(
    photo: Image.Image,
    scores: np.ndarray,
    mask: np.ndarray | None,
    style: str = "jet",
    panel_width: int = 640,
    opacity: float = 0.5,
    floor: float = FLOOR,
    limit: tuple[float, float, float, float] | None = None,
) -> Image.Image:
    """anomalib's visualizer grid: the photo, the map over it, the map, the mask.

    Built at `panel_width` per panel rather than at the source resolution -- four
    panels of a 4K capture would be a 15000-pixel-wide bitmap to show a chip.
    """
    ratio = photo.height / max(photo.width, 1)
    size = (panel_width, max(int(panel_width * ratio), 1))
    base = photo.convert("RGBA").resize(size, Image.LANCZOS)

    tint = region_overlay(scores, mask, (0, 0, 1, 1), size, style, opacity, True, floor, limit)
    blended = Image.alpha_composite(base, tint)

    if style == "jet":
        pure = jet_overlay(scores, 1.0, limit).resize(size, Image.BILINEAR)
    else:
        pure = Image.alpha_composite(
            Image.new("RGBA", size, (255, 255, 255, 255)),
            colorize(scores, 1.0, floor, limit).resize(size, Image.BILINEAR),
        )

    solid = np.asarray(mask).astype(bool) if mask is not None else np.zeros((2, 2), dtype=bool)
    mask_rgb = np.where(solid[..., None], np.array([255, 255, 255], dtype=np.uint8),
                        np.array([13, 17, 23], dtype=np.uint8))
    mask_panel = Image.fromarray(mask_rgb.astype(np.uint8), mode="RGB").resize(size, Image.NEAREST)

    panels = [
        caption(base, "Image"),
        caption(blended, "Image + Anomaly Map"),
        caption(pure, "Anomaly Map"),
        caption(mask_panel, "Predicted Mask"),
    ]
    gap = 6
    grid = Image.new("RGB", (size[0] * len(panels) + gap * (len(panels) - 1), size[1]),
                     (252, 252, 251))
    for index, panel in enumerate(panels):
        grid.paste(panel.convert("RGB"), (index * (size[0] + gap), 0))
    return grid
