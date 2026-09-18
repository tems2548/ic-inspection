"""Naming the kind of defect from the shape of the damaged area.

Synthetic damage maps, drawn to the proportions the real ones have: a square
map over a 16:9 frame, the part a box in the middle of it. Each test is one
kind and the one feature that makes it that kind.
"""

import numpy as np
import pytest

from icinspect.anomaly.shape import (
    KIND_BROKEN,
    KIND_NONE,
    KIND_PATCH,
    KIND_SCRATCH,
    KIND_SCRATCHES,
    KIND_SPECKS,
    KIND_SPOT,
    KIND_WEAR,
    _label_flood,
    analyse,
    label_blobs,
)

Image = pytest.importorskip("PIL.Image", reason="pillow required")

MAP = (256, 256)
PART = (0.30, 0.22, 0.69, 0.71)         # x1, y1, x2, y2 of the frame, as on the fixture
ASPECT = 16 / 9


def part_px():
    x1, y1, x2, y2 = PART
    return int(x1 * MAP[1]), int(y1 * MAP[0]), int(x2 * MAP[1]), int(y2 * MAP[0])


def blank():
    return np.zeros(MAP, dtype=bool)


def kind_of(mask, **kw):
    return analyse(mask, None, PART, ASPECT, **kw).kind


def test_nothing_damaged_is_no_kind():
    assert kind_of(blank()) == KIND_NONE
    assert analyse(None, None, PART).kind == KIND_NONE


def test_a_long_thin_mark_is_a_scratch():
    mask = blank()
    mask[120:126, 90:160] = True                     # 70 wide, 6 tall
    shape = analyse(mask, None, PART, ASPECT)
    assert shape.kind == KIND_SCRATCH
    assert shape.largest.elongation > 3
    assert "longer than wide" in shape.reason


def test_two_long_marks_are_scratches():
    mask = blank()
    mask[100:106, 90:160] = True
    mask[140:146, 90:160] = True
    assert kind_of(mask) == KIND_SCRATCHES


def test_a_compact_blob_is_a_spot():
    mask = blank()
    yy, xx = np.ogrid[:MAP[0], :MAP[1]]
    # A round drop in the *frame* comes out 16:9 taller than wide in the
    # square map, because a map column covers more of the frame than a row.
    mask[((xx - 128) / 4.5) ** 2 + ((yy - 120) / 8.0) ** 2 <= 1] = True
    shape = analyse(mask, None, PART, ASPECT)
    assert shape.kind == KIND_SPOT
    assert shape.largest.elongation < 2, "the frame's aspect is corrected for"


def test_an_irregular_area_is_a_patch():
    mask = blank()
    mask[100:120, 100:140] = True                    # a lump ...
    mask[110:145, 130:140] = True                    # ... with an arm: fills its box badly
    assert kind_of(mask) == KIND_PATCH


def test_most_of_the_part_damaged_is_wear():
    mask = blank()
    px1, py1, px2, py2 = part_px()
    mask[py1:py2, px1:px1 + int((px2 - px1) * 0.6)] = True
    assert kind_of(mask) == KIND_WEAR


def test_scattered_tiny_marks_are_specks():
    mask = blank()
    for y, x in ((90, 100), (110, 130), (140, 110), (150, 150)):
        mask[y:y + 3, x:x + 3] = True
    assert kind_of(mask) == KIND_SPECKS


def test_backdrop_showing_through_the_part_is_a_chunk_missing():
    """The same blob is a patch on a dirty part and a broken corner on a
    chipped one; only the photo tells them apart."""
    mask = blank()
    px1, py1, px2, py2 = part_px()
    mask[py2 - 30:py2, px2 - 30:px2] = True          # the bottom-right corner
    assert kind_of(mask) == KIND_PATCH

    photo = Image.new("L", (1920, 1080), 235)        # bright fixture backdrop
    from PIL import ImageDraw

    draw = ImageDraw.Draw(photo)
    draw.rectangle((PART[0] * 1920, PART[1] * 1080, PART[2] * 1920, PART[3] * 1080), fill=60)
    assert kind_of(mask, photo=photo) == KIND_PATCH, "an intact part: dirt"
    draw.rectangle(((PART[2] - 0.12) * 1920, (PART[3] - 0.12) * 1080,
                    PART[2] * 1920, PART[3] * 1080), fill=235)   # corner gone
    shape = analyse(mask, None, PART, ASPECT, photo=photo)
    assert shape.kind == KIND_BROKEN
    assert shape.largest.background > 0.2


def test_a_shrunken_package_outline_is_a_chunk_missing_too():
    """A broken side shows the fracture face, not the backdrop; the package
    finder measuring a smaller body is what gives it away."""
    mask = blank()
    px1, py1, px2, py2 = part_px()
    mask[py1:py2, px2 - 25:px2] = True
    assert kind_of(mask) == KIND_PATCH
    assert kind_of(mask, body_frac=0.8) == KIND_BROKEN
    assert kind_of(mask, body_frac=0.95) == KIND_PATCH


def test_the_bevel_strip_along_the_border_is_not_counted():
    mask = blank()
    px1, py1, px2, py2 = part_px()
    mask[py2 - 4:py2, px1 + 10:px2 - 10] = True      # a thin bright strip on the bottom edge
    assert kind_of(mask) == KIND_NONE
    mask[120:126, 90:160] = True
    shape = analyse(mask, None, PART, ASPECT)
    assert shape.kind == KIND_SCRATCH and shape.count == 1, "the strip does not make it two"


def test_the_fallback_labeller_agrees_with_scipy():
    mask = blank()
    mask[10:20, 10:20] = True
    mask[30:32, 30:32] = True
    mask[19, 20] = True                                # diagonal neighbour: 8-connected
    labels, count = _label_flood(mask)
    assert count == 2
    assert labels[19, 20] == labels[15, 15]
    other, n = label_blobs(mask)
    assert n == 2


def test_describe_reads_as_a_sentence():
    mask = blank()
    mask[120:126, 90:160] = True
    text = analyse(mask, None, PART, ASPECT).describe()
    assert text.startswith("scratch, ") and "% of the part" in text
