"""Line segmentation of whole IC photos (PIL only, no torch)."""

import pytest

from icinspect.ocr.segment import (
    _column_segments,
    _otsu,
    auto_lines,
    find_chip,
    half_lines,
    save_crops,
    segment_config,
    split_image,
)

Image = pytest.importorskip("PIL.Image", reason="pillow required")
ImageDraw = pytest.importorskip("PIL.ImageDraw", reason="pillow required")


def make_chip(tmp_path, name="chip.png", lines=("STM32F407", "VGT6 991"), invert=False):
    """A dark package with two light marking lines (or the reverse)."""
    bg, fg = (40, 220) if not invert else (220, 40)
    img = Image.new("L", (240, 120), bg)
    draw = ImageDraw.Draw(img)
    for i, text in enumerate(lines):
        draw.text((20, 25 + i * 45), text, fill=fg)
    path = tmp_path / name
    img.save(path)
    return path, img


def test_auto_finds_two_marking_lines(tmp_path):
    _, img = make_chip(tmp_path)
    found = auto_lines(img)
    assert len(found) == 2
    top, bottom = found
    assert top.box[1] < bottom.box[1]                 # returned top to bottom
    assert top.box[3] <= bottom.box[3]
    assert all(ln.image.size[0] > 0 and ln.image.size[1] > 0 for ln in found)


def test_auto_is_polarity_agnostic(tmp_path):
    """Laser-etched light-on-dark and printed dark-on-light both segment."""
    _, dark = make_chip(tmp_path, "dark.png")
    _, light = make_chip(tmp_path, "light.png", invert=True)
    assert len(auto_lines(dark)) == len(auto_lines(light)) == 2


def test_auto_survives_uneven_illumination(tmp_path):
    """A brightness ramp across the package must not swamp the ink profile."""
    import numpy as np

    _, img = make_chip(tmp_path)
    arr = np.asarray(img, dtype=np.float32)
    ramp = np.linspace(0, 120, arr.shape[0], dtype=np.float32)[:, None]
    lit = Image.fromarray(np.clip(arr + ramp, 0, 255).astype("uint8"), mode="L")
    assert len(auto_lines(lit)) == 2


def test_max_lines_keeps_the_strongest_bands_in_order(tmp_path):
    _, img = make_chip(tmp_path, lines=("STM32F407", "VGT6", "."))
    cfg = {"segment": {"max_lines": 2}}
    found = auto_lines(img, cfg)
    assert len(found) == 2
    assert found[0].box[1] < found[1].box[1]
    assert [ln.index for ln in found] == [0, 1]


def test_blank_image_yields_no_bands(tmp_path):
    assert auto_lines(Image.new("L", (200, 80), 128)) == []


def test_unmarked_package_yields_no_bands(tmp_path):
    """Sensor noise on a blank package is below the contrast floor, so: no lines."""
    import numpy as np

    rng = np.random.default_rng(0)
    grain = np.clip(60 + rng.normal(0, 3, (80, 200)), 0, 255).astype("uint8")
    assert auto_lines(Image.fromarray(grain, mode="L")) == []


def test_faint_text_below_the_contrast_floor_is_ignored(tmp_path):
    """A marking that barely differs from the package is not read as text."""
    _, faint = make_chip(tmp_path, "faint.png")
    faint = Image.eval(faint, lambda v: 60 if v < 130 else 66)   # 6 levels of contrast
    assert auto_lines(faint) == []
    assert len(auto_lines(faint, {"segment": {"min_contrast": 0.01}})) == 2


def test_otsu_splits_a_bimodal_signal():
    import numpy as np

    values = np.concatenate([np.full(800, 0.1), np.full(200, 0.8)])
    assert 0.1 <= _otsu(values) <= 0.8
    assert _otsu(np.zeros(100)) == 0.0          # nothing to separate


def test_split_image_falls_back_to_the_whole_image(tmp_path):
    path = tmp_path / "blank.png"
    Image.new("L", (200, 80), 128).save(path)
    lines, chip = split_image(path, mode="auto")
    assert len(lines) == 1
    assert lines[0].box == (0, 0, 200, 80)
    assert chip is None


def test_half_splits_into_equal_strips(tmp_path):
    img = Image.new("L", (100, 61), 128)
    top, bottom = half_lines(img)
    assert top.box == (0, 0, 100, 30)
    assert bottom.box == (0, 30, 100, 61)          # odd height: no lost row
    assert len(half_lines(img, 3)) == 3


def test_half_tightens_each_strip_onto_its_ink(tmp_path):
    """The fixed cut decides where to split; the ink decides what to keep."""
    _, img = make_chip(tmp_path)                    # 240x120, midline at y=60
    top, bottom = half_lines(img)
    assert top.box[3] <= 60 <= bottom.box[1]        # neither strip crosses the cut
    for box in (top.box, bottom.box):
        assert box[2] - box[0] < 240                # blank left/right margins gone
        assert box[3] - box[1] < 60                 # and most of the blank strip too


def test_half_keeps_a_blank_strip_whole(tmp_path):
    """Nothing to tighten onto -- return the strip rather than an empty crop."""
    img = Image.new("L", (240, 120), 40)
    draw = ImageDraw.Draw(img)
    draw.text((20, 25), "LM358N", fill=220)         # top half only
    top, bottom = half_lines(img)
    assert bottom.box == (0, 60, 240, 120)
    assert top.box != (0, 0, 240, 60)


def test_split_mode_none_returns_the_untouched_image(tmp_path):
    path, _ = make_chip(tmp_path)
    lines, chip = split_image(path, mode="none")
    assert len(lines) == 1 and lines[0].box == (0, 0, 240, 120)
    assert chip is None                             # no cropping without splitting


def test_split_image_rejects_unknown_mode(tmp_path):
    path, _ = make_chip(tmp_path)
    with pytest.raises(ValueError, match="Unknown split mode"):
        split_image(path, mode="diagonal")


def test_half_mode_reads_max_lines_from_config(tmp_path):
    path, _ = make_chip(tmp_path)
    assert len(split_image(path, mode="half", cfg={"segment": {"max_lines": 3}})[0]) == 3
    assert len(split_image(path, mode="half", cfg=None)[0]) == 2


def test_segment_config_overlays_defaults():
    merged = segment_config({"segment": {"ink_frac": 0.5}})
    assert merged["ink_frac"] == 0.5
    assert merged["pad_frac"] == 0.25               # untouched default survives
    assert segment_config(None)["mode"] == "auto"


def test_save_crops_prefixes_the_source_index(tmp_path):
    path, _ = make_chip(tmp_path)
    lines, _ = split_image(path, mode="half")
    written = save_crops(path, lines, tmp_path / "crops", index=7)
    assert [p.name for p in written] == ["0007_chip_l0p0.png", "0007_chip_l1p0.png"]
    assert all(p.exists() for p in written)


def test_save_crops_does_not_collide_across_same_named_sources(tmp_path):
    """Two folders, one basename -- the crops must not overwrite each other."""
    folders = []
    for name in ("a", "b"):
        folder = tmp_path / name
        folder.mkdir()
        folders.append(make_chip(folder)[0])
    a, b = folders
    out = tmp_path / "crops"
    first = save_crops(a, split_image(a, mode="half")[0], out, index=0)
    second = save_crops(b, split_image(b, mode="half")[0], out, index=1)
    assert not ({p.name for p in first} & {p.name for p in second})


# --- chip detection -------------------------------------------------------

def make_photo(tmp_path, name="photo.png", lines=("NE555P", "93M", "DNJW26"), leads=True):
    """A dark package with leads, on a bright background -- the fixture layout."""
    img = Image.new("L", (400, 300), 235)          # bright backdrop
    draw = ImageDraw.Draw(img)
    body = (90, 70, 310, 230)
    if leads:
        for x in range(110, 300, 50):              # four leads top and bottom
            draw.rectangle((x, 45, x + 26, 70), fill=170)
            draw.rectangle((x, 230, x + 26, 255), fill=170)
    draw.rectangle(body, fill=55)
    for i, text in enumerate(lines):
        draw.text((130, 95 + i * 40), text, fill=205)
    path = tmp_path / name
    img.save(path)
    return path, img, body


def test_find_chip_returns_the_body_without_the_leads(tmp_path):
    _, img, body = make_photo(tmp_path)
    box = find_chip(img)
    assert box is not None
    x1, y1, x2, y2 = box
    assert body[1] <= y1 and y2 <= body[3]          # leads (y<70, y>230) excluded
    assert body[0] <= x1 and x2 <= body[2]
    assert (x2 - x1) > 0.6 * (body[2] - body[0])    # and it is not a sliver


def test_find_chip_gives_up_on_an_empty_frame(tmp_path):
    assert find_chip(Image.new("L", (400, 300), 235)) is None


def test_find_chip_declines_an_already_tight_crop(tmp_path):
    """A crop that is all package has no background to measure against."""
    _, img, body = make_photo(tmp_path)
    assert find_chip(img.crop(body)) is None


def test_split_image_crops_to_the_chip_and_reports_source_boxes(tmp_path):
    path, _, body = make_photo(tmp_path)
    lines, chip = split_image(path, mode="auto")
    assert chip is not None
    assert len(lines) == 3
    for ln in lines:                                # boxes are in photo coordinates
        assert chip[0] <= ln.box[0] and ln.box[2] <= chip[2]
        assert chip[1] <= ln.box[1] and ln.box[3] <= chip[3]
    assert [ln.box[1] for ln in lines] == sorted(ln.box[1] for ln in lines)


def test_chip_detection_can_be_turned_off(tmp_path):
    path, _, _ = make_photo(tmp_path)
    lines, chip = split_image(path, mode="auto", detect_chip=False)
    assert chip is None
    assert lines and lines[0].box[0] >= 0


def test_a_thin_band_on_the_crop_border_is_dropped_as_moulding_rim():
    """A real chip crop: three marking lines plus the lit package edge below them.

    Bands are drawn as dashes rather than glyphs so the row coverage matches a
    real marking line (~40% of the width); the rim is a thinner, sparser streak
    flush against the bottom, exactly what survives `find_chip`'s inset.
    """
    img = Image.new("L", (240, 160), 40)
    draw = ImageDraw.Draw(img)
    for row in (20, 65, 110):                       # three marking lines, 22 px tall
        for x in range(30, 200, 16):
            draw.rectangle((x, row, x + 7, row + 22), fill=210)
    for x in range(60, 150, 16):                    # lit moulding edge, 4 px tall
        draw.rectangle((x, 156, x + 7, 159), fill=210)

    found = auto_lines(img)
    assert len(found) == 3
    assert [ln.index for ln in found] == [0, 1, 2]
    assert all(ln.box[1] < 140 for ln in found)     # none of them is the rim


def test_a_full_height_line_touching_the_border_is_kept():
    """The rim rule must not eat a real line on a tightly cropped package."""
    img = Image.new("L", (240, 120), 40)
    draw = ImageDraw.Draw(img)
    for row in (0, 60):                             # the first line runs into the top edge
        for x in range(30, 200, 16):
            draw.rectangle((x, row, x + 7, row + 30), fill=210)
    assert len(auto_lines(img)) == 2


# --- segments within a line ------------------------------------------------

def test_column_segments_cut_only_at_wide_gaps():
    cols = [(0, 40), (50, 90), (200, 260)]        # two runs close together, then a jump
    assert _column_segments(cols, gap=30) == [(0, 90), (200, 260)]
    assert _column_segments(cols, gap=200) == [(0, 260)]     # nothing is wide enough
    assert _column_segments(cols, gap=0) == [(0, 260)]       # splitting disabled
    assert _column_segments([], gap=30) == []


def test_a_logo_beside_the_text_becomes_its_own_segment():
    """The gap between a logo and the text beside it is wider than a word space."""
    img = Image.new("L", (300, 120), 40)
    draw = ImageDraw.Draw(img)
    draw.ellipse((20, 40, 90, 80), fill=210)          # a blob standing in for the logo
    for x in range(190, 280, 16):                      # dashes standing in for "93M"
        draw.rectangle((x, 40, x + 8, 80), fill=210)

    found = auto_lines(img)
    assert len(found) == 2
    assert [ln.part for ln in found] == [0, 1]
    assert all(ln.index == 0 for ln in found)          # both on the same text line
    assert found[0].box[2] < found[1].box[0]           # and they do not overlap


def test_a_normal_word_space_does_not_split_a_line():
    """`DNJW26 C4` is one line: its space is far narrower than a line height."""
    img = Image.new("L", (300, 120), 40)
    draw = ImageDraw.Draw(img)
    for x in list(range(40, 150, 16)) + list(range(165, 230, 16)):   # 15 px word space
        draw.rectangle((x, 40, x + 8, 80), fill=210)
    assert len(auto_lines(img)) == 1


def test_splitting_can_be_turned_off():
    img = Image.new("L", (300, 120), 40)
    draw = ImageDraw.Draw(img)
    draw.ellipse((20, 40, 90, 80), fill=210)
    for x in range(190, 280, 16):
        draw.rectangle((x, 40, x + 8, 80), fill=210)
    assert len(auto_lines(img, {"segment": {"split_gap_frac": 0}})) == 1


def big_marking(text="NE555P", size=(900, 300)):
    """A marking at photo scale: default-font text is 11 px tall, and at that
    size a speck plus padding is as wide as a letter."""
    ImageFont = pytest.importorskip("PIL.ImageFont")
    img = Image.new("L", size, 40)
    try:
        font = ImageFont.load_default(size=90)
    except TypeError:                                   # Pillow < 10.1
        pytest.skip("needs a scalable default font")
    ImageDraw.Draw(img).text((200, 90), text, fill=220, font=font)
    return img


def test_a_narrow_mark_on_the_crop_border_is_not_a_segment(tmp_path):
    """The pin-1 dimple: a notch flush against the crop edge, cut off as its
    own segment by the gap rule, and read as a digit often enough to matter."""
    img = big_marking()
    ImageDraw.Draw(img).ellipse((-40, 100, 40, 200), fill=220)
    found = auto_lines(img)
    assert len(found) == 1, "the dimple does not get a segment of its own"
    assert found[0].box[0] > 40, "and the text's segment does not start at it"


def test_a_speck_beside_a_word_is_not_a_segment(tmp_path):
    img = big_marking()
    ImageDraw.Draw(img).rectangle((720, 130, 726, 170), fill=220)   # a fleck of texture
    found = auto_lines(img)
    assert len(found) == 1 and found[0].box[2] < 700


def test_a_narrow_mark_is_kept_when_it_is_all_there_is(tmp_path):
    img = Image.new("L", (60, 40), 40)
    ImageDraw.Draw(img).ellipse((0, 12, 8, 26), fill=220)
    assert len(auto_lines(img)) == 1, "a lone segment is never dropped: the reader decides"


def test_a_bright_streak_on_the_crop_border_is_not_a_line():
    """The lit bevel of the package: a solid band flush against the crop edge,
    brighter than any text row. It must neither set the threshold nor be read."""
    img = big_marking()
    ImageDraw.Draw(img).rectangle((0, 282, 900, 300), fill=220)     # solid streak at the bottom
    found = auto_lines(img)
    assert [ln.index for ln in found] == [0]
    assert found[0].box[3] < 260, "the streak is not a line of its own"


def test_a_band_much_thinner_than_the_others_is_grown_to_a_full_line():
    """A faint first line clears the threshold only at its densest rows."""
    from icinspect.ocr.segment import _grow_thin_bands

    rows = [(138, 191), (376, 548), (638, 807)]           # 53 tall next to ~170
    grown = _grow_thin_bands(rows, 975, 0.7)
    assert grown[1:] == rows[1:], "full lines are left alone"
    top, bottom = grown[0]
    assert bottom - top == pytest.approx(170, abs=2)
    assert top < 138 and bottom > 191, "widened about its centre"
    assert bottom <= 376, "never into the next line"


def test_growing_stops_at_the_crop_and_at_a_neighbour():
    from icinspect.ocr.segment import _grow_thin_bands

    assert _grow_thin_bands([(0, 30), (60, 230)], 300, 0.7) == [(0, 60), (60, 230)]
    assert _grow_thin_bands([(100, 270), (280, 300)], 300, 0.7) == [(100, 270), (270, 300)]


def test_a_lone_band_is_not_grown():
    from icinspect.ocr.segment import _grow_thin_bands

    assert _grow_thin_bands([(10, 20)], 100, 0.7) == [(10, 20)]
