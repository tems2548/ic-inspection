"""The package box drawn by hand on the Inspect tab.

A fixture holds every part in the same place, so one box drawn on one photo
can stand in for both the region PatchCore scores and the package the marking
reader crops to -- and, unlike a detector, it does not wander off a part that
is too damaged to make sense of.
"""

from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("numpy", reason="numpy required")
Image = pytest.importorskip("PIL.Image", reason="pillow required")
ImageDraw = pytest.importorskip("PIL.ImageDraw", reason="pillow required")
pytest.importorskip("torch", reason="the service imports the torch-backed dataset")

from icinspect.gui.service import (  # noqa: E402
    REGION_MANUAL,
    REGION_MODES,
    Inspector,
    Options,
    _region_pixels,
)
from icinspect.ocr.segment import clamp_box, split_image  # noqa: E402


def make_photo(path: Path, size=(200, 150)) -> Path:
    """A dark package with one light marking line, on a clean background."""
    img = Image.new("L", size, 235)
    draw = ImageDraw.Draw(img)
    draw.rectangle((60, 40, 140, 110), fill=60)
    draw.text((75, 65), "NE555P", fill=220)
    img.save(path)
    return path


@pytest.fixture
def inspector(tmp_path, monkeypatch):
    insp = Inspector()
    monkeypatch.setattr(insp, "paths", lambda: SimpleNamespace(anomaly_root=tmp_path / "anomaly"))
    monkeypatch.setattr(insp, "patchcore_dir", lambda: tmp_path / "results")
    return insp, tmp_path


# -- the segmenter takes a pinned package ------------------------------------


def test_a_pinned_box_replaces_the_package_search(tmp_path):
    photo = make_photo(tmp_path / "p.png")
    pinned = (70, 50, 130, 100)
    lines, chip = split_image(photo, "auto", chip_box=pinned)
    assert chip == pinned, "no search: the box handed in is the package"
    for line in lines:
        x1, y1, x2, y2 = line.box
        assert 70 <= x1 <= x2 <= 130 and 50 <= y1 <= y2 <= 100, "lines are cut inside it"


def test_a_pinned_box_is_clamped_to_the_photo(tmp_path):
    photo = make_photo(tmp_path / "p.png")
    _lines, chip = split_image(photo, "auto", chip_box=(-20, 30, 260, 400))
    assert chip == (0, 30, 200, 150)


def test_a_box_entirely_outside_the_photo_is_nothing():
    assert clamp_box((300, 300, 400, 400), (200, 150)) is None
    assert clamp_box((10, 10, 10, 50), (200, 150)) is None, "zero width"


def test_split_none_ignores_the_box(tmp_path):
    photo = make_photo(tmp_path / "p.png")
    _lines, chip = split_image(photo, "none", chip_box=(70, 50, 130, 100))
    assert chip is None, "a pre-cut crop has no package to pin"


# -- the drawn box is remembered ---------------------------------------------


def test_drawn_box_is_a_region_mode():
    assert REGION_MANUAL in REGION_MODES


def test_no_box_until_one_is_drawn(inspector):
    insp, _ = inspector
    assert insp.manual_region() is None


def test_the_drawn_box_survives_a_new_session(inspector):
    insp, tmp_path = inspector
    saved = insp.set_manual_region((0.3, 0.2, 0.7, 0.7), drawn_on=Path("p.png"))
    assert saved == (0.3, 0.2, 0.7, 0.7)
    assert (tmp_path / "results" / "manual_region.json").exists()

    fresh = Inspector()
    fresh.paths = insp.paths                    # type: ignore[method-assign]
    fresh.patchcore_dir = insp.patchcore_dir    # type: ignore[method-assign]
    assert fresh.manual_region() == (0.3, 0.2, 0.7, 0.7)


def test_the_drawn_box_is_clamped_to_the_frame(inspector):
    insp, _ = inspector
    assert insp.set_manual_region((-0.1, 0.2, 1.4, 0.7)) == (0.0, 0.2, 1.0, 0.7)


def test_a_box_with_no_area_is_refused(inspector):
    insp, _ = inspector
    with pytest.raises(ValueError):
        insp.set_manual_region((0.5, 0.2, 0.5, 0.7))
    assert insp.manual_region() is None


def test_forgetting_the_box_removes_it_from_disk_too(inspector):
    insp, tmp_path = inspector
    insp.set_manual_region((0.3, 0.2, 0.7, 0.7))
    insp.clear_manual_region()
    assert insp.manual_region() is None
    assert not (tmp_path / "results" / "manual_region.json").exists()


def test_the_learned_region_and_the_drawn_box_are_separate(inspector):
    """Relearning after the fixture moves must not also throw the drawing away."""
    insp, tmp_path = inspector
    good = tmp_path / "anomaly" / "train" / "good"
    good.mkdir(parents=True)
    make_photo(good / "good.png")
    insp.set_manual_region((0.3, 0.2, 0.7, 0.7))
    assert insp.reference_region() is not None
    insp.forget_region()
    assert insp.manual_region() == (0.3, 0.2, 0.7, 0.7)


def test_unloading_rereads_the_box_from_disk(inspector):
    insp, tmp_path = inspector
    assert insp.manual_region() is None            # cached as "nothing there"
    (tmp_path / "results").mkdir()
    (tmp_path / "results" / "manual_region.json").write_text(
        '{"region": [0.1, 0.1, 0.9, 0.9]}', encoding="utf-8")
    insp.unload()
    assert insp.manual_region() == (0.1, 0.1, 0.9, 0.9)


def test_region_pixels_scales_to_the_photo(tmp_path):
    photo = make_photo(tmp_path / "p.png", size=(200, 150))
    assert _region_pixels((0.3, 0.2, 0.7, 0.7), photo) == (60, 30, 140, 105)
    assert _region_pixels(None, photo) is None


# -- and it drives both checks -------------------------------------------------


def test_the_reader_crops_to_the_drawn_box(inspector, monkeypatch):
    """In drawn-box mode the reader is handed the box instead of searching."""
    import icinspect.gui.service as service

    insp, tmp_path = inspector
    photo = make_photo(tmp_path / "p.png")
    insp.set_manual_region((0.3, 0.2, 0.7, 0.7))
    seen: dict = {}

    def fake_split(model, charset, cfg, images, device, mode, crops_dir, **kwargs):
        seen["chip_box"] = kwargs.get("chip_box")
        return [], []

    monkeypatch.setattr(insp, "_load_ocr", lambda prog: (None, None, {}, None))
    monkeypatch.setattr("icinspect.ocr.infer.predict_split", fake_split)
    monkeypatch.setattr("icinspect.ocr.infer.predict_paths", lambda *a, **k: [])

    read = insp._ocr_reader(Options(part_region_mode=REGION_MANUAL), service.Progress.__new__(service.Progress))
    read(photo)
    assert seen["chip_box"] == (60, 30, 140, 105)

    read = insp._ocr_reader(Options(), service.Progress.__new__(service.Progress))
    read(photo)
    assert seen["chip_box"] is None, "in the other modes the package is still searched for"
