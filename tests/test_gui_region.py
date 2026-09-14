"""Where the scored region comes from.

The body detector measures the part in front of it, which is exactly wrong for
a part with a chunk missing: it reports a smaller body, and the damage falls
outside the region being scored. The region therefore comes from the good
parts -- something the defect cannot influence.
"""

from types import SimpleNamespace

import pytest

pytest.importorskip("numpy", reason="numpy required")
Image = pytest.importorskip("PIL.Image", reason="pillow required")
ImageDraw = pytest.importorskip("PIL.ImageDraw", reason="pillow required")
pytest.importorskip("torch", reason="the service imports the torch-backed dataset")

from icinspect.gui.service import REGION_REFERENCE, Inspector  # noqa: E402
from icinspect.ocr.segment import find_chip  # noqa: E402

BODY = (60, 40, 140, 110)          # where the package sits in every fake capture


# A diagonal bite out of the bottom-right corner, like a part that has had a
# chunk knocked off it: big enough that the rows it crosses stop reading as
# body, which is what makes the detector follow the damage.
BROKEN_OFF = [(140, 55), (140, 110), (75, 110)]


def make_part(path, missing=None, size=(200, 150)):
    """A dark package on a clean background, optionally with a chunk gone."""
    img = Image.new("L", size, 235)
    draw = ImageDraw.Draw(img)
    draw.rectangle(BODY, fill=60)
    if missing:
        draw.polygon(missing, fill=235)        # the chunk that broke off
    img.save(path)
    return path


def area(box) -> float:
    return (box[2] - box[0]) * (box[3] - box[1])


@pytest.fixture
def inspector(tmp_path, monkeypatch):
    """An inspector whose dataset and cache live under tmp_path."""
    good = tmp_path / "anomaly" / "train" / "good"
    good.mkdir(parents=True)
    for i in range(5):
        make_part(good / f"good_{i}.png")
    insp = Inspector()
    monkeypatch.setattr(insp, "paths", lambda: SimpleNamespace(anomaly_root=tmp_path / "anomaly"))
    monkeypatch.setattr(insp, "patchcore_dir", lambda: tmp_path / "results")
    return insp, tmp_path


# -- the problem being solved --------------------------------------------


def test_a_broken_part_measures_smaller_than_a_whole_one(tmp_path):
    """This is the bug: damage shrinks the box that damage is judged inside."""
    whole = make_part(tmp_path / "whole.png")
    broken = make_part(tmp_path / "broken.png", missing=BROKEN_OFF)
    with Image.open(whole) as img:
        whole_box = find_chip(img)
    with Image.open(broken) as img:
        broken_box = find_chip(img)
    assert whole_box and broken_box
    assert area(broken_box) < area(whole_box) * 0.9, (
        "the detector follows the damage, so a per-photo region would cut it out"
    )


# -- the fix --------------------------------------------------------------


def test_reference_region_is_learned_from_the_good_parts(inspector):
    insp, tmp_path = inspector
    region = insp.reference_region()
    assert region is not None
    with Image.open(tmp_path / "anomaly" / "train" / "good" / "good_0.png") as img:
        width, height = img.size
        expected = find_chip(img)
    for got, want in zip(region, (expected[0] / width, expected[1] / height,
                                  expected[2] / width, expected[3] / height)):
        assert got == pytest.approx(want, abs=0.01)


def test_the_reference_does_not_shrink_for_a_broken_part(inspector):
    """The whole point: a chunk missing changes the score, not the window."""
    insp, tmp_path = inspector
    reference = insp.reference_region()
    broken = make_part(tmp_path / "broken.png", missing=BROKEN_OFF)
    with Image.open(broken) as img:
        measured = insp._part_from_chip(find_chip(img), broken)
    assert area(measured) < area(reference) * 0.9, "measuring this photo would shrink it"
    assert insp.reference_region() == reference, "the fixed region is unmoved"


def test_the_region_is_cached_next_to_the_checkpoint(inspector):
    insp, tmp_path = inspector
    region = insp.reference_region()
    cache = tmp_path / "results" / "part_region.json"
    assert cache.exists(), "computing it again every session would be wasteful"

    fresh = Inspector()
    fresh.paths = insp.paths          # type: ignore[method-assign]
    fresh.patchcore_dir = insp.patchcore_dir      # type: ignore[method-assign]
    assert fresh.reference_region() == region, "a new session reads the cache"


def test_clearing_forces_a_fresh_measurement(inspector):
    insp, tmp_path = inspector
    first = insp.reference_region()
    good = tmp_path / "anomaly" / "train" / "good"
    for path in good.glob("*.png"):        # the fixture moved: parts sit elsewhere now
        make_part(path, size=(200, 150))
        img = Image.new("L", (200, 150), 235)
        draw = ImageDraw.Draw(img)
        draw.rectangle((20, 20, 180, 130), fill=60)
        img.save(path)
    insp.clear_reference()
    second = insp.reference_region(refresh=True)
    assert second != first
    assert area(second) > area(first)


def test_no_good_parts_means_no_reference(tmp_path, monkeypatch):
    insp = Inspector()
    monkeypatch.setattr(insp, "paths", lambda: SimpleNamespace(anomaly_root=tmp_path / "empty"))
    monkeypatch.setattr(insp, "patchcore_dir", lambda: tmp_path / "results")
    assert insp.reference_region() is None, "fall back to measuring each photo"


def test_reference_is_the_default_mode():
    from icinspect.gui.service import Options

    assert Options().part_region_mode == REGION_REFERENCE
