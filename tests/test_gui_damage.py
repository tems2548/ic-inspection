"""How much of the part is damaged.

The number an operator grades by is the share of the *chip*, not of the frame:
the frame is mostly fixture, and a broken corner that reads 5% of the frame is
27% of the part. The area is measured at the same cutoff the verdict uses, so a
part that fails can never report nothing damaged.
"""

from pathlib import Path

import pytest

np = pytest.importorskip("numpy", reason="numpy required")

from icinspect.gui import heatmap  # noqa: E402

pytest.importorskip("torch", reason="the service imports the torch-backed dataset")

from icinspect.gui.service import Result  # noqa: E402

PART = (0.25, 0.25, 0.75, 0.75)     # the middle half of a 40x40 map: 400 of 1600 px
CUTOFF = 0.4


def _map(hot=None, quiet=40):
    scores = np.full((40, 40), quiet, dtype=np.uint8)
    if hot:
        y1, y2, x1, x2 = hot
        scores[y1:y2, x1:x2] = 230
    return scores


def _result(**kwargs):
    return Result(path=Path("p.jpg"), **kwargs)


# -- the measurement ------------------------------------------------------


def test_a_clean_part_is_nothing_damaged():
    assert heatmap.damage_ratio(_map(), PART, CUTOFF) == 0.0


def test_the_ratio_is_of_the_part_not_the_frame():
    """Half the part damaged reads 50%, even though it is 12.5% of the frame."""
    scores = _map(hot=(10, 30, 10, 20))      # half the part's width, its full height
    assert heatmap.damage_ratio(scores, PART, CUTOFF) == pytest.approx(0.5, abs=0.01)
    whole_frame = heatmap.damage_ratio(scores, None, CUTOFF)
    assert whole_frame == pytest.approx(0.125, abs=0.01)


def test_damage_outside_the_part_does_not_count():
    scores = _map(hot=(0, 8, 0, 8))          # hot in the corner: fixture
    assert heatmap.damage_ratio(scores, PART, CUTOFF) == 0.0


def test_the_mask_is_the_area_the_ratio_counted():
    scores = _map(hot=(10, 30, 10, 20))
    mask = heatmap.damage_mask(scores, PART, CUTOFF)
    ratio = heatmap.damage_ratio(scores, PART, CUTOFF)
    inside = heatmap.limit_mask(scores.shape, PART)
    assert mask.sum() / inside.sum() == pytest.approx(ratio, abs=1e-4)
    assert not mask[0, 0], "the picture and the number exclude the same pixels"


def test_a_hotter_cutoff_counts_less():
    scores = _map()
    scores[12:28, 12:28] = int(0.45 * 255)
    assert heatmap.damage_ratio(scores, PART, 0.40) > 0
    assert heatmap.damage_ratio(scores, PART, 0.50) == 0.0


# -- what a result reports ------------------------------------------------


def test_a_failing_part_never_reports_nothing_damaged():
    """anomalib's own mask is empty below its pixel line; this must not be."""
    scores = _map()
    scores[12:28, 12:28] = int(0.46 * 255)          # fails a 0.39 cutoff, under anomalib's 0.5
    result = _result(heat=scores, part=PART, part_score=0.46, part_decides=True, cutoff=0.39,
                     mask=np.zeros((40, 40), dtype=bool))
    assert result.defect_area == 0.0, "anomalib's mask sees nothing here"
    assert result.damage_ratio > 0.5, "the deciding cutoff does, and that is what is reported"


def test_no_cutoff_means_no_ratio():
    assert _result(heat=_map()).damage_ratio is None
    assert _result(heat=_map()).damage is None


def test_ratio_reaches_the_report(tmp_path):
    import csv

    from icinspect.gui.service import write_report

    result = _result(heat=_map(hot=(10, 30, 10, 20)), part=PART, cutoff=CUTOFF, score=0.8)
    out = write_report([result], tmp_path / "r.csv")
    with out.open(encoding="utf-8") as fh:
        row = next(csv.DictReader(fh))
    assert float(row["damage_ratio"]) == pytest.approx(0.5, abs=0.01)


# -- the excluded area is left alone -------------------------------------


def test_the_excluded_background_is_not_painted_by_jet():
    """JET draws zero as dark blue, so it has to be told where the part is."""
    scores = _map()
    without = np.asarray(heatmap.jet_overlay(scores, 0.5))
    with_limit = np.asarray(heatmap.jet_overlay(scores, 0.5, limit=PART))
    assert without[0, 0, 3] > 0, "unlimited, every pixel is tinted"
    assert with_limit[0, 0, 3] == 0, "outside the part nothing is painted"
    assert with_limit[20, 20, 3] > 0, "inside it, the map still shows"


def test_the_ramp_view_also_respects_the_limit():
    scores = _map(hot=(0, 8, 0, 8))
    tinted = np.asarray(heatmap.colorize(scores, 0.75, limit=PART))
    assert tinted[2, 2, 3] == 0, "a hot patch of fixture is not painted either"


# -- failing on too much damage ------------------------------------------


def test_a_damage_limit_is_off_by_default():
    from icinspect.gui.service import Options

    assert Options().max_damage is None
