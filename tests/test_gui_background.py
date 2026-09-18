"""Excluding the fixture from the score.

PatchCore scores every patch of the frame, so the hottest thing in the map is
often the halo around the part rather than the part. These cover the three
pieces that follow from cutting the background out: the region maths, the
numbers a result then reports, and the cutoff -- which has to be refitted,
because anomalib's own line was calibrated with the background still counting.
"""

import pytest

np = pytest.importorskip("numpy", reason="numpy required")

from icinspect.gui import heatmap  # noqa: E402

torch = pytest.importorskip("torch", reason="the service imports the torch-backed dataset")

from icinspect.gui.service import (  # noqa: E402
    PIXEL_THRESHOLD,
    Inspector,
    Options,
    Result,
    _best_f1,
)
from icinspect.gui.report import auroc  # noqa: E402


def _map(hot_region=None, background=40, hot=230):
    """A 40x40 map: a quiet field, optionally with a hot square somewhere."""
    scores = np.full((40, 40), background, dtype=np.uint8)
    if hot_region:
        y1, y2, x1, x2 = hot_region
        scores[y1:y2, x1:x2] = hot
    return scores


PART = (0.25, 0.25, 0.75, 0.75)          # the middle half of the frame


# -- the region maths ----------------------------------------------------


def test_restrict_zeroes_everything_outside_the_part():
    scores = _map(hot_region=(0, 6, 0, 6))          # hot in the corner: background
    kept = heatmap.restrict(scores, PART)
    assert kept[2, 2] == 0, "the corner is outside the part and stops counting"
    assert kept[20, 20] == scores[20, 20], "inside the part nothing changes"


def test_peak_ignores_a_hot_background():
    scores = _map(hot_region=(0, 6, 0, 6))
    assert heatmap.peak(scores) == pytest.approx(230 / 255, abs=1e-3)
    assert heatmap.peak(scores, PART) == pytest.approx(40 / 255, abs=1e-3)


def test_peak_still_sees_a_defect_on_the_part():
    scores = _map(hot_region=(18, 24, 18, 24))      # hot in the middle: the part
    assert heatmap.peak(scores, PART) == pytest.approx(230 / 255, abs=1e-3)


def test_restrict_without_a_region_changes_nothing():
    scores = _map(hot_region=(0, 6, 0, 6))
    assert np.array_equal(heatmap.restrict(scores, None), scores)


# -- what a result then reports ------------------------------------------


def _result(**kwargs):
    from pathlib import Path

    return Result(path=Path("p.jpg"), **kwargs)


def test_deciding_score_prefers_the_part_once_the_cutoff_is_fitted():
    assert _result(score=0.9, part_score=0.4, part_decides=True).deciding_score == 0.4
    assert _result(score=0.9).deciding_score == 0.9, "no part found: the frame decides"


def test_an_unfitted_cutoff_leaves_the_verdict_with_the_whole_frame():
    """Judging a part score against anomalib's line would pass every defect."""
    result = _result(score=0.56, part_score=0.46, part=PART, part_decides=False)
    assert result.deciding_score == 0.56, "the measured part score is shown, not obeyed"


def test_scored_map_and_area_follow_the_part():
    mask = np.zeros((40, 40), dtype=bool)
    mask[0:6, 0:6] = True                            # flagged region in the background
    result = _result(score=0.9, heat=_map(hot_region=(0, 6, 0, 6)), mask=mask, part=PART,
                     part_score=0.16)
    assert result.scored_heat[2, 2] == 0
    assert result.defect_area == 0.0, "a flagged patch of fixture is not a flagged part"
    assert result.peak_score == pytest.approx(40 / 255, abs=1e-3)


def test_a_flagged_region_on_the_part_still_counts():
    mask = np.zeros((40, 40), dtype=bool)
    mask[18:24, 18:24] = True
    result = _result(score=0.9, heat=_map(hot_region=(18, 24, 18, 24)), mask=mask, part=PART)
    assert result.defect_area == pytest.approx(36 / 1600)


def test_report_carries_both_numbers(tmp_path):
    import csv

    from icinspect.gui.service import write_report

    result = _result(score=0.56, part_score=0.46, part=PART, heat=_map(), part_decides=True)
    out = write_report([result], tmp_path / "r.csv")
    with out.open(encoding="utf-8") as fh:
        row = next(csv.DictReader(fh))
    assert row["anomaly_score"] == "0.56", "anomalib's own number is still reported"
    assert row["part_score"] == "0.46"
    assert row["part"].startswith("0.2500,")
    assert row["decided_on"] == "part"


# -- the cutoff ----------------------------------------------------------


def _unpinned(tmp_path) -> Inspector:
    """An inspector whose config pins nothing, whatever this project's config says."""
    config = tmp_path / "patchcore.yaml"
    config.write_text("threshold:\n  image_score: null\n  part_score: null\n", encoding="utf-8")
    insp = Inspector(patchcore_config=str(config))
    insp.part_threshold = None
    return insp


def test_cutoff_falls_back_to_anomalibs_pixel_line(tmp_path):
    insp = _unpinned(tmp_path)
    assert insp.part_cutoff() == PIXEL_THRESHOLD
    assert not insp.part_cutoff_fitted(), "a stand-in must not decide verdicts"


def test_a_pinned_config_value_is_a_fitted_cutoff(tmp_path):
    config = tmp_path / "patchcore.yaml"
    config.write_text("threshold:\n  part_score: 0.3784\n", encoding="utf-8")
    insp = Inspector(patchcore_config=str(config))
    assert insp.part_cutoff() == pytest.approx(0.3784)
    assert insp.part_cutoff_fitted()
    assert insp.part_cutoff_source()[1] == "config"


def test_a_calibrated_cutoff_wins_over_the_fallback(tmp_path):
    insp = _unpinned(tmp_path)
    insp.part_threshold = 0.36
    assert insp.part_cutoff() == pytest.approx(0.36)
    assert insp.part_cutoff_fitted()


def test_a_pinned_run_threshold_wins_over_everything(tmp_path):
    insp = _unpinned(tmp_path)
    insp.part_threshold = 0.36
    assert insp.part_cutoff(Options(threshold=0.42)) == pytest.approx(0.42)


def test_best_f1_puts_the_cutoff_between_the_classes():
    scored = [(0.20, 0), (0.25, 0), (0.27, 0), (0.46, 1), (0.48, 1), (0.52, 1)]
    best = _best_f1(scored)
    assert 0.27 < best["threshold"] <= 0.46
    assert best["f1"] == 1.0
    assert auroc(scored) == 1.0


def test_best_f1_on_overlapping_classes_trades_precision_for_recall():
    scored = [(0.30, 0), (0.45, 0), (0.40, 1), (0.50, 1)]
    best = _best_f1(scored)
    assert 0 < best["f1"] < 1.0, "overlap cannot be separated perfectly"
    assert best["recall"] >= best["precision"], "F1 favours catching both defects here"


def test_auroc_is_a_half_when_the_score_says_nothing():
    scored = [(0.5, 0), (0.5, 1), (0.5, 0), (0.5, 1)]
    assert auroc(scored) == pytest.approx(0.5)


def test_calibration_needs_a_labelled_test_set(tmp_path, monkeypatch):
    from icinspect.gui.runner import Progress
    import queue
    import threading

    insp = Inspector()

    class EmptyPaths:
        anomaly_root = tmp_path / "nothing"

    monkeypatch.setattr(insp, "paths", lambda: EmptyPaths())
    with pytest.raises(FileNotFoundError, match="test/good"):
        insp.calibrate_part_threshold(Progress(queue.Queue(), threading.Event()))
