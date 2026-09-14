"""The statistics and heat-map layers, without a window.

The charts are drawn onto a headless Agg figure and asserted on structurally --
that the threshold rule is there, that two series carry a legend, that an empty
run says so rather than showing bare axes. That is the part that breaks
silently; how pretty it looks is a screenshot's job.
"""

import json
from collections import Counter

import pytest

np = pytest.importorskip("numpy", reason="numpy required")

from icinspect.gui import heatmap  # noqa: E402
from icinspect.gui.stats import RunStats, ocr_history, run_stats  # noqa: E402


# -- the heat ramp -------------------------------------------------------


def test_quiet_pixels_leave_the_photo_alone():
    quiet = np.full((4, 4), int(0.10 * 255), dtype=np.uint8)
    rgba = np.asarray(heatmap.colorize(quiet))
    assert rgba[..., 3].max() == 0, "below the floor nothing should be painted"


def test_hot_pixels_are_painted_and_get_hotter_with_score():
    warm = np.full((4, 4), int(0.45 * 255), dtype=np.uint8)
    hot = np.full((4, 4), int(0.90 * 255), dtype=np.uint8)
    warm_rgba = np.asarray(heatmap.colorize(warm))
    hot_rgba = np.asarray(heatmap.colorize(hot))
    assert warm_rgba[..., 3].max() > 0
    assert hot_rgba[..., 3].max() >= warm_rgba[..., 3].max()
    # The ramp runs yellow -> red, so red rises and green falls with the score.
    assert hot_rgba[0, 0, 1] < warm_rgba[0, 0, 1]


def test_sensitivity_floors_are_ordered_and_bracket_the_threshold():
    floors = list(heatmap.SENSITIVITY.values())
    assert floors == sorted(floors, reverse=True)
    assert max(floors) < heatmap.THRESHOLD, "even the strictest floor sits under the threshold"


def test_lower_sensitivity_paints_more():
    scores = np.full((4, 4), int(0.30 * 255), dtype=np.uint8)
    strict = np.asarray(heatmap.colorize(scores, floor=heatmap.SENSITIVITY["defects only"]))
    loose = np.asarray(heatmap.colorize(scores, floor=heatmap.SENSITIVITY["all activity"]))
    assert strict[..., 3].max() == 0
    assert loose[..., 3].max() > 0


def test_to_uint8_maps_the_normalised_scale():
    raw = np.array([[0.0, 0.5], [1.0, 2.0]], dtype=np.float32)
    stored = heatmap.to_uint8(raw)
    assert stored.dtype == np.uint8
    assert stored[0, 0] == 0 and stored[0, 1] == 128
    assert stored[1, 0] == 255 and stored[1, 1] == 255, "scores above 1 clamp, they do not wrap"


def test_mask_is_outlined_not_filled():
    mask = np.zeros((10, 10), dtype=bool)
    mask[3:8, 3:8] = True
    edge = np.asarray(heatmap.outline(mask, width=1))[..., 3] > 0
    assert edge[3, 3] and edge[3, 7], "the border is drawn"
    assert not edge[5, 5], "the middle stays clear so the pixels can be judged"


def test_overlay_region_crops_in_map_space():
    """The visible slice decides what is drawn -- panning must not drag the tint along."""
    scores = np.zeros((100, 100), dtype=np.uint8)
    scores[:50, :] = 230                      # hot in the top half only
    top = np.asarray(heatmap.overlay_region(scores, None, (0, 0, 1, 0.5), (20, 20)))[..., 3]
    bottom = np.asarray(heatmap.overlay_region(scores, None, (0, 0.5, 1, 1), (20, 20)))[..., 3]
    assert top.min() > 0, "the hot half is painted throughout"
    # Bilinear resampling reaches across the cut, so the row on the seam picks
    # up a little of the half above it; everything below it stays clear.
    assert bottom[1:].max() == 0
    assert bottom.mean() < top.mean() * 0.05


def test_scale_strip_is_the_legend_for_the_same_floor():
    strip = np.asarray(heatmap.scale_strip(100, 6, floor=0.5))
    assert strip.shape == (6, 100, 4)
    assert strip[0, 10, 3] == 0, "left of the floor the strip is empty"
    assert strip[0, 90, 3] > 0, "the hot end is solid"


# -- run statistics ------------------------------------------------------


class FakeResult:
    def __init__(self, verdict="PASS", score=None, confidence=0.0, reasons=(), peak=None, area=None):
        self.verdict = verdict
        self.score = score
        self.reasons = list(reasons)
        self.reads = [object()] if confidence else []
        self.confidence = confidence
        self.peak_score = peak
        self.defect_area = area


def test_run_stats_counts_and_averages():
    stats = run_stats([
        FakeResult("PASS", 0.10, 0.99),
        FakeResult("FAIL", 0.80, 0.50, ["anomaly"], peak=0.9, area=0.02),
        FakeResult("FAIL", 0.90, 0.0, ["anomaly", "marking-unreadable"]),
    ])
    assert (stats.total, stats.passed, stats.failed) == (3, 1, 2)
    assert stats.pass_rate == pytest.approx(1 / 3)
    assert stats.mean_score == pytest.approx(0.6)
    assert stats.mean_confidence == pytest.approx(0.745)   # only photos that read something
    assert stats.reasons == Counter({"anomaly": 2, "marking-unreadable": 1})
    assert stats.areas == [0.02]


def test_empty_run_is_falsey_and_has_no_averages():
    stats = RunStats()
    assert not stats
    assert stats.pass_rate is None and stats.mean_score is None


def test_ocr_history_reads_metrics_json(tmp_path):
    (tmp_path / "metrics.json").write_text(json.dumps({
        "best_score": 0.99,
        "history": [
            {"epoch": 2, "loss": 0.5, "char_accuracy": 0.9, "seq_accuracy": 0.7, "seconds": 30},
            {"epoch": 1, "loss": 3.0, "char_accuracy": 0.1, "seq_accuracy": 0.0, "seconds": 90},
        ],
    }), encoding="utf-8")
    history = ocr_history(tmp_path)
    assert history.epochs == [1, 2], "epochs come back in order whatever the file says"
    assert history.loss == [3.0, 0.5]
    assert history.best_score == 0.99
    assert history.total_minutes == pytest.approx(2.0)


def test_missing_history_is_empty_not_an_error(tmp_path):
    history = ocr_history(tmp_path)
    assert not history
    assert history.epochs == []


# -- charts --------------------------------------------------------------


@pytest.fixture
def axes():
    matplotlib = pytest.importorskip("matplotlib", reason="matplotlib required for the charts")
    matplotlib.use("Agg")
    from matplotlib.figure import Figure

    return Figure(figsize=(4, 3)).subplots()


def test_score_histogram_marks_the_threshold(axes):
    from icinspect.gui import charts

    charts.score_histogram(axes, [0.1, 0.2, 0.8, 0.9], threshold=0.5)
    rules = [line.get_xdata()[0] for line in axes.lines]
    assert 0.5 in rules, "the deciding threshold is drawn on the chart"
    assert any("threshold" in text.get_text() for text in axes.texts), "and labelled"


def test_empty_chart_explains_itself(axes):
    from icinspect.gui import charts

    charts.score_histogram(axes, [])
    assert not axes.patches, "no bars invented out of nothing"
    assert any(text.get_text() for text in axes.texts), "it says what would fill it"


def test_accuracy_curves_carry_a_legend_and_two_series(axes):
    from icinspect.gui import charts

    charts.accuracy_curves(axes, [1, 2, 3], [0.5, 0.8, 0.9], [0.1, 0.6, 0.85])
    assert len(axes.lines) == 2
    assert axes.get_legend() is not None, "two series are never identified by colour alone"
    colors = {line.get_color() for line in axes.lines}
    assert colors == {charts.SERIES[0], charts.SERIES[1]}


def test_loss_curve_uses_a_log_scale(axes):
    from icinspect.gui import charts

    charts.loss_curve(axes, [1, 2, 3], [3.0, 0.05, 0.001])
    assert axes.get_yscale() == "log"


def test_charts_never_use_a_second_y_axis(axes):
    from icinspect.gui import charts

    charts.accuracy_curves(axes, [1, 2], [0.5, 0.9], [0.4, 0.8])
    assert len(axes.figure.axes) == 1, "no dual-axis chart: loss lives on its own plot"
