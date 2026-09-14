"""Picking a model, and the anomalib-style views.

Both are about honesty: the run has to use the checkpoint that is named in the
box, and the JET view has to be the same ramp anomalib draws, so a screenshot
from this window and one from anomalib's visualizer can be compared.
"""

from pathlib import Path

import pytest

np = pytest.importorskip("numpy", reason="numpy required")
Image = pytest.importorskip("PIL.Image", reason="pillow required")

from icinspect.gui import heatmap  # noqa: E402


# -- anomalib's JET ------------------------------------------------------


def test_jet_runs_blue_through_green_to_red():
    """OpenCV's JET, which is what anomalib paints its maps with."""
    low, mid, high = (heatmap.JET[0], heatmap.JET[128], heatmap.JET[255])
    assert low[2] > low[0] and low[1] == 0, "cold end is blue"
    assert mid[1] > mid[0] and mid[1] > mid[2], "middle is green"
    assert high[0] > high[1] and high[2] == 0, "hot end is red"


def test_jet_paints_everywhere_unlike_the_threshold_ramp():
    """The whole point of the anomalib look: no floor, the field is always visible."""
    quiet = np.full((4, 4), int(0.05 * 255), dtype=np.uint8)
    assert np.asarray(heatmap.colorize(quiet))[..., 3].max() == 0
    jet = np.asarray(heatmap.jet_overlay(quiet, opacity=0.5))
    assert jet[..., 3].min() > 0, "even a quiet pixel is tinted in the anomalib view"


def test_region_overlay_routes_by_style():
    scores = np.full((16, 16), int(0.05 * 255), dtype=np.uint8)
    region, size = (0, 0, 1, 1), (8, 8)
    ours = np.asarray(heatmap.region_overlay(scores, None, region, size, style="threshold"))
    theirs = np.asarray(heatmap.region_overlay(scores, None, region, size, style="jet"))
    assert ours[..., 3].max() == 0
    assert theirs[..., 3].min() > 0


def test_jet_strip_is_a_full_range_legend():
    strip = np.asarray(heatmap.jet_strip(64, 5))
    assert strip.shape == (5, 64, 4)
    assert strip[..., 3].min() == 255, "the JET legend is opaque end to end"
    assert strip[0, 0, 2] > strip[0, -1, 2], "blue at 0, not at 1"


# -- the panel grid ------------------------------------------------------


def _photo(width=80, height=45):
    return Image.new("RGB", (width, height), (120, 120, 120))


def test_panel_view_is_four_captioned_panels():
    scores = np.zeros((32, 32), dtype=np.uint8)
    scores[8:16, 8:16] = 230
    mask = scores > 200
    grid = heatmap.panel_view(_photo(), scores, mask, panel_width=100)
    # Four panels of 100px, three 6px gaps between them.
    assert grid.width == 4 * 100 + 3 * 6
    assert grid.height == int(100 * 45 / 80)

    # Each panel carries a caption band across its top, so the top row is dark
    # where the panel body is not.
    top = np.asarray(grid.convert("RGB"))[2, :100]
    assert top.mean() < 90, "the caption band darkens the top of the panel"


def test_panel_view_survives_a_missing_mask():
    scores = np.full((16, 16), 100, dtype=np.uint8)
    grid = heatmap.panel_view(_photo(), scores, None, panel_width=60)
    assert grid.width == 4 * 60 + 3 * 6, "the mask panel is still drawn, just empty"


# -- choosing a checkpoint ----------------------------------------------


@pytest.fixture
def inspector(tmp_path, monkeypatch):
    pytest.importorskip("torch", reason="service imports the torch-backed dataset")
    from icinspect.gui.service import Inspector

    insp = Inspector()
    ocr_dir, pc_dir = tmp_path / "ocr", tmp_path / "patchcore"
    (pc_dir / "Patchcore" / "ic" / "v1" / "weights").mkdir(parents=True)
    (pc_dir / "Patchcore" / "ic" / "v2" / "weights").mkdir(parents=True)
    ocr_dir.mkdir(parents=True)
    monkeypatch.setattr(insp, "ocr_dir", lambda: ocr_dir)
    monkeypatch.setattr(insp, "patchcore_dir", lambda: pc_dir)
    return insp, ocr_dir, pc_dir


def _touch(path: Path, mtime: float) -> Path:
    path.write_bytes(b"x")
    import os

    os.utime(path, (mtime, mtime))
    return path


def test_checkpoints_are_listed_newest_first(inspector):
    insp, _ocr, pc = inspector
    old = _touch(pc / "Patchcore" / "ic" / "v1" / "weights" / "model.ckpt", 1_000_000)
    new = _touch(pc / "Patchcore" / "ic" / "v2" / "weights" / "model.ckpt", 2_000_000)
    assert insp.anomaly_checkpoints() == [new, old]
    assert insp.anomaly_checkpoint() == new, "the newest run is the default"


def test_best_reader_leads_even_when_last_is_newer(inspector):
    insp, ocr, _pc = inspector
    _touch(ocr / "crnn_best.pt", 1_000_000)
    last = _touch(ocr / "crnn_last.pt", 2_000_000)
    listed = insp.ocr_checkpoints()
    assert listed[0].name == "crnn_best.pt", "best is the sane default, newest or not"
    assert last in listed
    assert insp.ocr_checkpoint().name == "crnn_best.pt"


def test_selecting_a_model_overrides_the_default(inspector):
    insp, _ocr, pc = inspector
    old = _touch(pc / "Patchcore" / "ic" / "v1" / "weights" / "model.ckpt", 1_000_000)
    new = _touch(pc / "Patchcore" / "ic" / "v2" / "weights" / "model.ckpt", 2_000_000)
    insp.select_anomaly(old)
    assert insp.anomaly_checkpoint() == old, "the chosen run is the one that gets used"
    insp.select_anomaly(None)
    assert insp.anomaly_checkpoint() == new, "clearing the choice returns to the newest"


def test_a_deleted_choice_falls_back_instead_of_failing(inspector):
    insp, _ocr, pc = inspector
    kept = _touch(pc / "Patchcore" / "ic" / "v1" / "weights" / "model.ckpt", 1_000_000)
    gone = pc / "Patchcore" / "ic" / "v2" / "weights" / "model.ckpt"
    insp.select_anomaly(gone)
    assert insp.anomaly_checkpoint() == kept


def test_no_checkpoints_at_all_is_not_an_error(inspector):
    insp, _ocr, _pc = inspector
    assert insp.anomaly_checkpoints() == []
    assert insp.anomaly_checkpoint() is None
    assert insp.ocr_checkpoint() is None
