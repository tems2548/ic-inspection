"""Scoring the window against the truth: the confusion matrix and IoU.

The matrix answers "was the verdict right", IoU answers "was the damage in the
right place" -- a part can be failed correctly with the damage found somewhere
else entirely, and only the second notices.
"""

from pathlib import Path
from types import SimpleNamespace

import pytest

np = pytest.importorskip("numpy", reason="numpy required")
Image = pytest.importorskip("PIL.Image", reason="pillow required")
pytest.importorskip("torch", reason="the service imports the torch-backed dataset")

from icinspect.gui.service import Inspector, Result  # noqa: E402
from icinspect.gui.stats import Confusion, run_stats  # noqa: E402

PART = (0.25, 0.25, 0.75, 0.75)
CUTOFF = 0.4


def _map(hot=None, quiet=40):
    scores = np.full((40, 40), quiet, dtype=np.uint8)
    if hot:
        y1, y2, x1, x2 = hot
        scores[y1:y2, x1:x2] = 230
    return scores


def _result(verdict="PASS", **kwargs):
    return Result(path=Path("p.jpg"), verdict=verdict, **kwargs)


# -- the confusion matrix -------------------------------------------------


def test_every_cell_lands_where_it_belongs():
    stats = run_stats([
        _result("FAIL", truth=1),      # caught
        _result("PASS", truth=1),      # missed -- the one that ships
        _result("FAIL", truth=0),      # false alarm
        _result("PASS", truth=0),      # correct pass
        _result("FAIL", truth=None),   # a raw capture: no folder to judge against
    ])
    matrix = stats.confusion
    assert (matrix.tp, matrix.fn, matrix.fp, matrix.tn) == (1, 1, 1, 1)
    assert matrix.labelled == 4, "the unlabelled photo is counted nowhere"
    assert stats.total == 5


def test_the_grid_is_rows_truth_columns_verdict():
    matrix = Confusion(tp=7, fp=2, tn=5, fn=1)
    assert matrix.as_grid() == [[5, 2], [1, 7]]


def test_the_rates_are_the_textbook_ones():
    matrix = Confusion(tp=8, fp=2, tn=6, fn=4)
    assert matrix.accuracy == pytest.approx(14 / 20)
    assert matrix.precision == pytest.approx(8 / 10)
    assert matrix.recall == pytest.approx(8 / 12)
    assert matrix.specificity == pytest.approx(6 / 8)
    assert matrix.f1 == pytest.approx(2 * 0.8 * (2 / 3) / (0.8 + 2 / 3))


def test_an_unlabelled_run_has_no_matrix():
    stats = run_stats([_result("FAIL"), _result("PASS")])
    assert not stats.confusion
    assert stats.confusion.accuracy is None


def test_a_perfect_run_reads_perfectly():
    stats = run_stats([_result("FAIL", truth=1)] * 3 + [_result("PASS", truth=0)] * 2)
    matrix = stats.confusion
    assert matrix.accuracy == 1.0 and matrix.recall == 1.0 and matrix.precision == 1.0
    assert matrix.fn == 0, "nothing shipped"


# -- IoU -------------------------------------------------------------------


def test_iou_is_one_when_the_damage_is_exactly_where_it_was_marked():
    found = _map(hot=(12, 28, 12, 28))
    marked = found >= 128
    result = _result("FAIL", heat=found, part=PART, cutoff=CUTOFF, truth_mask=marked)
    assert result.iou == 1.0


def test_iou_is_zero_when_the_damage_is_somewhere_else():
    found = _map(hot=(12, 18, 12, 18))
    marked = np.zeros((40, 40), dtype=bool)
    marked[22:28, 22:28] = True                    # marked elsewhere in the part
    result = _result("FAIL", heat=found, part=PART, cutoff=CUTOFF, truth_mask=marked)
    assert result.iou == 0.0, "right verdict, wrong place -- and the number says so"


def test_iou_is_the_overlap_over_the_union():
    found = _map(hot=(12, 20, 12, 20))              # 64 px
    marked = np.zeros((40, 40), dtype=bool)
    marked[16:24, 12:20] = True                     # 64 px, half of them shared
    result = _result("FAIL", heat=found, part=PART, cutoff=CUTOFF, truth_mask=marked)
    assert result.iou == pytest.approx(32 / 96, abs=1e-3)


def test_no_mask_means_no_iou():
    assert _result(heat=_map(), part=PART, cutoff=CUTOFF).iou is None


def test_a_clean_part_with_an_empty_mask_agrees_completely():
    result = _result("PASS", heat=_map(), part=PART, cutoff=CUTOFF,
                     truth_mask=np.zeros((40, 40), dtype=bool))
    assert result.iou == 1.0, "nothing marked and nothing found is agreement, not failure"


def test_mean_iou_is_only_over_photos_that_have_a_mask():
    marked = np.zeros((40, 40), dtype=bool)
    marked[12:28, 12:28] = True
    stats = run_stats([
        _result("FAIL", heat=_map(hot=(12, 28, 12, 28)), part=PART, cutoff=CUTOFF,
                truth_mask=marked),
        _result("FAIL", heat=_map(hot=(12, 28, 12, 28)), part=PART, cutoff=CUTOFF),
    ])
    assert stats.ious == [1.0]
    assert stats.mean_iou == 1.0


# -- reading masks off disk -----------------------------------------------


@pytest.fixture
def dataset(tmp_path, monkeypatch):
    root = tmp_path / "anomaly"
    (root / "test" / "defect").mkdir(parents=True)
    (root / "ground_truth" / "defect").mkdir(parents=True)
    insp = Inspector()
    monkeypatch.setattr(insp, "paths", lambda: SimpleNamespace(anomaly_root=root))
    return insp, root


def _mask_file(path, box, size=(80, 60)):
    img = Image.new("L", size, 0)
    for y in range(box[1], box[3]):
        for x in range(box[0], box[2]):
            img.putpixel((x, y), 255)
    img.save(path)


def test_a_mask_is_found_by_name_and_resampled_to_the_map(dataset):
    insp, root = dataset
    photo = root / "test" / "defect" / "P1.jpg"
    photo.write_bytes(b"")
    _mask_file(root / "ground_truth" / "defect" / "P1.png", (0, 0, 40, 60))
    mask = insp.truth_mask_for(photo, (20, 20))
    assert mask is not None and mask.shape == (20, 20)
    assert mask[:, :10].all() and not mask[:, 10:].any(), "the left half is marked"


def test_a_mask_named_with_the_mask_suffix_is_found_too(dataset):
    insp, root = dataset
    photo = root / "test" / "defect" / "P2.jpg"
    photo.write_bytes(b"")
    _mask_file(root / "ground_truth" / "defect" / "P2_mask.png", (0, 0, 80, 60))
    assert insp.truth_mask_for(photo, (10, 10)) is not None


def test_no_mask_folder_is_not_an_error(dataset, tmp_path, monkeypatch):
    insp, root = dataset
    monkeypatch.setattr(insp, "paths", lambda: SimpleNamespace(anomaly_root=tmp_path / "nothing"))
    assert insp.truth_mask_for(root / "test" / "defect" / "P1.jpg", (10, 10)) is None
