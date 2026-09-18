"""The reporting table: escape and overkill rates with intervals, PRO, the
detection limit in millimetres, repeatability from repeated shots, latency."""

from pathlib import Path
from types import SimpleNamespace

import pytest

np = pytest.importorskip("numpy", reason="numpy required")

from icinspect.gui.report import (  # noqa: E402
    Rate,
    auroc,
    build_table,
    detection_limit,
    equivalent_diameter_mm,
    mm_per_pixel,
    pooled_sigma,
    region_overlaps,
    repeat_groups,
    wilson,
)


def _result(name, truth=None, verdict="PASS", score=None, damage=None, mask=None,
            part=(0.25, 0.25, 0.75, 0.75), seconds=None):
    return SimpleNamespace(path=Path(name), truth=truth, verdict=verdict, deciding_score=score,
                           damage=damage, truth_mask=mask, part=part, seconds=seconds)


def _blob(y1, y2, x1, x2, shape=(40, 40)):
    mask = np.zeros(shape, dtype=bool)
    mask[y1:y2, x1:x2] = True
    return mask


# -- the statistics -----------------------------------------------------------


def test_wilson_interval_is_honest_at_zero():
    low, high = wilson(0, 220)
    assert low == 0.0 and 0.01 < high < 0.02, "no escapes in 220 is not 'zero', it is 'under 1.7%'"
    low, high = wilson(5, 10)
    assert low < 0.5 < high
    assert wilson(0, 0) == (0.0, 1.0)


def test_rate_text_carries_interval_and_n():
    text = Rate(2, 100).text()
    assert text.startswith("2.00%") and "95% CI" in text and "n=100" in text


def test_auroc_needs_both_classes():
    assert auroc([(0.2, 0), (0.9, 1)]) == 1.0
    assert auroc([(0.2, 1), (0.9, 1)]) is None


def test_pro_counts_each_marked_region_once():
    marked = _blob(2, 4, 2, 4) | _blob(20, 36, 10, 36)     # a pinhole and a broken corner
    found = _blob(18, 40, 8, 40)                            # only the corner was found
    overlaps = region_overlaps(found, marked)
    assert sorted(overlaps) == [0.0, 1.0], "the pinhole counts as much as the corner"
    assert region_overlaps(found, np.zeros((10, 10), bool)) == []


def test_pooled_sigma_is_within_part_only():
    parts = [[0.20, 0.22, 0.21], [0.60, 0.62, 0.61], [0.40]]
    sigma, used, shots = pooled_sigma(parts)
    assert used == 2 and shots == 6, "a single shot says nothing about repeatability"
    assert sigma == pytest.approx(0.01, abs=1e-9)
    assert pooled_sigma([[0.5]]) == (None, 0, 0)


def test_detection_limit_sits_above_the_largest_miss():
    limit, missed = detection_limit([(0.3, False), (0.5, True), (0.4, True), (0.8, True)])
    assert (limit, missed) == (0.4, 0.3)
    assert detection_limit([(0.3, True), (0.5, True)]) == (0.3, None)
    assert detection_limit([(0.9, False), (0.5, True)]) == (None, 0.9), "nothing caught above the miss"


def test_mm_scale_undoes_the_finder_inset_and_the_square_map():
    # A 9.5 x 6.35 mm body inset by 8% fills a region half the 100 x 100 map wide.
    sx, sy = mm_per_pixel((0.25, 0.25, 0.75, 0.75), (100, 100), (9.5, 6.35), inset=0.08)
    assert sx == pytest.approx(9.5 * 0.84 / 50)
    assert sy == pytest.approx(6.35 * 0.84 / 50)
    disc = _blob(0, 10, 0, 10, shape=(100, 100))
    assert equivalent_diameter_mm(disc, (0.1, 0.2)) == pytest.approx(np.sqrt(4 * 100 * 0.02 / np.pi))


def test_repeat_groups_from_names_and_from_a_listing(tmp_path):
    names = ["A17_1.jpg", "A17_2.jpg", "B02-1.jpg", "P00400.jpg"]
    groups = repeat_groups(names, r"^(.+)[_-]\d+$")
    assert groups["A17"] == [0, 1] and groups["B02"] == [2]
    assert sum(1 for k in groups if k.startswith("\0")) == 1, "P00400 is its own group"

    listing = tmp_path / "repeats.csv"
    listing.write_text("image,part\nP00400.jpg,partX\nA17_1.jpg,partX\n", encoding="utf-8")
    groups = repeat_groups(names, r"^(.+)[_-]\d+$", listing)
    assert groups["partX"] == [0, 3], "the listing wins over the name"


# -- the table ----------------------------------------------------------------


def _table(results, **kwargs):
    settings = dict(cutoff=0.44, body_mm=(9.5, 6.35), inset=0.08, repeats_pattern=r"^(.+)[_-]\d+$",
                    repeats_listing=None, device="test GPU")
    settings.update(kwargs)
    return build_table(results, **settings)


def _by_metric(table):
    return {row.metric.split(" @")[0]: row for row in table.rows}


def test_an_empty_run_says_what_each_row_needs():
    rows = _by_metric(_table([], fallback_auroc="0.9514"))
    assert [r.metric for r in _table([]).rows] == [
        "Image AUROC", "PRO score", "Escape rate", "Overkill rate", "Detection limit",
        "Score repeatability σ", "Inference latency"]
    assert rows["Image AUROC"].value == "0.9514" and not rows["Image AUROC"].available
    assert "ground_truth" in rows["PRO score"].basis
    assert "repeats.csv" in rows["Score repeatability σ"].basis
    assert "test GPU" in rows["Inference latency"].basis


def test_rates_come_from_the_folder_labels_with_n():
    results = [_result(f"d{i}.jpg", truth=1, verdict="FAIL", score=0.6) for i in range(9)]
    results.append(_result("d9.jpg", truth=1, verdict="PASS", score=0.3))      # the escape
    results += [_result(f"g{i}.jpg", truth=0, verdict="PASS", score=0.2) for i in range(19)]
    results.append(_result("g19.jpg", truth=0, verdict="FAIL", score=0.5))     # the overkill
    results.append(_result("raw.jpg", truth=None, verdict="FAIL", score=0.9))  # no label: not counted
    rows = _by_metric(_table(results))
    assert rows["Escape rate"].value.startswith("10.00%") and "n=10" in rows["Escape rate"].value
    assert rows["Overkill rate"].value.startswith("5.00%") and "n=20" in rows["Overkill rate"].value
    assert rows["Escape rate"].metric == "Escape rate @ 0.4400"
    assert rows["Image AUROC"].available and "30 labelled" in rows["Image AUROC"].basis


def test_detection_limit_and_pro_come_from_the_masks():
    small = _blob(10, 12, 10, 12)          # 4 px
    medium = _blob(10, 14, 10, 14)         # 16 px
    large = _blob(10, 20, 10, 20)          # 100 px
    results = [
        _result("d0.jpg", truth=1, verdict="PASS", score=0.3, damage=np.zeros((40, 40), bool), mask=small),
        _result("d1.jpg", truth=1, verdict="FAIL", score=0.6, damage=medium, mask=medium),
        _result("d2.jpg", truth=1, verdict="FAIL", score=0.7, damage=large, mask=large),
    ]
    rows = _by_metric(_table(results))
    scale = mm_per_pixel((0.25, 0.25, 0.75, 0.75), (40, 40), (9.5, 6.35), 0.08)
    expected = equivalent_diameter_mm(medium, scale)
    assert rows["Detection limit"].available
    assert rows["Detection limit"].value == f"{expected:.2f} mm (equivalent diameter)"
    assert "2/3 masked defects caught" in rows["Detection limit"].basis
    assert rows["PRO score"].value == f"{(0 + 1 + 1) / 3:.4f}"
    assert "3 marked region(s) on 3 photo(s)" in rows["PRO score"].basis


def test_detection_limit_needs_a_body_size():
    results = [_result("d1.jpg", truth=1, verdict="FAIL", score=0.6, damage=_blob(1, 3, 1, 3),
                       mask=_blob(1, 3, 1, 3))]
    row = _by_metric(_table(results, body_mm=None))["Detection limit"]
    assert not row.available and "body_mm" in row.basis


def test_repeatability_and_latency_from_the_run():
    results = [
        _result("A_1.jpg", score=0.20, seconds=1.0), _result("A_2.jpg", score=0.22, seconds=1.2),
        _result("B_1.jpg", score=0.60, seconds=1.1), _result("B_2.jpg", score=0.62, seconds=30.0),
        _result("lone.jpg", score=0.40, seconds=1.0),
    ]
    rows = _by_metric(_table(results))
    sigma = rows["Score repeatability σ"]
    assert sigma.available and sigma.value == f"{np.sqrt(0.0002):.4f}"
    assert "2 part(s), 4 shots" in sigma.basis and "% of the cutoff" in sigma.basis
    latency = rows["Inference latency"]
    assert latency.value == "1100 ms/unit", "the median, so one slow photo does not set it"
    assert "5 photo(s)" in latency.basis


def test_markdown_has_one_line_per_metric():
    text = _table([]).as_markdown()
    assert text.count("\n") == 8 and text.startswith("| Metric | Value | Basis |")
