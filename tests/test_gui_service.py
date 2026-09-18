"""The GUI's inspection engine, minus the window.

Only the parts that decide something are tested here: what counts as a read,
how a verdict is reached, what the report says, and the staging that keeps a
one-photo inspection from scoring the other 55 in the folder. Nothing in this
file opens a Tk window, so it runs on a headless machine like the rest.
"""

from pathlib import Path

import pytest

torch = pytest.importorskip("torch", reason="the GUI service imports the torch-backed dataset")

from icinspect.gui.service import (  # noqa: E402 - after the torch guard
    Options,
    Result,
    Segment,
    _staging,
    write_report,
)


def segment(text="NE555P", conf=1.0, ignored=False, line=0, part=0):
    return Segment(line=line, part=part, box=(0, 0, 10, 10), text=text,
                   confidence=conf, raw_text=text, ignored=ignored)


def test_reads_skip_ignored_and_empty_segments():
    result = Result(
        path=Path("p.jpg"),
        segments=[
            segment("NE555P", 1.0),
            segment("R7", 0.9, ignored=True, line=1),      # a logo
            segment("", 0.0, line=1, part=1),              # nothing survived
            segment("93M", 0.8, line=1, part=2),
        ],
    )
    assert result.marking == "NE555P 93M"
    assert result.confidence == pytest.approx(0.9)


def test_confidence_is_zero_when_nothing_was_read():
    assert Result(path=Path("p.jpg")).confidence == 0.0
    assert Result(path=Path("p.jpg")).marking == ""


def test_report_columns_match_the_pipeline_csv(tmp_path):
    result = Result(
        path=Path("p.jpg"), verdict="FAIL", reasons=["marking-mismatch"],
        score=0.42, anomaly_label=0, chip=(1, 2, 3, 4), segments=[segment()],
    )
    out = write_report([result], tmp_path / "r.csv", expected="LM358N")
    text = out.read_text(encoding="utf-8")
    header, row = text.splitlines()[:2]
    columns = header.split(",")
    # The pipeline's columns come first and in its order, so a reader written
    # against `icinspect.pipeline` keeps working; the per-pixel numbers, which
    # the pipeline has no data for, are appended after them.
    assert columns[:9] == [
        "image", "verdict", "reasons", "anomaly_score", "anomaly_label",
        "marking", "marking_lines", "ocr_confidence", "expected",
    ]
    assert columns[9:] == ["chip", "peak_score", "damage_ratio", "defect_area",
                           "part_score", "truth", "iou", "decided_on", "part",
                           "defect_kind", "defect_detail", "defect_blobs"]
    assert "FAIL" in row and "marking-mismatch" in row and "LM358N" in row
    assert '"1,2,3,4"' in row


def test_report_carries_the_map_numbers_when_there_is_a_map(tmp_path):
    import numpy as np

    heat = np.zeros((8, 8), dtype=np.uint8)
    heat[2:4, 2:4] = 204                      # 0.8 on the normalised scale
    mask = np.zeros((8, 8), dtype=bool)
    mask[2:4, 2:4] = True
    result = Result(path=Path("p.jpg"), score=0.8, heat=heat, mask=mask)
    assert result.peak_score == pytest.approx(0.8, abs=0.005)
    assert result.defect_area == pytest.approx(4 / 64)
    import csv

    out = write_report([result], tmp_path / "r.csv")
    with out.open(encoding="utf-8") as fh:
        row = next(csv.DictReader(fh))
    assert row["defect_area"] == "0.0625"
    assert row["part_score"] == "", "nothing was excluded, so there is no part-only score"
    assert row["decided_on"] == "frame"


def test_report_refuses_an_empty_run(tmp_path):
    with pytest.raises(ValueError):
        write_report([], tmp_path / "r.csv")


def _images(folder: Path, count: int) -> list[Path]:
    from PIL import Image

    folder.mkdir(parents=True, exist_ok=True)
    made = []
    for i in range(count):
        path = folder / f"ic_{i:03d}.png"
        Image.new("L", (8, 8), 200).save(path)
        made.append(path)
    return made


def test_staging_passes_whole_folders_straight_through(tmp_path):
    images = _images(tmp_path / "raw", 3)
    with _staging(images) as (folders, back):
        assert folders == [tmp_path / "raw"]
        assert set(back) == {str(p.resolve()) for p in images}
        assert back[str(images[0].resolve())] == str(images[0].resolve())


def test_staging_isolates_a_subset_and_maps_it_back(tmp_path):
    images = _images(tmp_path / "raw", 5)
    picked = [images[1]]
    with _staging(picked) as (folders, back):
        staged_dir = folders[0]
        assert staged_dir != tmp_path / "raw"
        staged = list(staged_dir.iterdir())
        assert len(staged) == 1, "only the picked photo should be scored"
        assert back[str(staged[0].resolve())] == str(picked[0].resolve())
    assert not staged_dir.exists(), "the scratch folder is cleaned up"


def test_staging_keeps_same_named_files_apart(tmp_path):
    good = _images(tmp_path / "test" / "good", 2)
    defect = _images(tmp_path / "test" / "defect", 2)
    picked = [good[0], defect[0]]     # both are ic_000.png
    with _staging(picked) as (folders, back):
        assert len(list(folders[0].iterdir())) == 2
        assert set(back.values()) == {str(p.resolve()) for p in picked}


def test_options_defaults_match_the_cli_defaults():
    opts = Options()
    assert opts.split == "auto"           # the pipeline's default
    assert opts.check_anomaly and opts.check_ocr
    assert opts.min_char_confidence is None   # None means "whatever ocr.yaml says"
    assert opts.threshold is None


def test_a_truncated_capture_is_skipped_not_fatal(tmp_path):
    """anomalib stops its whole pass on one bad file; the window must not."""
    from PIL import Image

    from icinspect.gui.service import _readable

    good = tmp_path / "good.png"
    Image.new("L", (20, 20), 90).save(good)
    bad = tmp_path / "bad.jpg"
    bad.write_bytes(good.read_bytes()[:40])
    messages = []

    class Log:
        def log(self, text):
            messages.append(text)

    assert _readable([good, bad], Log()) == [good]
    assert messages and "bad.jpg" in messages[0]
