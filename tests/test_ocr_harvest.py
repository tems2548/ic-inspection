"""Harvesting labelled line crops from photos of a known marking."""

from pathlib import Path

import pytest

from icinspect.ocr.harvest import assign, harvest, parse_expected

Image = pytest.importorskip("PIL.Image", reason="pillow required")


def row(line, part, x, raw="", text=None):
    return {"line": line, "part": part, "box": f"{x},{line * 100},{x + 50},{line * 100 + 40}",
            "raw_text": raw, "text": raw if text is None else text, "image": "p.jpg"}


def test_expected_marking_is_lines_of_segments():
    assert parse_expected("NE555P | 93M | DNJW26 G4") == [["NE555P"], ["93M"], ["DNJW26", "G4"]]
    assert parse_expected("ne555p|93m") == [["NE555P"], ["93M"]]
    with pytest.raises(ValueError):
        parse_expected(" | ")


def test_segments_are_labelled_by_position():
    rows = [row(0, 0, 10, "NE5S5P"), row(1, 0, 60, "93M"), row(2, 0, 10, "DNJW26"), row(2, 1, 80, "G4")]
    picks, problem = assign(rows, parse_expected("NE555P | 93M | DNJW26 G4"))
    assert problem is None
    assert [(p.row["line"], p.row["part"], p.text) for p in picks] == [
        (0, 0, "NE555P"), (1, 0, "93M"), (2, 0, "DNJW26"), (2, 1, "G4")]


def test_a_faint_line_the_reader_gave_up_on_still_gets_its_label():
    """The whole point: the crop the reader cannot read is the one to train on."""
    rows = [row(0, 0, 10, raw="NCZEED", text=""), row(1, 0, 60, "93M"), row(2, 0, 10, "DNJW26")]
    picks, problem = assign(rows, parse_expected("NE555P | 93M | DNJW26"))
    assert problem is None
    assert picks[0].text == "NE555P"


def test_the_logo_beside_the_text_is_left_out_by_its_read():
    rows = [row(0, 0, 10, "NE555P"), row(1, 0, 10, raw="R7", text=""), row(1, 1, 60, "93M")]
    picks, problem = assign(rows, parse_expected("NE555P | 93M"))
    assert problem is None
    assert [p.row["part"] for p in picks if p.row["line"] == 1] == [1]
    assert picks[1].text == "93M"


def test_the_extra_segment_is_dropped_by_position_when_reads_are_junk():
    """Two segments, one label: order still decides which of a poor pair fits."""
    rows = [row(0, 0, 10, "X"), row(0, 1, 60, "NE555P")]
    picks, _ = assign(rows, parse_expected("NE555P"))
    assert picks[0].row["part"] == 1


def test_a_photo_with_the_wrong_number_of_lines_is_skipped():
    rows = [row(0, 0, 10, "NE555P"), row(1, 0, 60, "93M")]
    picks, problem = assign(rows, parse_expected("NE555P | 93M | DNJW26 G4"))
    assert picks == [] and "2 line(s)" in problem


def test_a_line_with_too_few_segments_skips_the_photo():
    rows = [row(0, 0, 10, "NE555P"), row(1, 0, 60, "93M"), row(2, 0, 10, "DNJW26G4")]
    picks, problem = assign(rows, parse_expected("NE555P | 93M | DNJW26 G4"))
    assert picks == [] and "line 2" in problem


def test_harvest_writes_crops_and_appends_labels_once(tmp_path):
    photo = tmp_path / "raw" / "P1.jpg"
    photo.parent.mkdir()
    Image.new("L", (200, 300), 90).save(photo)
    rows = [row(0, 0, 10, "NE555P"), row(1, 0, 60, "93M")]
    for r in rows:
        r["image"] = str(photo)
    ocr = tmp_path / "ocr"
    labels = ocr / "train.txt"
    out = ocr / "images" / "harvest"

    written, skipped = harvest([photo], parse_expected("NE555P | 93M"), out, labels,
                               predict=lambda images: rows, log=lambda *_: None)
    assert (written, skipped) == (2, 0)
    names = sorted(p.name for p in out.iterdir())
    # The folders lead the name: capture sessions reuse photo numbers.
    assert names == [f"{tmp_path.name}_raw_P1_l0p0.png", f"{tmp_path.name}_raw_P1_l1p0.png"]
    text = labels.read_text(encoding="utf-8")
    assert f"images/harvest/{names[0]}\tNE555P" in text
    assert f"images/harvest/{names[1]}\t93M" in text

    harvest([photo], parse_expected("NE555P | 93M"), out, labels,
            predict=lambda images: rows, log=lambda *_: None)
    assert labels.read_text(encoding="utf-8").count("P1_l0p0.png") == 1, "re-running does not duplicate"


def test_harvest_skips_and_counts_a_bad_photo(tmp_path):
    photo = tmp_path / "P2.jpg"
    Image.new("L", (200, 300), 90).save(photo)
    rows = [row(0, 0, 10, "NE555P")]
    rows[0]["image"] = str(photo)
    messages = []
    written, skipped = harvest([photo], parse_expected("NE555P | 93M"), tmp_path / "out",
                               tmp_path / "train.txt", predict=lambda images: rows, log=messages.append)
    assert (written, skipped) == (0, 1)
    assert messages and "skip P2.jpg" in messages[0]
    assert not (tmp_path / "train.txt").exists()
