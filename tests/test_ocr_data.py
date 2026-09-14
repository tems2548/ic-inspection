"""Label parsing, image normalisation and the synthetic generator (PIL only)."""

import pytest

from icinspect.ocr.charset import Charset
from icinspect.ocr.dataset import (
    filter_encodable,
    list_images,
    load_image,
    read_label_file,
    split_samples,
)

Image = pytest.importorskip("PIL.Image", reason="pillow required")

SYMBOLS = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ-+./ "


def make_dataset(tmp_path, entries):
    images = tmp_path / "images"
    images.mkdir(exist_ok=True)
    lines = []
    for name, label, size in entries:
        Image.new("L", size, 128).save(images / name)
        lines.append(f"images/{name}\t{label}")
    label_file = tmp_path / "train.txt"
    label_file.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return label_file


def test_read_label_file_parses_tsv(tmp_path):
    label_file = make_dataset(tmp_path, [("a.png", "STM32F4", (80, 20)), ("b.png", "LM358N", (60, 18))])
    samples = read_label_file(label_file)
    assert [s.text for s in samples] == ["STM32F4", "LM358N"]
    assert all(s.image_path.exists() for s in samples)


def test_read_label_file_skips_bad_lines_but_keeps_good_ones(tmp_path, capsys):
    label_file = make_dataset(tmp_path, [("a.png", "OK1", (60, 20))])
    label_file.write_text(
        label_file.read_text(encoding="utf-8")
        + "# a comment\n\nimages/missing.png\tGHOST\nnoseparator\n",
        encoding="utf-8",
    )
    samples = read_label_file(label_file)
    assert [s.text for s in samples] == ["OK1"]
    assert "skipped 2 line(s)" in capsys.readouterr().out


def test_read_label_file_raises_when_nothing_usable(tmp_path):
    label_file = tmp_path / "empty.txt"
    label_file.write_text("# only comments\n", encoding="utf-8")
    with pytest.raises(ValueError, match="No usable samples"):
        read_label_file(label_file)


def test_missing_label_file_is_explicit(tmp_path):
    with pytest.raises(FileNotFoundError):
        read_label_file(tmp_path / "nope.txt")


def test_filter_encodable_drops_out_of_charset(tmp_path, capsys):
    label_file = make_dataset(tmp_path, [("a.png", "GOOD1", (60, 20)), ("b.png", "BAD_1", (60, 20))])
    kept = filter_encodable(read_label_file(label_file), Charset(SYMBOLS))
    assert [s.text for s in kept] == ["GOOD1"]
    assert "_" in capsys.readouterr().out


@pytest.mark.parametrize("size", [(200, 50), (20, 50), (160, 32), (10, 10)])
def test_load_image_always_returns_target_shape(tmp_path, size):
    path = tmp_path / "x.png"
    Image.new("L", size, 90).save(path)
    img = load_image(path, height=32, width=160)
    assert img.size == (160, 32)
    assert img.mode == "L"


def test_load_image_pads_with_border_tone_not_black(tmp_path):
    path = tmp_path / "narrow.png"
    Image.new("L", (20, 32), 200).save(path)   # bright, much narrower than the target
    img = load_image(path, height=32, width=160)
    assert img.getpixel((150, 16)) > 150       # padded region kept the bright tone


def test_split_samples_is_disjoint_and_deterministic(tmp_path):
    label_file = make_dataset(tmp_path, [(f"{i}.png", f"LBL{i}", (60, 20)) for i in range(20)])
    samples = read_label_file(label_file)
    train, val = split_samples(samples, 0.25, seed=7)
    assert len(val) == 5 and len(train) == 15
    assert not ({s.image_path for s in train} & {s.image_path for s in val})
    assert split_samples(samples, 0.25, seed=7)[1] == val


def test_split_samples_never_empties_train(tmp_path):
    label_file = make_dataset(tmp_path, [("a.png", "A1", (60, 20)), ("b.png", "B2", (60, 20))])
    train, val = split_samples(read_label_file(label_file), 0.9, seed=1)
    assert len(train) >= 1 and len(val) >= 1


def test_list_images_finds_files_and_accepts_single_file(tmp_path):
    (tmp_path / "sub").mkdir()
    Image.new("L", (10, 10)).save(tmp_path / "a.png")
    Image.new("L", (10, 10)).save(tmp_path / "sub" / "b.jpg")
    (tmp_path / "notes.txt").write_text("ignore me", encoding="utf-8")
    assert len(list_images(tmp_path)) == 2
    assert list_images(tmp_path / "a.png") == [tmp_path / "a.png"]


def test_synth_generates_matching_images_and_labels(tmp_path):
    from icinspect.ocr.synth import generate, load_fonts

    fonts = load_fonts("C:/Windows/Fonts", ["arial.ttf"])
    out_dir = tmp_path / "synth"
    label_file = tmp_path / "labels.txt"
    n = generate(out_dir, label_file, count=6, fonts=fonts, height=32, seed=3, relative_to=tmp_path)

    assert n == 6
    charset = Charset(SYMBOLS)
    samples = read_label_file(label_file, [tmp_path])
    assert len(samples) == 6
    for s in samples:
        assert charset.is_encodable(s.text), f"synth produced {s.text!r} outside the charset"
        assert 3 <= len(s.text) <= 12
        assert Image.open(s.image_path).height == 32


def test_synth_is_reproducible(tmp_path):
    from icinspect.ocr.synth import generate

    texts = []
    for run in ("a", "b"):
        lf = tmp_path / f"{run}.txt"
        generate(tmp_path / run, lf, count=4, fonts=[], height=32, seed=99, relative_to=tmp_path)
        texts.append(lf.read_text(encoding="utf-8"))
    assert texts[0].replace("a/", "") == texts[1].replace("b/", "")
