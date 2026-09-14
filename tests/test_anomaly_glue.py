"""Version-tolerant anomalib glue + the pipeline's ROI cropping.

These cover the parts that break silently when a dependency changes shape, so
they must run without anomalib or torch installed where possible.
"""

import pytest

from icinspect.anomaly.common import _dir_has_images, _validate_images, filter_kwargs


def test_filter_kwargs_drops_unsupported_names():
    def target(a, b=1, c=2):
        return a, b, c

    assert filter_kwargs(target, {"a": 1, "c": 3, "zzz": 9}) == {"a": 1, "c": 3}


def test_filter_kwargs_keeps_everything_for_var_keyword():
    def target(a, **kwargs):
        return a, kwargs

    passed = {"a": 1, "anything": 2}
    assert filter_kwargs(target, passed) == passed


def test_filter_kwargs_survives_uninspectable_callables():
    # Some C builtins expose no signature at all; the helper must pass kwargs
    # through rather than crash (min raises ValueError, a non-callable TypeError).
    assert filter_kwargs(min, {"x": 1}) == {"x": 1}
    assert filter_kwargs(42, {"x": 1}) == {"x": 1}


def test_dir_has_images_is_extension_and_case_aware(tmp_path):
    exts = [".png", ".jpg"]
    assert not _dir_has_images(tmp_path, exts)
    assert not _dir_has_images(tmp_path / "missing", exts)

    (tmp_path / "notes.txt").write_text("x", encoding="utf-8")
    assert not _dir_has_images(tmp_path, exts)

    nested = tmp_path / "sub"
    nested.mkdir()
    (nested / "a.PNG").write_bytes(b"")
    assert _dir_has_images(tmp_path, exts), "should recurse and ignore case"


def test_validate_images_reports_truncated_files(tmp_path):
    Image = pytest.importorskip("PIL.Image")

    valid = tmp_path / "valid.jpg"
    Image.new("RGB", (32, 32), "white").save(valid, quality=95)
    broken = tmp_path / "broken.jpg"
    broken.write_bytes(valid.read_bytes()[:-20])

    with pytest.raises(ValueError, match="broken.jpg"):
        _validate_images(tmp_path, [".jpg"])


def test_rows_from_batch_handles_dict_and_object():
    pytest.importorskip("torch", reason="infer module imports torch via anomalib helpers")
    from icinspect.anomaly.infer_patchcore import _rows_from_batch

    class Batch:
        image_path = ["a.png", "b.png"]
        pred_score = [0.1, 0.9]
        pred_label = [0, 1]

    for batch in (Batch(), {"image_path": Batch.image_path, "pred_score": Batch.pred_score,
                            "pred_label": Batch.pred_label}):
        rows = _rows_from_batch(batch)
        assert [r["image"] for r in rows] == ["a.png", "b.png"]
        assert rows[1]["score"] == pytest.approx(0.9)
        assert rows[1]["pred_label"] == 1


def test_rows_from_batch_tolerates_missing_scores():
    pytest.importorskip("torch")
    from icinspect.anomaly.infer_patchcore import _rows_from_batch

    rows = _rows_from_batch({"image_path": ["only.png"]})
    assert rows == [{"image": "only.png", "score": None, "pred_label": None}]


def test_crop_roi_writes_the_expected_region(tmp_path):
    pytest.importorskip("torch", reason="pipeline imports the OCR dataset module")
    Image = pytest.importorskip("PIL.Image")
    from icinspect.pipeline import crop_roi

    src = tmp_path / "board.png"
    Image.new("L", (400, 200), 100).save(src)
    out = crop_roi(src, (0.25, 0.5, 0.75, 1.0), tmp_path / "roi")

    assert out.exists()
    assert Image.open(out).size == (200, 100)


def test_crop_roi_rejects_inverted_box(tmp_path):
    pytest.importorskip("torch")
    Image = pytest.importorskip("PIL.Image")
    from icinspect.pipeline import crop_roi

    src = tmp_path / "board.png"
    Image.new("L", (100, 100), 100).save(src)
    with pytest.raises(ValueError, match="Invalid ROI"):
        crop_roi(src, (0.8, 0.1, 0.2, 0.9), tmp_path / "roi")


def test_crop_roi_does_not_collide_on_shared_basenames(tmp_path):
    """test/good/ic_000.png and test/defect/ic_000.png must not overwrite each other."""
    pytest.importorskip("torch")
    Image = pytest.importorskip("PIL.Image")
    from icinspect.pipeline import crop_roi

    sources = []
    for i, (folder, tone) in enumerate((("good", 40), ("defect", 200))):
        d = tmp_path / folder
        d.mkdir()
        src = d / "ic_000.png"
        Image.new("L", (100, 100), tone).save(src)
        sources.append((src, tone, i))

    out_dir = tmp_path / "roi"
    crops = [(crop_roi(src, (0.0, 0.0, 1.0, 1.0), out_dir, index=i), tone) for src, tone, i in sources]

    assert len({c for c, _ in crops}) == 2, "crops must land in distinct files"
    for crop, tone in crops:
        assert Image.open(crop).getpixel((50, 50)) == tone, "crop content belongs to the wrong source"
