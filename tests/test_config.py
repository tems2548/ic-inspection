"""Config loading and path resolution."""

from pathlib import Path

import pytest
import yaml

from icinspect.config import CONFIG_DIR, Paths, load_yaml, resolve_output, resolve_paths


def test_shipped_configs_parse():
    for name in ("paths.yaml", "patchcore.yaml", "ocr.yaml"):
        assert isinstance(load_yaml(name), dict), name


def test_patchcore_config_has_required_keys():
    cfg = load_yaml("patchcore.yaml")
    assert cfg["dataset"]["normal_dir"]
    assert cfg["model"]["backbone"]
    assert len(cfg["image"]["size"]) == 2


def test_ocr_charset_has_no_duplicates():
    symbols = load_yaml("ocr.yaml")["charset"]["symbols"]
    assert len(set(symbols)) == len(symbols)


def test_ocr_width_fits_longest_synthetic_label():
    cfg = load_yaml("ocr.yaml")
    max_len = cfg["synth"]["max_len"]
    # CRNN emits width//4 - 1 CTC frames; CTC needs at least one frame per character.
    assert cfg["image"]["width"] // 4 - 1 >= max_len


def test_env_var_overrides_dataset_root(tmp_path, monkeypatch):
    monkeypatch.setenv("IC_DATASET_ROOT", str(tmp_path))
    paths = resolve_paths()
    assert paths.dataset_root == tmp_path
    assert paths.anomaly_root == tmp_path / "anomaly" / "ic"
    assert paths.ocr_images == tmp_path / "ocr" / "images"
    assert paths.labels_file("train") == tmp_path / "ocr" / "train.txt"


def test_results_dir_stays_inside_the_project(monkeypatch, tmp_path):
    monkeypatch.setenv("IC_DATASET_ROOT", str(tmp_path))
    paths = resolve_paths()
    assert paths.results_dir.is_absolute()
    assert paths.results_dir.parent == CONFIG_DIR.parent   # project root, not the dataset


def test_require_reports_missing_locations(tmp_path):
    paths = Paths(
        dataset_root=tmp_path,
        anomaly_root=tmp_path / "nope",
        ocr_root=tmp_path / "ocr",
        raw_root=tmp_path / "raw",
        results_dir=tmp_path / "results",
    )
    with pytest.raises(FileNotFoundError, match="init_dataset"):
        paths.require("anomaly_root")


def test_load_yaml_rejects_non_mapping(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text("- just\n- a list\n", encoding="utf-8")
    with pytest.raises(ValueError, match="mapping"):
        load_yaml(bad)


def test_relative_output_resolves_under_project():
    assert resolve_output("results/ocr").is_absolute()
    assert Path(resolve_output("results/ocr")).name == "ocr"


def test_configs_are_utf8_and_loadable_directly():
    for path in CONFIG_DIR.glob("*.yaml"):
        yaml.safe_load(path.read_text(encoding="utf-8"))
