"""Config + path resolution.

Everything that touches the filesystem goes through here so the dataset stays
outside the repo and can be moved by editing configs/paths.yaml (or setting
IC_DATASET_ROOT).
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

# Packaged app (PyInstaller): configs/, results/ and scripts/ sit beside the exe.
FROZEN = bool(getattr(sys, "frozen", False))
PROJECT_ROOT = Path(sys.executable).resolve().parent if FROZEN else Path(__file__).resolve().parents[2]
CONFIG_DIR = PROJECT_ROOT / "configs"


def load_yaml(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    if not path.is_absolute():
        path = CONFIG_DIR / path
    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a YAML mapping, got {type(data).__name__}")
    return data


@dataclass(frozen=True)
class Paths:
    """Resolved absolute locations for data and results."""

    dataset_root: Path
    anomaly_root: Path
    ocr_root: Path
    raw_root: Path
    results_dir: Path

    @property
    def ocr_images(self) -> Path:
        return self.ocr_root / "images"

    @property
    def ocr_synth(self) -> Path:
        return self.ocr_root / "synth"

    def labels_file(self, split: str) -> Path:
        """<ocr_root>/train.txt | val.txt | test.txt"""
        return self.ocr_root / f"{split}.txt"

    def require(self, *which: str) -> None:
        """Fail loudly with an actionable message instead of a confusing empty dataset."""
        missing = [getattr(self, name) for name in which if not getattr(self, name).exists()]
        if missing:
            listed = "\n  ".join(str(p) for p in missing)
            raise FileNotFoundError(
                f"Missing dataset location(s):\n  {listed}\n"
                "Run: python scripts/init_dataset.py"
            )


def resolve_paths(paths_cfg: str | Path = "paths.yaml") -> Paths:
    cfg = load_yaml(paths_cfg)

    # Env var wins so you can point at a second dataset without editing configs.
    root = Path(os.environ.get("IC_DATASET_ROOT", cfg["dataset_root"])).expanduser()

    results = Path(cfg.get("results_dir", "results"))
    if not results.is_absolute():
        results = PROJECT_ROOT / results

    return Paths(
        dataset_root=root,
        anomaly_root=root / cfg.get("anomaly_subdir", "anomaly/ic"),
        ocr_root=root / cfg.get("ocr_subdir", "ocr"),
        raw_root=root / cfg.get("raw_subdir", "raw"),
        results_dir=results,
    )


def resolve_output(path_value: str | Path) -> Path:
    """Interpret a config output path relative to the project root."""
    p = Path(path_value)
    return p if p.is_absolute() else PROJECT_ROOT / p
