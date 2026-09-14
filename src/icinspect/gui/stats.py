"""What the Stats tab knows, with no window attached.

Three sources, none of which needs a model loaded:

* `results/ocr/metrics.json` -- the reader's per-epoch training history, written
  by `icinspect.ocr.train`;
* `results/patchcore/train_summary.json` -- the detector's test AUROC and F1;
* the checkpoints themselves, for the geometry and hyperparameters that were
  actually trained with rather than whatever the config says today.

Everything here returns plain numbers and strings, so the charts and the facts
table can be tested without opening a window or a GPU.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from icinspect.config import load_yaml

Fact = tuple[str, str]


def _read_json(path: Path) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
        return data if isinstance(data, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def _short(path: Path) -> str:
    """A checkpoint under the project shows as a relative path; the column is narrow."""
    from icinspect.config import PROJECT_ROOT

    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def _when(path: Path) -> str:
    return datetime.fromtimestamp(path.stat().st_mtime).strftime("%Y-%m-%d %H:%M")


def _size(path: Path) -> str:
    mb = path.stat().st_size / (1024 * 1024)
    return f"{mb:.1f} MB"


@dataclass
class TrainingHistory:
    """The reader's training run, epoch by epoch."""

    epochs: list[int] = field(default_factory=list)
    loss: list[float] = field(default_factory=list)
    char_accuracy: list[float] = field(default_factory=list)
    seq_accuracy: list[float] = field(default_factory=list)
    seconds: list[float] = field(default_factory=list)
    best_score: float | None = None

    def __bool__(self) -> bool:
        return bool(self.epochs)

    @property
    def total_minutes(self) -> float:
        return sum(self.seconds) / 60.0


def ocr_history(ocr_dir: Path) -> TrainingHistory:
    """Read `metrics.json`; an interrupted run leaves partial epochs, which is fine."""
    data = _read_json(ocr_dir / "metrics.json")
    rows = [r for r in data.get("history", []) if isinstance(r, dict)]
    rows.sort(key=lambda r: r.get("epoch", 0))
    history = TrainingHistory(best_score=data.get("best_score"))
    for row in rows:
        history.epochs.append(int(row.get("epoch", len(history.epochs) + 1)))
        history.loss.append(float(row.get("loss", float("nan"))))
        history.char_accuracy.append(float(row.get("char_accuracy", float("nan"))))
        history.seq_accuracy.append(float(row.get("seq_accuracy", float("nan"))))
        history.seconds.append(float(row.get("seconds", 0.0)))
    return history


def ocr_facts(ckpt: Path | None, config_name: str = "ocr.yaml") -> list[Fact]:
    """Everything worth knowing about the marking reader on disk."""
    if ckpt is None:
        return [("model", "not trained yet")]
    import torch

    state = torch.load(ckpt, map_location="cpu", weights_only=False)
    cfg = state.get("config") or {}
    charset = state.get("charset") or {}
    image = cfg.get("image", {})
    model = cfg.get("model", {})
    metrics = state.get("metrics") or {}
    live = load_yaml(config_name)
    decode = live.get("decode") or {}

    params = sum(int(t.numel()) for t in (state.get("model") or {}).values())
    symbols = charset.get("symbols", "")
    width, height = int(image.get("width", 160)), int(image.get("height", 32))
    facts: list[Fact] = [
        ("checkpoint", _short(ckpt)),
        ("trained", _when(ckpt)),
        ("size on disk", _size(ckpt)),
        ("parameters", f"{params:,}"),
        ("input", f"{width} x {height} px, {'grayscale' if image.get('grayscale', True) else 'colour'}"),
        ("CTC frames", str(width // 4 - 1)),
        ("BiLSTM", f"{model.get('rnn_layers', 2)} x {model.get('rnn_hidden', 256)} hidden, "
                   f"dropout {model.get('dropout', 0)}"),
        ("charset", f"{len(symbols)} symbols + CTC blank"),
        ("best epoch", str(state.get("epoch", "-"))),
    ]
    if metrics:
        facts += [
            ("val char accuracy", f"{metrics.get('char_accuracy', float('nan')):.4f}"),
            ("val sequence accuracy", f"{metrics.get('seq_accuracy', float('nan')):.4f}"),
            ("val samples", f"{int(metrics.get('samples', 0)):,}"),
        ]
    # These two are read at inference time, so they describe the next run
    # rather than the training run -- worth seeing side by side with it.
    facts += [
        ("min char confidence", str(decode.get("min_char_confidence", 0.0))),
        ("max unsure fraction", str(decode.get("max_unsure_frac", 1.0))),
    ]
    return facts


def patchcore_facts(ckpt: Path | None, results_dir: Path, config_name: str = "patchcore.yaml") -> list[Fact]:
    """The detector's hyperparameters as trained, plus its test scores."""
    cfg = load_yaml(config_name)
    model = cfg.get("model", {})
    image = cfg.get("image", {})
    facts: list[Fact] = []
    if ckpt is None:
        return [("model", "not trained yet")]

    summary = _read_json(results_dir / "train_summary.json")
    trained_cfg: dict[str, Any] = {}
    try:
        import torch

        state = torch.load(ckpt, map_location="cpu", weights_only=False)
        trained_cfg = state.get("hyper_parameters") or {}
    except Exception:      # a checkpoint from another anomalib version still lists
        trained_cfg = {}   # everything else; the config values below stand in

    def trained(name: str, fallback: Any) -> str:
        return str(trained_cfg.get(name, fallback))

    facts += [
        ("checkpoint", _short(ckpt)),
        ("trained", _when(ckpt)),
        ("size on disk", _size(ckpt)),
        ("backbone", trained("backbone", model.get("backbone", "-"))),
        ("layers", trained("layers", model.get("layers", "-"))),
        ("coreset ratio", trained("coreset_sampling_ratio", model.get("coreset_sampling_ratio", "-"))),
        ("neighbours", trained("num_neighbors", model.get("num_neighbors", "-"))),
        ("input", "{} x {} px".format(*(image.get("size") or ["?", "?"]))),
    ]
    for row in summary.get("test") or []:
        for key, value in row.items():
            if isinstance(value, (int, float)):
                facts.append((f"test {key.replace('image_', 'image ')}", f"{value:.4f}"))
    pinned = (cfg.get("threshold") or {}).get("image_score")
    facts.append(("threshold", "pinned at " + str(pinned) if pinned is not None
                  else "adaptive (normalised to 0.50)"))
    return facts


@dataclass
class Confusion:
    """The 2x2 of verdict against ground truth, defect being positive."""

    tp: int = 0     # failed, and it was defective
    fp: int = 0     # failed a good part
    tn: int = 0     # passed, and it was good
    fn: int = 0     # passed a defective part -- the one that ships

    def __bool__(self) -> bool:
        return self.labelled > 0

    @property
    def labelled(self) -> int:
        return self.tp + self.fp + self.tn + self.fn

    @property
    def accuracy(self) -> float | None:
        return None if not self.labelled else (self.tp + self.tn) / self.labelled

    @property
    def precision(self) -> float | None:
        """Of the parts failed, how many deserved it."""
        called = self.tp + self.fp
        return None if not called else self.tp / called

    @property
    def recall(self) -> float | None:
        """Of the defective parts, how many were caught."""
        actual = self.tp + self.fn
        return None if not actual else self.tp / actual

    @property
    def specificity(self) -> float | None:
        """Of the good parts, how many were left alone."""
        actual = self.tn + self.fp
        return None if not actual else self.tn / actual

    @property
    def f1(self) -> float | None:
        precision, recall = self.precision, self.recall
        if precision is None or recall is None or precision + recall == 0:
            return None
        return 2 * precision * recall / (precision + recall)

    def as_grid(self) -> list[list[int]]:
        """[[TN, FP], [FN, TP]] -- rows are the truth, columns the verdict."""
        return [[self.tn, self.fp], [self.fn, self.tp]]


@dataclass
class RunStats:
    """Summary of the inspection currently on screen."""

    total: int = 0
    passed: int = 0
    failed: int = 0
    scores: list[float] = field(default_factory=list)
    confidences: list[float] = field(default_factory=list)
    peaks: list[float] = field(default_factory=list)
    areas: list[float] = field(default_factory=list)
    ious: list[float] = field(default_factory=list)
    reasons: Counter = field(default_factory=Counter)
    confusion: Confusion = field(default_factory=Confusion)

    def __bool__(self) -> bool:
        return self.total > 0

    @property
    def pass_rate(self) -> float | None:
        return None if not self.total else self.passed / self.total

    @property
    def mean_score(self) -> float | None:
        return sum(self.scores) / len(self.scores) if self.scores else None

    @property
    def mean_confidence(self) -> float | None:
        return sum(self.confidences) / len(self.confidences) if self.confidences else None

    @property
    def mean_iou(self) -> float | None:
        return sum(self.ious) / len(self.ious) if self.ious else None


def run_stats(results) -> RunStats:
    """Fold a list of `service.Result` into the numbers the tab plots."""
    stats = RunStats(total=len(results))
    for result in results:
        if result.verdict == "FAIL":
            stats.failed += 1
        else:
            stats.passed += 1
        # The deciding score, not always anomalib's image score: with the
        # background excluded the verdict comes from the part alone, and a
        # histogram of the other number would not explain the verdicts.
        deciding = getattr(result, "deciding_score", result.score)
        if deciding is not None:
            stats.scores.append(float(deciding))
        if result.reads:
            stats.confidences.append(float(result.confidence))
        if result.peak_score is not None:
            stats.peaks.append(float(result.peak_score))
        # The share of the *part* that is damaged, which is what an operator
        # grades by; the frame share is mostly fixture.
        ratio = getattr(result, "damage_ratio", None)
        if ratio is not None:
            stats.areas.append(float(ratio))
        elif result.defect_area is not None:
            stats.areas.append(float(result.defect_area))
        for reason in result.reasons:
            stats.reasons[reason] += 1

        iou = getattr(result, "iou", None)
        if iou is not None:
            stats.ious.append(float(iou))

        # Only photos whose folder says what they are can score a verdict.
        truth = getattr(result, "truth", None)
        if truth is None:
            continue
        predicted = getattr(result, "predicted", int(result.verdict == "FAIL"))
        if truth and predicted:
            stats.confusion.tp += 1
        elif truth and not predicted:
            stats.confusion.fn += 1
        elif predicted:
            stats.confusion.fp += 1
        else:
            stats.confusion.tn += 1
    return stats
