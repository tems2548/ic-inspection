"""Shared anomalib glue.

anomalib's public API has moved between 1.x and 2.x (image_size left the
datamodule, PreProcessor arrived on the model). Rather than pinning ourselves to
one shape, we introspect the signatures and pass only what the installed version
accepts.
"""

from __future__ import annotations

import inspect
from pathlib import Path
from typing import Any

from icinspect.config import Paths


def filter_kwargs(target: Any, kwargs: dict[str, Any]) -> dict[str, Any]:
    """Drop keys the installed version's signature does not accept."""
    try:
        params = inspect.signature(target).parameters
    except (TypeError, ValueError):
        return dict(kwargs)
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()):
        return dict(kwargs)
    return {k: v for k, v in kwargs.items() if k in params}


def _dir_has_images(path: Path, extensions: list[str]) -> bool:
    if not path.is_dir():
        return False
    exts = {e.lower() for e in extensions}
    return any(p.suffix.lower() in exts for p in path.rglob("*") if p.is_file())


def _image_stems(path: Path, extensions: list[str]) -> dict[str, Path]:
    exts = {e.lower() for e in extensions}
    return {p.stem: p for p in sorted(path.rglob("*")) if p.is_file() and p.suffix.lower() in exts}


def _mask_dir_problem(abnormal_dir: Path, mask_dir: Path, extensions: list[str]) -> str | None:
    """Return why mask_dir cannot be handed to anomalib, or None if it is usable."""
    masks = _image_stems(mask_dir, extensions)
    if not masks:
        return "no mask images found"
    images = _image_stems(abnormal_dir, extensions)
    missing = sorted(set(images) - set(masks))
    extra = sorted(set(masks) - set(images))
    if not missing and not extra:
        return None
    parts = [f"{len(masks)} masks for {len(images)} defect images"]
    if missing:
        parts.append(f"{len(missing)} defect images without a mask (e.g. {missing[0]})")
    if extra:
        parts.append(f"{len(extra)} masks with no matching defect image (e.g. {extra[0]})")
    return "; ".join(parts) + ". Masks must be one binary image per defect photo, same file stem."


def _validate_images(path: Path, extensions: list[str]) -> None:
    """Fail before training if an image cannot be fully decoded."""
    from PIL import Image

    exts = {e.lower() for e in extensions}
    unreadable: list[str] = []
    for image_path in path.rglob("*"):
        if not image_path.is_file() or image_path.suffix.lower() not in exts:
            continue
        try:
            with Image.open(image_path) as image:
                image.load()
        except (OSError, ValueError) as exc:
            unreadable.append(f"{image_path} ({exc})")

    if unreadable:
        details = "\n".join(f"  - {item}" for item in unreadable)
        raise ValueError(
            "Unreadable images found in the PatchCore training set. "
            "Replace or remove them, then retry:\n" + details
        )


def build_datamodule(cfg: dict[str, Any], paths: Paths):
    """Create an anomalib Folder datamodule from configs/patchcore.yaml."""
    from anomalib.data import Folder

    ds = cfg["dataset"]
    root = paths.anomaly_root
    exts = list(ds.get("extensions", [".png", ".jpg", ".jpeg", ".bmp"]))

    normal_dir = root / ds["normal_dir"]
    if not _dir_has_images(normal_dir, exts):
        raise FileNotFoundError(
            f"No training images found in {normal_dir}\n"
            "PatchCore needs defect-free images there. Run scripts/init_dataset.py, "
            "then copy your good IC photos in."
        )
    _validate_images(normal_dir, exts)

    kwargs: dict[str, Any] = {
        "name": "ic",
        "root": root,
        "normal_dir": ds["normal_dir"],
        "extensions": exts,
        "train_batch_size": ds.get("train_batch_size", 8),
        "eval_batch_size": ds.get("eval_batch_size", 8),
        "num_workers": ds.get("num_workers", 0),
        "seed": ds.get("seed", 42),
    }

    # Optional test folders: only pass them once they actually hold images,
    # otherwise anomalib builds an empty test set and metrics come out as NaN.
    for key in ("abnormal_dir", "normal_test_dir"):
        rel = ds.get(key)
        if rel and _dir_has_images(root / rel, exts):
            kwargs[key] = rel

    # anomalib pairs masks with abnormal images by position, so the mask folder
    # must hold exactly one file per abnormal image with the same stem. Anything
    # else makes Folder() crash deep inside pandas; skip it with a warning instead.
    mask_rel = ds.get("mask_dir")
    if mask_rel and "abnormal_dir" in kwargs:
        problem = _mask_dir_problem(root / kwargs["abnormal_dir"], root / mask_rel, exts + [".png"])
        if problem is None:
            kwargs["mask_dir"] = mask_rel
        elif (root / mask_rel).is_dir():
            print(f"[patchcore] ignoring mask_dir {mask_rel}: {problem}")

    # anomalib 1.x accepted image_size on the datamodule; 2.x does not.
    size = cfg.get("image", {}).get("size")
    if size:
        kwargs["image_size"] = tuple(size)

    return Folder(**filter_kwargs(Folder.__init__, kwargs))


def build_pre_processor(cfg: dict[str, Any]):
    """Resize/normalize transform for anomalib >= 2.0. Returns None on 1.x."""
    size = cfg.get("image", {}).get("size")
    if not size:
        return None
    try:
        from anomalib.pre_processing import PreProcessor
    except ImportError:
        try:
            from anomalib.pre_processing.pre_processor import PreProcessor  # type: ignore
        except ImportError:
            return None

    from torchvision.transforms.v2 import CenterCrop, Compose, Normalize, Resize

    steps: list[Any] = [Resize(tuple(size), antialias=True)]
    crop = cfg.get("image", {}).get("center_crop")
    if crop:
        steps.append(CenterCrop(tuple(crop)))
    steps.append(Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
    return PreProcessor(transform=Compose(steps))


def build_model(cfg: dict[str, Any]):
    from anomalib.models import Patchcore

    m = cfg["model"]
    kwargs: dict[str, Any] = {
        "backbone": m.get("backbone", "wide_resnet50_2"),
        "layers": list(m.get("layers", ["layer2", "layer3"])),
        "pre_trained": m.get("pre_trained", True),
        "coreset_sampling_ratio": m.get("coreset_sampling_ratio", 0.1),
        "num_neighbors": m.get("num_neighbors", 9),
    }
    pre = build_pre_processor(cfg)
    if pre is not None:
        kwargs["pre_processor"] = pre
    return Patchcore(**filter_kwargs(Patchcore.__init__, kwargs))


def load_patchcore(ckpt_path: Path | str, cfg: dict[str, Any] | None = None):
    """Load a checkpoint written by train_patchcore.py.

    anomalib saves its PreProcessor/PostProcessor objects into the checkpoint, and
    torch >= 2.6 refuses to unpickle those under the default weights_only=True. The
    restricted loader is tried first; the fallback is safe only because these files
    are produced locally by this project -- do not point it at an untrusted .ckpt.
    """
    import pickle

    from anomalib.models import Patchcore

    pre = build_pre_processor(cfg) if cfg else None
    kwargs = filter_kwargs(Patchcore.__init__, {"pre_processor": pre} if pre else {})

    try:
        return Patchcore.load_from_checkpoint(str(ckpt_path), **kwargs)
    except (pickle.UnpicklingError, RuntimeError) as exc:
        if "eights only" not in str(exc):
            raise
        return Patchcore.load_from_checkpoint(str(ckpt_path), weights_only=False, **kwargs)


def build_engine(cfg: dict[str, Any], default_root_dir: Path, report_progress: bool = False):
    """The anomalib Engine. `report_progress` adds a callback that prints a
    `[progress]` line per batch (see `icinspect.progress`) for the window's
    bar -- lightning's own bar only reaches a pipe when a stage ends."""
    from anomalib.engine import Engine

    t = cfg.get("trainer", {})
    kwargs = {
        "accelerator": t.get("accelerator", "auto"),
        "devices": t.get("devices", 1),
        "max_epochs": t.get("max_epochs", 1),
        "default_root_dir": str(default_root_dir),
    }
    if report_progress:
        kwargs["callbacks"] = [progress_callback()]
    return Engine(**filter_kwargs(Engine.__init__, kwargs))


def progress_callback():
    """A lightning Callback printing `[progress]` per batch of each stage.

    PatchCore's fit is one pass extracting features (one counter), then the
    coreset selection, which has no batches and is announced as a status so
    the bar does not look stuck, then validation and test with counters of
    their own. Each stage restarts the count; the window restarts its ETA.
    """
    from lightning.pytorch.callbacks import Callback

    from icinspect.progress import Reporter

    class Report(Callback):
        def __init__(self) -> None:
            self._reporter: Reporter | None = None

        def _begin(self, total: int) -> None:
            self._reporter = Reporter(max(int(total), 1), every=0.5)

        def _tick(self, done: int, text: str) -> None:
            if self._reporter is not None:
                self._reporter.step(done, text, force=done >= self._reporter.total)

        def on_train_epoch_start(self, trainer, pl_module) -> None:
            self._begin(trainer.num_training_batches)

        def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx) -> None:
            self._tick(batch_idx + 1, f"extracting features {batch_idx + 1}/{trainer.num_training_batches}")

        def on_train_epoch_end(self, trainer, pl_module) -> None:
            print("[patchcore] selecting the coreset (memory bank)...", flush=True)

        def on_validation_epoch_start(self, trainer, pl_module) -> None:
            total = sum(int(n) for n in _as_list(trainer.num_val_batches))
            self._begin(total)

        def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0) -> None:
            self._tick(batch_idx + 1, f"validating {batch_idx + 1}/{self._reporter.total if self._reporter else '?'}")

        def on_test_epoch_start(self, trainer, pl_module) -> None:
            total = sum(int(n) for n in _as_list(trainer.num_test_batches))
            self._begin(total)

        def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0) -> None:
            self._tick(batch_idx + 1, f"testing {batch_idx + 1}/{self._reporter.total if self._reporter else '?'}")

    return Report()


def _as_list(value) -> list:
    return list(value) if isinstance(value, (list, tuple)) else [value]
