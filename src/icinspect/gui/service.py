"""The GUI's inspection engine: same checks as `icinspect.pipeline`, kept warm.

Two differences from the CLI path, both for the sake of the window:

* models stay loaded between runs -- clicking Inspect twice should not pay for
  a PatchCore memory bank twice;
* every segment keeps the box it was cut from, so the reads can be drawn back
  onto the photo instead of only listed as text.

Torch and anomalib are imported inside the methods that need them: opening the
GUI must not wait ten seconds for CUDA.
"""

from __future__ import annotations

import csv
import json
import os
import shutil
import tempfile
import time
from datetime import datetime
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from icinspect.anomaly.shape import Shape
from icinspect.config import load_yaml, resolve_output, resolve_paths
from icinspect.gui.heatmap import peak
from icinspect.gui.report import auroc
from icinspect.gui.runner import Progress
from icinspect.ocr.charset import cer
from icinspect.ocr.dataset import list_images

Box = tuple[int, int, int, int]


PIXEL_THRESHOLD = 0.5   # anomalib normalises its pixel map so the threshold lands here
MASK_SUFFIXES = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")

# Where the scored region comes from.
REGION_REFERENCE = "normal part"   # fixed, learned from the good training parts
REGION_PHOTO = "this photo"        # measured on the photo being inspected
REGION_MANUAL = "drawn box"        # fixed, drawn by hand on a photo in the window
REGION_MODES = (REGION_REFERENCE, REGION_PHOTO, REGION_MANUAL)
REFERENCE_FILE = "part_region.json"
MANUAL_FILE = "manual_region.json"

Region = tuple[float, float, float, float]   # x1, y1, x2, y2 as fractions of the frame


def _box(text: str) -> Box:
    x1, y1, x2, y2 = (int(v) for v in text.split(","))
    return (x1, y1, x2, y2)


def device_name() -> str:
    """The hardware an inspection ran on, for the latency row of the report."""
    try:
        import torch

        if torch.cuda.is_available():
            return torch.cuda.get_device_name(0)
    except Exception:
        pass
    import platform

    return f"CPU ({platform.processor() or platform.machine()})"


def _readable(images: list[Path], prog: Progress) -> list[Path]:
    """The photos that decode in full; the rest are named in the log and dropped."""
    from PIL import Image

    good: list[Path] = []
    for image in images:
        try:
            with Image.open(image) as img:
                img.load()
        except Exception as exc:
            prog.log(f"[anomaly] skipping {image.name}: {exc}")
            continue
        good.append(image)
    return good


def _region_pixels(region: "Region | None", image: Path) -> Box | None:
    """A frame-fraction region in this photo's pixels; None when there is none."""
    if region is None:
        return None
    from PIL import Image

    try:
        with Image.open(image) as img:
            width, height = img.size
    except Exception:
        return None
    x1, y1, x2, y2 = region
    return (int(x1 * width), int(y1 * height), int(x2 * width), int(y2 * height))


@contextmanager
def _staging(images: list[Path]):
    """Yield (folders to score, {scored path: original path}).

    When the selection is everything in its folders, those folders are handed
    to anomalib directly. When it is a subset -- one photo out of a capture run
    -- the files are hardlinked into a temporary folder so nothing else gets
    scored. Links are free and same-volume; a copy is the fallback for the rest.
    """
    folders = sorted({p.parent for p in images})
    in_folders = sum(len(list_images(folder)) for folder in folders)
    if in_folders <= len(images):
        yield folders, {str(p.resolve()): str(p.resolve()) for p in images}
        return

    tmp = Path(tempfile.mkdtemp(prefix="icinspect-"))
    back: dict[str, str] = {}
    try:
        for index, source in enumerate(images):
            # Photos from different folders share basenames all the time
            # (test/good/ic_000.png vs test/defect/ic_000.png).
            staged = tmp / f"{index:04d}_{source.name}"
            try:
                os.link(source, staged)
            except OSError:
                shutil.copy2(source, staged)
            back[str(staged.resolve())] = str(source.resolve())
        yield [tmp], back
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def _maps_from_batch(batch: Any) -> list[dict[str, Any]]:
    """Per-image `anomaly_map` and `pred_mask`, one entry per item in the batch.

    anomalib 2.x returns an ImageBatch dataclass and 1.x a dict; either may
    leave the maps out entirely (a model that scores without localising), which
    is why every field here is optional rather than assumed.
    """
    get = batch.get if isinstance(batch, dict) else (lambda k, d=None: getattr(batch, k, d))
    heat = get("anomaly_map")
    mask = get("pred_mask")
    if heat is None:
        return []

    heat = np.asarray(heat.detach().cpu() if hasattr(heat, "detach") else heat, dtype=np.float32)
    if heat.ndim == 2:                      # a lone image, not a batch of one
        heat = heat[None, ...]
    masks = None
    if mask is not None:
        masks = np.asarray(mask.detach().cpu() if hasattr(mask, "detach") else mask)
        if masks.ndim == 2:
            masks = masks[None, ...]

    from icinspect.gui.heatmap import to_uint8

    out: list[dict[str, Any]] = []
    for index in range(heat.shape[0]):
        out.append({
            "heat": to_uint8(heat[index]),
            "mask": None if masks is None else masks[index].astype(bool),
        })
    return out


@dataclass
class Segment:
    """One marking segment: what it read, and where on the photo it came from."""

    line: int
    part: int
    box: Box
    text: str
    confidence: float
    raw_text: str = ""
    dropped: int = 0
    ignored: bool = False   # a logo or a mark: decoded to the model's guesswork

    @property
    def label(self) -> str:
        return f"{self.line}.{self.part}"


@dataclass
class Result:
    """Everything the UI shows about one photo."""

    path: Path
    verdict: str = "PASS"
    reasons: list[str] = field(default_factory=list)
    score: float | None = None
    anomaly_label: int | None = None
    chip: Box | None = None
    segments: list[Segment] = field(default_factory=list)
    # PatchCore's per-pixel view of the same decision: the score map as uint8
    # (0..255 for anomalib's 0..1 normalised scale, threshold at 128) and the
    # region it actually called defective. Both are at model resolution.
    heat: "np.ndarray | None" = None
    mask: "np.ndarray | None" = None
    # The part itself, as fractions of the frame, when the background was cut
    # out of the scoring; None means the whole frame counted.
    part: "tuple[float, float, float, float] | None" = None
    part_score: float | None = None
    part_decides: bool = False       # False until the part-only cutoff is fitted
    cutoff: float | None = None      # the line this result was judged against
    # Ground truth, when the photo came from a labelled folder: 1 defective,
    # 0 good, None unknown. `truth_mask` is the hand-drawn defect region, at the
    # anomaly map's resolution, when one exists under ground_truth/.
    truth: int | None = None
    truth_mask: "np.ndarray | None" = None
    # What kind of defect the damaged area is, read off its shape (see
    # `icinspect.anomaly.shape`). Filled in by the inspection; None until then.
    shape: "Shape | None" = None
    # Wall time this photo cost with the models already loaded: its share of
    # the batched anomaly pass plus its own marking read and shape analysis.
    seconds: float | None = None

    @property
    def deciding_score(self) -> float | None:
        """The number the verdict came from: the part's, or the whole frame's."""
        if self.part_decides and self.part_score is not None:
            return self.part_score
        return self.score

    @property
    def peak_score(self) -> float | None:
        """The hottest pixel, on the same scale as `score`, inside the scored region."""
        from icinspect.gui.heatmap import peak

        return None if self.heat is None else peak(self.heat, self.part)

    @property
    def scored_heat(self) -> "np.ndarray | None":
        """The map as it was counted -- background zeroed when it was excluded."""
        from icinspect.gui.heatmap import restrict

        return None if self.heat is None else restrict(self.heat, self.part)

    @property
    def scored_mask(self) -> "np.ndarray | None":
        """The flagged region, background dropped the same way."""
        from icinspect.gui.heatmap import limit_mask

        if self.mask is None:
            return None
        if self.part is None:
            return self.mask
        return self.mask & limit_mask(self.mask.shape, self.part)

    @property
    def damage(self) -> "np.ndarray | None":
        """The damaged area itself: everything at or above the deciding cutoff."""
        from icinspect.gui.heatmap import damage_mask

        if self.heat is None or self.cutoff is None:
            return None
        return damage_mask(self.heat, self.part, self.cutoff)

    @property
    def damage_ratio(self) -> float | None:
        """How much of the part is damaged, 0..1 -- the share of the part, not the frame."""
        from icinspect.gui.heatmap import damage_ratio

        if self.heat is None or self.cutoff is None:
            return None
        return damage_ratio(self.heat, self.part, self.cutoff)

    @property
    def iou(self) -> float | None:
        """Overlap between the damage found and the damage marked, 0..1.

        Intersection over union, the standard measure of *where* rather than
        *whether*: a part can be failed correctly and still have the damage in
        the wrong place, and only this notices.
        """
        found = self.damage
        if found is None or self.truth_mask is None:
            return None
        marked = np.asarray(self.truth_mask, dtype=bool)
        if marked.shape != found.shape:
            return None
        union = int((found | marked).sum())
        if not union:
            return 1.0          # nothing marked, nothing found: perfect agreement
        return round(int((found & marked).sum()) / union, 4)

    @property
    def predicted(self) -> int:
        """The verdict as a label: 1 when this photo was failed as defective."""
        return int(self.verdict == "FAIL")

    @property
    def defect_area(self) -> float | None:
        """Share of the frame anomalib's own mask called defective (0..1)."""
        flagged = self.scored_mask
        return None if flagged is None else round(float(flagged.mean()), 4)

    @property
    def reads(self) -> list[Segment]:
        return [s for s in self.segments if not s.ignored and s.text]

    @property
    def marking(self) -> str:
        return " ".join(s.text for s in self.reads)

    @property
    def confidence(self) -> float:
        kept = self.reads
        return round(sum(s.confidence for s in kept) / len(kept), 4) if kept else 0.0

    def as_row(self, expected: str = "") -> dict[str, Any]:
        return {
            "image": str(self.path),
            "verdict": self.verdict,
            "reasons": ";".join(self.reasons),
            "anomaly_score": "" if self.score is None else round(self.score, 6),
            "anomaly_label": "" if self.anomaly_label is None else self.anomaly_label,
            "marking": self.marking,
            "marking_lines": " | ".join(s.text for s in self.reads),
            "ocr_confidence": self.confidence,
            "expected": expected,
            "chip": "{},{},{},{}".format(*self.chip) if self.chip else "",
            "peak_score": "" if self.peak_score is None else self.peak_score,
            "damage_ratio": "" if self.damage_ratio is None else self.damage_ratio,
            "defect_area": "" if self.defect_area is None else self.defect_area,
            "part_score": "" if self.part_score is None else self.part_score,
            "truth": "" if self.truth is None else ("defect" if self.truth else "good"),
            "iou": "" if self.iou is None else self.iou,
            "decided_on": "part" if (self.part_decides and self.part_score is not None) else "frame",
            "part": "" if self.part is None else "{:.4f},{:.4f},{:.4f},{:.4f}".format(*self.part),
            "defect_kind": "" if self.shape is None else self.shape.kind,
            "defect_detail": "" if self.shape is None else self.shape.describe(),
            "defect_blobs": "" if self.shape is None else self.shape.count,
        }

    @property
    def kind(self) -> str:
        """The defect kind's label for the window, blank when not analysed."""
        return "" if self.shape is None else self.shape.label


@dataclass
class Options:
    """The Inspect tab's settings, in one place."""

    check_anomaly: bool = True
    check_ocr: bool = True
    split: str = "auto"
    expect: str = ""
    max_cer: float = 0.0
    min_confidence: float = 0.0
    max_lines: int = 0
    min_char_confidence: float | None = None   # None -> whatever ocr.yaml says
    threshold: float | None = None             # None -> the config's, else anomalib's own
    detect_chip: bool = True
    save_crops: bool = False
    ignore_background: bool = True   # score the part only, not the fixture around it
    part_region_mode: str = REGION_REFERENCE
    max_damage: float | None = None  # fail when more than this share of the part is damaged


class Inspector:
    """Loads the two models on demand and keeps them for the next run."""

    def __init__(
        self,
        ocr_config: str = "ocr.yaml",
        patchcore_config: str = "patchcore.yaml",
        paths_config: str = "paths.yaml",
    ) -> None:
        self.ocr_config = ocr_config
        self.patchcore_config = patchcore_config
        self.paths_config = paths_config
        self._ocr: tuple[Path, Any, Any, dict, Any] | None = None
        self._anomaly: tuple[Path, Any, Any] | None = None
        # A run can be pointed at a specific checkpoint -- an older PatchCore
        # version, or crnn_last.pt against crnn_best.pt -- instead of whatever
        # is newest. None means "the default", which is what the CLI uses.
        self.ocr_override: Path | None = None
        self.anomaly_override: Path | None = None
        self._reference: tuple[float, float, float, float] | None = None
        # The box drawn by hand, once read from disk; False until then, since
        # "read and found nothing" is a different state from "not read yet".
        self._manual: Region | None | bool = False
        # Cutoff for the part-only score, once calibrated in this session.
        # Excluding the background changes what is measured, so the threshold
        # anomalib fitted over whole frames no longer applies to it.
        self.part_threshold: float | None = None

    # -- where things live -------------------------------------------------

    def paths(self):
        return resolve_paths(self.paths_config)

    def ocr_dir(self) -> Path:
        cfg = load_yaml(self.ocr_config)
        return resolve_output(cfg.get("train", {}).get("output_dir", "results/ocr"))

    def patchcore_dir(self) -> Path:
        cfg = load_yaml(self.patchcore_config)
        return resolve_output(cfg.get("trainer", {}).get("default_root_dir", "results/patchcore"))

    def ocr_checkpoint(self) -> Path | None:
        """The reader in use: the chosen one, else `crnn_best.pt`."""
        if self.ocr_override is not None and self.ocr_override.exists():
            return self.ocr_override
        ckpt = self.ocr_dir() / "crnn_best.pt"
        return ckpt if ckpt.exists() else None

    def anomaly_checkpoint(self) -> Path | None:
        """The detector in use: the chosen one, else the newest trained."""
        if self.anomaly_override is not None and self.anomaly_override.exists():
            return self.anomaly_override
        found = self.anomaly_checkpoints()
        return found[0] if found else None

    def ocr_checkpoints(self) -> list[Path]:
        """Every reader on disk, newest first. `crnn_best.pt` leads if it exists."""
        root = self.ocr_dir()
        if not root.exists():
            return []
        found = sorted(root.glob("*.pt"), key=lambda p: p.stat().st_mtime, reverse=True)
        best = root / "crnn_best.pt"
        if best in found:
            found.remove(best)
            found.insert(0, best)
        return found

    def anomaly_checkpoints(self) -> list[Path]:
        """Every detector on disk, newest first.

        anomalib writes a new `vN` folder per training run and keeps the old
        ones, so this is the list of every version still around -- which is the
        point of being able to pick one: comparing a retrain against the run it
        replaced needs no file juggling.
        """
        root = self.patchcore_dir()
        if not root.exists():
            return []
        found = sorted(root.rglob("*.ckpt"), key=lambda p: p.stat().st_mtime, reverse=True)
        # anomalib keeps a `latest` junction beside the vN folders, so the newest
        # run turns up twice under the same mtime. Keep the real one: it is the
        # path that names its version, and the junction moves on the next train.
        unique: dict[Path, Path] = {}
        for path in sorted(found, key=lambda p: "latest" in p.parts):
            unique.setdefault(path.resolve(), path)
        return sorted(unique.values(), key=lambda p: p.stat().st_mtime, reverse=True)

    def select_ocr(self, ckpt: Path | None) -> None:
        self.ocr_override = ckpt

    def select_anomaly(self, ckpt: Path | None) -> None:
        self.anomaly_override = ckpt

    def unload(self) -> None:
        """Drop both models -- after retraining, so the next run picks up new weights."""
        self._ocr = None
        self._anomaly = None
        self._reference = None      # re-read from the cache, not re-measured
        self._manual = False

    # -- OCR ---------------------------------------------------------------

    def _load_ocr(self, prog: Progress):
        import torch

        from icinspect.ocr.infer import load_checkpoint

        ckpt = self.ocr_checkpoint()
        if ckpt is None:
            raise FileNotFoundError(
                f"No OCR checkpoint at {self.ocr_dir() / 'crnn_best.pt'}.\n"
                "Train one on the Train tab first -- synthetic data is enough to start."
            )
        if self._ocr and self._ocr[0] == ckpt:
            return self._ocr[1:]
        prog.status("Loading the OCR model...")
        prog.log(f"[ocr] checkpoint: {ckpt}")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model, charset, ckpt_cfg = load_checkpoint(ckpt, device)
        prog.log(f"[ocr] device: {device}")
        self._ocr = (ckpt, model, charset, ckpt_cfg or {}, device)
        return self._ocr[1:]

    def _ocr_run_config(self, ckpt_cfg: dict, opts: Options) -> dict:
        """Checkpoint geometry + live decode/segment knobs + this run's overrides.

        The checkpoint pins the input size, but the segmentation and decoding
        knobs come from the config file, so they can be retuned in the window
        without retraining anything.
        """
        from icinspect.ocr.segment import segment_config

        cfg = load_yaml(self.ocr_config)
        run_cfg = dict(ckpt_cfg or cfg)
        run_cfg["segment"] = segment_config(cfg)
        run_cfg["decode"] = dict(cfg.get("decode") or {})
        if opts.max_lines:
            run_cfg["segment"]["max_lines"] = opts.max_lines
        if opts.min_char_confidence is not None:
            run_cfg["decode"]["min_char_confidence"] = opts.min_char_confidence
        return run_cfg

    def _ocr_reader(self, opts: Options, prog: Progress):
        """Return read(image) -> (chip box, segments), with the model loaded once."""
        from PIL import Image

        from icinspect.ocr.infer import is_unreadable, max_unsure_frac, predict_paths, predict_split

        model, charset, ckpt_cfg, device = self._load_ocr(prog)
        run_cfg = self._ocr_run_config(ckpt_cfg, opts)
        max_unsure = max_unsure_frac(run_cfg)
        crops_dir = self.ocr_dir() / "crops" if opts.save_crops else None
        # In drawn-box mode the box is the package for the reader as well: the
        # fixture puts every part in the same place, and a hand-drawn box does
        # not wander off a part the detector cannot make sense of.
        pinned = self.manual_region() if opts.part_region_mode == REGION_MANUAL else None

        def read(image: Path) -> tuple[Box | None, list[Segment]]:
            if opts.split == "none":
                # Already a single-line crop: no package to find, no lines to cut.
                found = predict_paths(model, charset, run_cfg, [image], device)[0]
                with Image.open(image) as im:
                    whole = (0, 0, im.size[0], im.size[1])
                return None, [Segment(0, 0, whole, found["text"], found["confidence"],
                                      found.get("raw_text", ""), found.get("dropped", 0))]

            rows, _ = predict_split(
                model, charset, run_cfg, [image], device, opts.split, crops_dir,
                detect_chip=opts.detect_chip,
                drop_unreadable=False,   # flagged instead, so the UI can show what was skipped
                chip_box=_region_pixels(pinned, image),
            )
            chip: Box | None = None
            segments: list[Segment] = []
            for row in rows:
                if row["chip"]:
                    chip = _box(row["chip"])
                segments.append(
                    Segment(
                        line=row["line"],
                        part=row["part"],
                        box=_box(row["box"]),
                        text=row["text"],
                        confidence=row["confidence"],
                        raw_text=row.get("raw_text", ""),
                        dropped=row.get("dropped", 0),
                        ignored=is_unreadable(row, max_unsure),
                    )
                )
            return chip, segments

        return read

    def read_markings(
        self, images: list[Path], opts: Options, prog: Progress
    ) -> dict[str, tuple[Box | None, list[Segment]]]:
        """Read every photo. Returns {resolved path: (chip box, segments)}."""
        read = self._ocr_reader(opts, prog)
        out: dict[str, tuple[Box | None, list[Segment]]] = {}
        for index, image in enumerate(images):
            prog.check()
            prog.status(f"Reading marking {index + 1}/{len(images)}: {image.name}")
            out[str(image.resolve())] = read(image)
            prog.step(index + 1, len(images))
        return out

    # -- anomaly -----------------------------------------------------------

    def _load_anomaly(self, prog: Progress):
        from icinspect.anomaly.common import build_engine, load_patchcore

        ckpt = self.anomaly_checkpoint()
        if ckpt is None:
            raise FileNotFoundError(
                f"No PatchCore checkpoint under {self.patchcore_dir()}.\n"
                "Train one on the Train tab, or untick the anomaly check."
            )
        if self._anomaly and self._anomaly[0] == ckpt:
            return self._anomaly[1:]
        prog.status("Loading PatchCore...")
        prog.log(f"[anomaly] checkpoint: {ckpt}")
        cfg = load_yaml(self.patchcore_config)
        model = load_patchcore(ckpt, cfg)
        engine = build_engine(cfg, self.patchcore_dir())
        self._anomaly = (ckpt, model, engine)
        return self._anomaly[1:]

    def score_anomaly(self, images: list[Path], opts: Options, prog: Progress) -> dict[str, dict]:
        import torch

        from icinspect.anomaly.infer_patchcore import _rows_from_batch

        model, engine = self._load_anomaly(prog)
        cfg = load_yaml(self.patchcore_config)

        # anomalib scores a directory, not a file list. Handing it the parent
        # folder would score all 56 captures to answer for the one photo that
        # was picked, so a partial selection is hardlinked into a scratch folder
        # first -- seconds instead of half a minute for a single part.
        scored: dict[str, dict] = {}
        # A truncated capture would stop anomalib's whole pass; it is left out
        # here and comes back as "anomaly-missing" with the reason in the log.
        images = _readable(images, prog)
        if not images:
            return scored
        started = time.perf_counter()
        with torch.no_grad(), _staging(images) as (folders, back):
            for i, folder in enumerate(folders):
                prog.check()
                prog.status(f"Scoring {i + 1}/{len(folders)}: {folder.name}")
                for batch in engine.predict(model=model, data_path=str(folder)) or []:
                    maps = _maps_from_batch(batch)
                    for position, row in enumerate(_rows_from_batch(batch)):
                        key = back.get(str(Path(row["image"]).resolve()))
                        if key is not None:
                            row["image"] = key
                            row.update(maps[position] if position < len(maps) else {})
                            scored[key] = row
        # The pass is batched, so each photo is charged an equal share of it.
        share = (time.perf_counter() - started) / max(len(scored), 1)
        for row in scored.values():
            row["seconds"] = share

        threshold = opts.threshold
        if threshold is None:
            threshold = cfg.get("threshold", {}).get("image_score")
        if threshold is not None:
            for row in scored.values():
                if row.get("score") is not None:
                    row["pred_label"] = int(row["score"] >= float(threshold))
        return scored

    def reference_region(
        self, prog: Progress | None = None, samples: int = 60, refresh: bool = False
    ) -> tuple[float, float, float, float] | None:
        """The box a normal part fills, as fractions of the frame.

        Measured once over `train/good` -- the parts the detector was built
        from -- and cached beside the checkpoint. Taking the median rather than
        the union keeps one oddly-seated part from widening the window for
        every photo; the fixture holds them to a few percent of the frame, which
        is what makes a fixed region workable at all.
        """
        if self._reference is not None and not refresh:
            return self._reference

        cache = self.patchcore_dir() / REFERENCE_FILE
        if cache.exists() and not refresh:
            try:
                saved = json.loads(cache.read_text(encoding="utf-8"))
                region = tuple(float(v) for v in saved["region"])
                if len(region) == 4:
                    self._reference = region            # type: ignore[assignment]
                    return self._reference
            except (OSError, ValueError, KeyError, TypeError):
                pass                                    # recompute below

        from PIL import Image

        from icinspect.ocr.segment import find_chip

        good = list_images(self.paths().anomaly_root / "train" / "good")[:samples]
        if not good:
            return None
        cfg = load_yaml(self.ocr_config)
        boxes: list[tuple[float, float, float, float]] = []
        for path in good:
            try:
                with Image.open(path) as img:
                    img.load()
                    found = find_chip(img, cfg)
                    width, height = img.size
            except Exception:
                continue                                 # a truncated capture
            if found:
                boxes.append((found[0] / width, found[1] / height,
                              found[2] / width, found[3] / height))
        if not boxes:
            return None

        middle = len(boxes) // 2
        region = tuple(sorted(values)[middle] for values in zip(*boxes))
        self._reference = region                         # type: ignore[assignment]
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps({
            "region": list(region),
            "samples": len(boxes),
            "source": "train/good",
            "computed": datetime.now().isoformat(timespec="seconds"),
        }, indent=2), encoding="utf-8")
        if prog is not None:
            prog.log(f"[part] scored region from {len(boxes)} good part(s): "
                     + "{:.4f}, {:.4f}, {:.4f}, {:.4f}".format(*region))
        return self._reference

    def clear_reference(self) -> None:
        """Drop the in-memory region; the cache on disk still stands."""
        self._reference = None

    def forget_region(self) -> None:
        """Throw the learned region away for good -- the fixture or dataset moved.

        Measuring it again costs a second or two over the good parts, so the
        cache is only deleted when something that invalidates it happens, not
        on every model switch.
        """
        self._reference = None
        cache = self.patchcore_dir() / REFERENCE_FILE
        try:
            cache.unlink(missing_ok=True)
        except OSError:
            pass

    # -- the box drawn by hand ----------------------------------------------

    def manual_region(self) -> Region | None:
        """The box drawn on a photo in the window, as fractions of the frame.

        Kept beside the learned one, in its own file, so relearning the region
        after the fixture moves does not also throw away what was drawn.
        """
        if self._manual is not False:
            return self._manual                      # type: ignore[return-value]
        self._manual = None
        cache = self.patchcore_dir() / MANUAL_FILE
        if cache.exists():
            try:
                saved = json.loads(cache.read_text(encoding="utf-8"))
                region = tuple(float(v) for v in saved["region"])
                if len(region) == 4 and region[2] > region[0] and region[3] > region[1]:
                    self._manual = region            # type: ignore[assignment]
            except (OSError, ValueError, KeyError, TypeError):
                pass
        return self._manual                          # type: ignore[return-value]

    def set_manual_region(self, region: Region, drawn_on: Path | None = None) -> Region:
        """Keep a box drawn by hand, clamped to the frame, and remember it on disk."""
        x1, y1, x2, y2 = (min(1.0, max(0.0, float(v))) for v in region)
        if x2 <= x1 or y2 <= y1:
            raise ValueError("The box has no area.")
        self._manual = (x1, y1, x2, y2)
        cache = self.patchcore_dir() / MANUAL_FILE
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps({
            "region": list(self._manual),
            "source": "drawn by hand",
            "drawn_on": None if drawn_on is None else str(drawn_on),
            "computed": datetime.now().isoformat(timespec="seconds"),
        }, indent=2), encoding="utf-8")
        return self._manual

    def clear_manual_region(self) -> None:
        """Throw the drawn box away, in memory and on disk."""
        self._manual = None
        try:
            (self.patchcore_dir() / MANUAL_FILE).unlink(missing_ok=True)
        except OSError:
            pass

    @staticmethod
    def _part_from_chip(chip: Box | None, image: Path) -> tuple[float, float, float, float] | None:
        """The marking reader's package box, expressed as fractions of the frame."""
        if chip is None:
            return None
        from PIL import Image

        try:
            with Image.open(image) as img:
                width, height = img.size
        except Exception:
            return None
        x1, y1, x2, y2 = chip
        return (x1 / width, y1 / height, x2 / width, y2 / height)

    def part_cutoff(self, opts: "Options | None" = None) -> float:
        """The line a part-only score has to cross to fail.

        In order: whatever the run pins, then this session's calibration, then
        `threshold.part_score` in the config, then anomalib's pixel threshold --
        which is the right scale but the wrong calibration, because it was fitted
        with the background still in the picture.
        """
        value, _source = self.part_cutoff_source(opts)
        return value

    def part_cutoff_source(self, opts: "Options | None" = None) -> tuple[float, str]:
        """The cutoff and where it came from: run / session / config / default."""
        if opts is not None and opts.threshold is not None:
            return float(opts.threshold), "run"
        if self.part_threshold is not None:
            return float(self.part_threshold), "session"
        pinned = (load_yaml(self.patchcore_config).get("threshold") or {}).get("part_score")
        if pinned is not None:
            return float(pinned), "config"
        return PIXEL_THRESHOLD, "default"

    def part_cutoff_fitted(self, opts: "Options | None" = None) -> bool:
        """True when the part-only cutoff is a fitted number rather than a stand-in.

        anomalib's pixel line was calibrated with the background in the picture,
        so it is the right scale but the wrong number for a part-only score --
        and using it unfitted would quietly pass defects.
        """
        return self.part_cutoff_source(opts)[1] != "default"

    def calibrate_part_threshold(self, prog: Progress, opts: "Options | None" = None) -> dict:
        """Fit the part-only cutoff on the labelled test set.

        Scores `test/good` and `test/defect` with the background excluded, then
        picks the cutoff with the best F1 -- the same quantity anomalib's own
        adaptive threshold maximises, refitted for what is now being measured.
        """
        paths = self.paths()
        good = list_images(paths.anomaly_root / "test" / "good")
        bad = list_images(paths.anomaly_root / "test" / "defect")
        if not good or not bad:
            raise FileNotFoundError(
                "Calibration needs both test/good and test/defect under\n"
                f"{paths.anomaly_root}\nwith images in them."
            )

        settings = Options(check_anomaly=True, check_ocr=False, ignore_background=True)
        if opts is not None:
            settings.detect_chip = opts.detect_chip
            settings.part_region_mode = opts.part_region_mode
        prog.log(f"[calibrate] scoring {len(good)} good and {len(bad)} defective part(s)")
        results = self.inspect(good + bad, settings, prog)
        labels = [0] * len(good) + [1] * len(bad)

        scored = [(r.deciding_score, y) for r, y in zip(results, labels)
                  if r.deciding_score is not None]
        if not scored:
            raise ValueError("Nothing scored: is the detector trained?")

        best = _best_f1(scored)
        area = auroc(scored)
        best["auroc"] = float("nan") if area is None else round(area, 4)
        best["good"] = len(good)
        best["defect"] = len(bad)
        best["found_part"] = sum(1 for r in results if r.part is not None)
        best["scores"] = scored
        self.part_threshold = best["threshold"]
        prog.log(f"[calibrate] cutoff {best['threshold']:.4f}  F1 {best['f1']:.4f}  "
                 f"AUROC {best['auroc']:.4f}")
        return best

    def part_region(self, image: Path, cfg: dict | None = None) -> tuple[float, float, float, float] | None:
        """The package as fractions of the frame, or None when it cannot be found.

        This is the same detector the marking reader uses -- the fixture keeps
        its backdrop clear, so everything that differs from the frame border is
        the part, and the moulded body is what the rows and columns fill. It
        runs on a downscaled copy, so it costs milliseconds per photo.
        """
        from PIL import Image

        from icinspect.ocr.segment import find_chip

        try:
            with Image.open(image) as img:
                img.load()
                box = find_chip(img, cfg if cfg is not None else load_yaml(self.ocr_config))
                if box is None:
                    return None
                width, height = img.size
        except Exception:            # a truncated capture: fall back to the whole frame
            return None
        x1, y1, x2, y2 = box
        return (x1 / width, y1 / height, x2 / width, y2 / height)

    def shape_of(self, result: Result) -> "Shape | None":
        """Name the kind of defect on one result from the shape of its damage.

        The photo is decoded at a quarter of its size -- enough for the package
        finder, which works at 512 px wide anyway, and for the backdrop check,
        which works at the map's resolution -- so this costs tens of
        milliseconds per photo rather than the half second a 4K decode takes.
        """
        from PIL import Image

        from icinspect.anomaly.shape import analyse, body_fraction

        damage = result.damage
        if damage is None:
            return None
        photo = None
        body = None
        try:
            with Image.open(result.path) as img:
                width, height = img.size
                img.draft("L", (max(width // 4, 512), max(height // 4, 288)))
                photo = img.convert("L")
        except Exception:                 # a truncated capture: shape only
            photo = None
        if photo is not None:
            body = body_fraction(photo, result.part, load_yaml(self.ocr_config))
        return analyse(damage, result.heat, result.part, width / height if photo else None,
                       photo=photo, body_frac=body)

    def report_config(self) -> dict:
        """The `report:` block of patchcore.yaml (mm scale, repeated-shot grouping)."""
        block = load_yaml(self.patchcore_config).get("report") or {}
        return dict(block) if isinstance(block, dict) else {}

    def chip_inset(self) -> float:
        """How far the package finder pulls its box in off the moulding edge."""
        from icinspect.ocr.segment import segment_config

        try:
            return float(segment_config(load_yaml(self.ocr_config)).get("chip_inset", 0.0))
        except Exception:
            return 0.0

    def truth_of(self, image: Path) -> int | None:
        """1 for a defective part, 0 for a good one, None when the folder says nothing.

        The dataset layout is the label: anomalib is told which folders hold
        which, so the same names are trusted here rather than asking for the
        truth twice.
        """
        cfg = load_yaml(self.patchcore_config).get("dataset", {})
        root = self.paths().anomaly_root
        try:
            relative = image.resolve().relative_to(root.resolve()).as_posix()
        except (ValueError, OSError):
            return None
        abnormal = str(cfg.get("abnormal_dir", "test/defect")).strip("/")
        normals = [str(cfg.get(key, default)).strip("/") for key, default in
                   (("normal_dir", "train/good"), ("normal_test_dir", "test/good"))]
        if relative.startswith(abnormal + "/"):
            return 1
        if any(relative.startswith(name + "/") for name in normals):
            return 0
        return None

    def truth_mask_for(self, image: Path, shape: tuple[int, int]) -> "np.ndarray | None":
        """The hand-drawn defect mask for this photo, resampled to the map's grid.

        anomalib's convention: same file name under `mask_dir`, white where the
        defect is. `<stem>_mask.png` is accepted too, which is how several
        labelling tools write them.
        """
        from PIL import Image

        cfg = load_yaml(self.patchcore_config).get("dataset", {})
        folder = self.paths().anomaly_root / str(cfg.get("mask_dir", "ground_truth/defect"))
        if not folder.exists():
            return None
        candidates = [folder / image.name]
        candidates += [folder / (image.stem + suffix) for suffix in MASK_SUFFIXES]
        candidates += [folder / (image.stem + "_mask" + suffix) for suffix in MASK_SUFFIXES]
        found = next((p for p in candidates if p.exists()), None)
        if found is None:
            return None
        try:
            with Image.open(found) as mask:
                mask.load()
                # Nearest, not smooth: a mask is a decision per pixel, and a
                # blurred edge would invent half-marked pixels.
                small = mask.convert("L").resize((shape[1], shape[0]), Image.NEAREST)
        except Exception:
            return None
        return np.asarray(small) > 127

    # -- both together -----------------------------------------------------

    def inspect(self, images: list[Path], opts: Options, prog: Progress) -> list[Result]:
        """Run the enabled checks over `images` and decide PASS/FAIL for each."""
        if not opts.check_anomaly and not opts.check_ocr:
            raise ValueError("Nothing to do: tick at least one of the two checks.")

        # Anomaly scoring is one batched pass over the folder, so it happens
        # first and in full; the marking is then read photo by photo, and each
        # verdict is handed to the UI as it lands rather than at the end.
        anomaly: dict[str, dict] = {}
        if opts.check_anomaly:
            anomaly = self.score_anomaly(images, opts, prog)

        read = self._ocr_reader(opts, prog) if opts.check_ocr else None
        ocr_cfg = load_yaml(self.ocr_config) if opts.ignore_background else None
        reference = None
        if opts.ignore_background and opts.part_region_mode == REGION_MANUAL:
            reference = self.manual_region()
            if reference is None and prog is not None:
                prog.log("[part] no box has been drawn yet; using the region "
                         "learned from the good parts instead")
        if opts.ignore_background and reference is None and opts.part_region_mode != REGION_PHOTO:
            reference = self.reference_region(prog)
            if reference is None and prog is not None:
                prog.log("[part] no good parts to learn the region from; "
                         "measuring each photo instead")
        part_cutoff = self.part_cutoff(opts)
        part_decides = self.part_cutoff_fitted(opts)

        expected = opts.expect.strip().upper()
        results: list[Result] = []
        for index, image in enumerate(images):
            prog.check()
            key = str(image.resolve())
            started = time.perf_counter()
            if read is None:
                chip, segments = None, []
            else:
                prog.status(f"Reading marking {index + 1}/{len(images)}: {image.name}")
                try:
                    chip, segments = read(image)
                except OSError as exc:          # a truncated capture: no marking, not a crash
                    prog.log(f"[ocr] cannot read {image.name}: {exc}")
                    chip, segments = None, []
            row = anomaly.get(key, {})
            result = Result(
                path=image,
                score=row.get("score"),
                anomaly_label=row.get("pred_label"),
                chip=chip,
                segments=segments,
                heat=row.get("heat"),
                mask=row.get("mask"),
            )
            if opts.ignore_background and result.heat is not None:
                # Reuse the box the reader already found; only look again when
                # the marking check is off or the search there gave up.
                # The fixed region first: a broken part measured on its own
                # would hand back a box with the damage cut out of it.
                part = reference or self._part_from_chip(chip, image) or \
                    self.part_region(image, ocr_cfg)
                if part is not None:
                    result.part = part
                    result.part_score = peak(result.heat, part)
                    result.part_decides = part_decides
            if result.heat is not None:
                result.cutoff = part_cutoff
                result.truth = self.truth_of(image)
                result.truth_mask = self.truth_mask_for(image, result.heat.shape)
                result.shape = self.shape_of(result)
            if row or read is not None:
                result.seconds = (time.perf_counter() - started) + float(row.get("seconds", 0.0))
            reasons: list[str] = []
            if opts.check_anomaly:
                if not row:
                    reasons.append("anomaly-missing")
                elif result.part_decides and result.part_score is not None:
                    if result.part_score >= part_cutoff:
                        reasons.append("anomaly")
                elif row.get("pred_label") == 1:
                    reasons.append("anomaly")
                if opts.max_damage is not None and (result.damage_ratio or 0.0) > opts.max_damage:
                    reasons.append("damage-over-limit")
            if opts.check_ocr:
                if expected:
                    if cer(result.marking, expected) > opts.max_cer:
                        reasons.append("marking-mismatch")
                elif not result.marking:
                    reasons.append("marking-unreadable")
                if opts.min_confidence and result.confidence < opts.min_confidence:
                    reasons.append("low-confidence")
            result.reasons = reasons
            result.verdict = "FAIL" if reasons else "PASS"
            results.append(result)
            prog.partial(result)
            prog.step(index + 1, len(images))
        return results


def _best_f1(scored: list[tuple[float, int]]) -> dict:
    """The cutoff with the best F1, defect being the positive class."""
    values = sorted({score for score, _ in scored})
    # Candidates sit between observed scores, so the cutoff never lands exactly
    # on a sample and flip with a rounding error.
    candidates = [values[0] - 1e-6] + [
        (a + b) / 2 for a, b in zip(values, values[1:])
    ] + [values[-1] + 1e-6]
    best = {"threshold": candidates[0], "f1": -1.0, "precision": 0.0, "recall": 0.0}
    positives = sum(y for _, y in scored)
    for cutoff in candidates:
        tp = sum(1 for score, y in scored if y == 1 and score >= cutoff)
        fp = sum(1 for score, y in scored if y == 0 and score >= cutoff)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / positives if positives else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        if f1 > best["f1"]:
            best = {"threshold": round(cutoff, 4), "f1": round(f1, 4),
                    "precision": round(precision, 4), "recall": round(recall, 4)}
    return best


def write_report(results: list[Result], path: Path, expected: str = "") -> Path:
    """Write the same CSV shape as `icinspect.pipeline`, so anything downstream
    cannot tell whether the run came from the window or the command line."""
    if not results:
        raise ValueError("Nothing to export: run an inspection first.")
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [r.as_row(expected) for r in results]
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return path
