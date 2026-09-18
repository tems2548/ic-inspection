"""The reporting table: the numbers a qualification report asks for.

A confusion matrix says how a run went; a qualification report asks different
questions -- how often a defect escapes, how often a good part is scrapped,
the smallest defect the system can be trusted to catch, how repeatable the
score is, and how long a part takes. Each has a conventional form (a rate with
a confidence interval and its n; a size in millimetres; a sigma from repeated
measurements), and this module computes every one from what an inspection
already carries, with no model loaded:

    Image AUROC          rank-based, from the run's deciding scores vs truth
    PRO @ tau_hard       per-region overlap of the damage found with each
                         marked region, so a small defect counts as much as
                         a large one
    Escape rate          missed defects / defective parts, Wilson 95% CI
    Overkill rate        false alarms / good parts, Wilson 95% CI
    Detection limit      the smallest marked defect caught, in mm, above the
                         largest one missed
    Repeatability sigma  the pooled within-part standard deviation of the
                         score over repeated shots of the same part
    Inference latency    milliseconds per part on the hardware it ran on

Where a number cannot be computed, the row says what is missing rather than
showing a dash: masks, repeated shots, a labelled folder.
"""

from __future__ import annotations

import csv
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np

Z95 = 1.959964                      # two-sided 95% normal quantile


@dataclass(frozen=True)
class Rate:
    """A proportion with its Wilson interval and the count it came from."""

    successes: int
    n: int

    @property
    def value(self) -> float:
        return self.successes / self.n

    @property
    def interval(self) -> tuple[float, float]:
        return wilson(self.successes, self.n)

    def text(self) -> str:
        low, high = self.interval
        return f"{self.value * 100:.2f}%  (95% CI {low * 100:.2f}-{high * 100:.2f}%, n={self.n})"


@dataclass
class Row:
    """One line of the table: the metric, its value as text, and where it came from."""

    metric: str
    value: str
    basis: str
    available: bool = True


@dataclass
class ReportTable:
    rows: list[Row] = field(default_factory=list)

    def as_markdown(self) -> str:
        lines = ["| Metric | Value | Basis |", "|---|---|---|"]
        for row in self.rows:
            lines.append(f"| {row.metric} | {row.value} | {row.basis} |")
        return "\n".join(lines)


# -- statistics ------------------------------------------------------------


def wilson(successes: int, n: int, z: float = Z95) -> tuple[float, float]:
    """Wilson score interval for a proportion.

    Preferred over the normal approximation because the rates here sit at
    zero -- no escapes in 220 defective parts -- where Wald says exactly
    [0, 0] and Wilson says the honest [0, 1.7%].
    """
    if n <= 0:
        return (0.0, 1.0)
    p = successes / n
    denominator = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denominator
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denominator
    return (max(0.0, centre - half), min(1.0, centre + half))


def auroc(scored: Sequence[tuple[float, int]]) -> float | None:
    """Rank-based AUROC with ties shared; None unless both classes are present."""
    ranked = sorted(scored, key=lambda pair: pair[0])
    ranks: dict[int, float] = {}
    index = 0
    while index < len(ranked):
        stop = index
        while stop + 1 < len(ranked) and ranked[stop + 1][0] == ranked[index][0]:
            stop += 1
        shared = (index + stop) / 2 + 1
        for position in range(index, stop + 1):
            ranks[position] = shared
        index = stop + 1
    positives = [i for i, (_s, y) in enumerate(ranked) if y == 1]
    negatives = len(ranked) - len(positives)
    if not positives or not negatives:
        return None
    rank_sum = sum(ranks[i] for i in positives)
    return (rank_sum - len(positives) * (len(positives) + 1) / 2) / (len(positives) * negatives)


def region_overlaps(found: np.ndarray, marked: np.ndarray) -> list[float]:
    """The share of each marked region that the found damage covers.

    One number per connected region of the truth mask, which is what makes
    the PRO score size-independent: a pinhole and a broken corner each count
    once, where IoU would let the corner decide.
    """
    from icinspect.anomaly.shape import label_blobs

    found = np.asarray(found, dtype=bool)
    marked = np.asarray(marked, dtype=bool)
    if found.shape != marked.shape:
        return []
    labels, count = label_blobs(marked)
    overlaps = []
    for index in range(1, count + 1):
        region = labels == index
        overlaps.append(float((found & region).sum() / region.sum()))
    return overlaps


def pooled_sigma(groups: Sequence[Sequence[float]]) -> tuple[float | None, int, int]:
    """Repeatability from repeated measurements: (sigma, parts, shots).

    The pooled within-group standard deviation, as a Gage R&R study defines
    equipment variation: each part's shots vary around that part's own mean,
    and the variances are pooled weighted by their degrees of freedom. Groups
    with a single shot carry no information and are left out.
    """
    used = [list(g) for g in groups if len(g) >= 2]
    if not used:
        return None, 0, 0
    freedom = sum(len(g) - 1 for g in used)
    pooled = sum((len(g) - 1) * float(np.var(g, ddof=1)) for g in used) / freedom
    return math.sqrt(pooled), len(used), sum(len(g) for g in used)


def detection_limit(sizes: Sequence[tuple[float, bool]]) -> tuple[float | None, float | None]:
    """(limit, largest missed) from (defect size, caught) pairs.

    The limit is the smallest caught defect larger than every missed one:
    above it, nothing in the set escaped. With nothing missed it is simply the
    smallest defect caught. None when nothing was caught above the misses.
    """
    missed = [size for size, caught in sizes if not caught]
    floor = max(missed) if missed else None
    above = [size for size, caught in sizes if caught and (floor is None or size > floor)]
    return (min(above) if above else None), floor


def equivalent_diameter_mm(mask: np.ndarray, mm_per_pixel: tuple[float, float]) -> float:
    """The diameter of the circle with the mask's area, in millimetres.

    The map's pixels are not square on the photo -- a 16:9 frame is resampled
    to the model's square input -- so the area is taken with a separate scale
    per axis before it is turned into a length.
    """
    area = float(np.asarray(mask, dtype=bool).sum()) * mm_per_pixel[0] * mm_per_pixel[1]
    return math.sqrt(4 * area / math.pi)


def mm_per_pixel(
    part: tuple[float, float, float, float],
    shape: tuple[int, int],
    body_mm: tuple[float, float],
    inset: float = 0.0,
) -> tuple[float, float]:
    """Millimetres per map pixel along x and y, from the scored region.

    The region is the moulded body as the package finder returns it, pulled in
    by `inset` on every side, so it spans `1 - 2 * inset` of the body's length
    and width. `body_mm` is (along x, along y) on the photo.
    """
    h, w = shape[:2]
    x1, y1, x2, y2 = part
    span = max(1.0 - 2.0 * inset, 1e-6)
    width_px = max((x2 - x1) * w, 1e-6)
    height_px = max((y2 - y1) * h, 1e-6)
    return (body_mm[0] * span / width_px, body_mm[1] * span / height_px)


# -- grouping repeated shots -------------------------------------------------


def repeat_groups(
    names: Sequence[str],
    pattern: str | None = None,
    listing: Path | None = None,
) -> dict[str, list[int]]:
    """Which photos are shots of the same part: part id -> indexes into `names`.

    Two sources, the listing first: a `repeats.csv` with `image,part` columns
    (file names, any part id), or a regex whose first group is the part id
    (`^(.+)[_-]\\d+$` makes `A17_1.jpg` and `A17_2.jpg` one part). A name that
    matches neither is its own single-shot group, which the sigma ignores.
    """
    listed: dict[str, str] = {}
    if listing is not None and listing.exists():
        try:
            with listing.open("r", encoding="utf-8", newline="") as fh:
                for row in csv.DictReader(fh):
                    image = (row.get("image") or "").strip()
                    part = (row.get("part") or "").strip()
                    if image and part:
                        listed[Path(image).name] = part
        except (OSError, csv.Error):
            listed = {}
    regex = re.compile(pattern) if pattern else None
    groups: dict[str, list[int]] = {}
    for index, name in enumerate(names):
        base = Path(name).name
        part = listed.get(base)
        if part is None and regex is not None:
            found = regex.match(Path(base).stem)
            if found and found.groups():
                part = found.group(1)
        groups.setdefault(part if part is not None else f"\0{index}", []).append(index)
    return groups


# -- the table -----------------------------------------------------------------


def build_table(
    results,
    *,
    cutoff: float | None,
    body_mm: tuple[float, float] | None,
    inset: float,
    repeats_pattern: str | None,
    repeats_listing: Path | None,
    device: str,
    fallback_auroc: str | None = None,
) -> ReportTable:
    """Fold a run's `service.Result` list into the seven rows.

    `results` may be empty; every row then says what an inspection would add.
    """
    tau = "" if cutoff is None else f" @ {cutoff:.4f}"
    rows: list[Row] = []

    # AUROC -- for comparing detectors, not for judging one: it does not
    # depend on the cutoff at all.
    scored = [(float(r.deciding_score), int(r.truth)) for r in results
              if r.truth is not None and r.deciding_score is not None]
    value = auroc(scored)
    if value is not None:
        rows.append(Row("Image AUROC", f"{value:.4f}",
                        f"this run, {len(scored)} labelled photos; for model comparison only"))
    elif fallback_auroc:
        rows.append(Row("Image AUROC", fallback_auroc,
                        "training summary (anomalib, whole frame); inspect a labelled folder for this run's",
                        available=False))
    else:
        rows.append(Row("Image AUROC", "-", "needs labelled photos of both kinds in the run",
                        available=False))

    # PRO -- localisation, each marked region weighted the same.
    overlaps: list[float] = []
    masked = 0
    for result in results:
        found = getattr(result, "damage", None)
        marked = getattr(result, "truth_mask", None)
        if found is None or marked is None:
            continue
        masked += 1
        overlaps.extend(region_overlaps(found, marked))
    if overlaps:
        rows.append(Row(f"PRO score{tau}", f"{float(np.mean(overlaps)):.4f}",
                        f"per-region overlap, {len(overlaps)} marked region(s) on {masked} photo(s)"))
    else:
        rows.append(Row("PRO score", "-",
                        "needs masks under ground_truth/ for photos in the run", available=False))

    # Escape and overkill -- the two rates a line lives by.
    defective = [r for r in results if r.truth == 1]
    good = [r for r in results if r.truth == 0]
    if defective:
        rate = Rate(sum(1 for r in defective if r.verdict != "FAIL"), len(defective))
        rows.append(Row(f"Escape rate{tau}", rate.text(),
                        "missed defects / defective parts (folder labels), Wilson interval"))
    else:
        rows.append(Row("Escape rate", "-", "needs photos from test/defect in the run", available=False))
    if good:
        rate = Rate(sum(1 for r in good if r.verdict == "FAIL"), len(good))
        rows.append(Row(f"Overkill rate{tau}", rate.text(),
                        "false alarms / good parts (folder labels), Wilson interval"))
    else:
        rows.append(Row("Overkill rate", "-", "needs photos from test/good or train/good in the run",
                        available=False))

    # Detection limit -- the headline number.
    sizes: list[tuple[float, bool]] = []
    if body_mm is not None:
        for result in defective:
            marked = getattr(result, "truth_mask", None)
            part = getattr(result, "part", None)
            if marked is None or part is None or not np.any(marked):
                continue
            scale = mm_per_pixel(part, marked.shape, body_mm, inset)
            sizes.append((equivalent_diameter_mm(marked, scale), result.verdict == "FAIL"))
    if sizes:
        limit, largest_missed = detection_limit(sizes)
        caught = sum(1 for _s, c in sizes if c)
        if limit is None:
            rows.append(Row("Detection limit", "-",
                            f"nothing caught above the largest missed defect "
                            f"({largest_missed:.2f} mm); {caught}/{len(sizes)} caught", available=False))
        else:
            note = (f"{caught}/{len(sizes)} masked defects caught; smallest caught above the "
                    f"largest missed" + (f" ({largest_missed:.2f} mm)" if largest_missed else "")
                    + f"; body {body_mm[0]:g} x {body_mm[1]:g} mm")
            rows.append(Row("Detection limit", f"{limit:.2f} mm (equivalent diameter)", note))
    elif body_mm is None:
        rows.append(Row("Detection limit", "-",
                        "set report.body_mm in configs/patchcore.yaml (package length x width)",
                        available=False))
    else:
        rows.append(Row("Detection limit", "-",
                        "needs masked defective photos in the run with the background ignored",
                        available=False))

    # Repeatability -- the score's own noise.
    names = [str(r.path) for r in results]
    groups = repeat_groups(names, repeats_pattern, repeats_listing)
    series = []
    for indexes in groups.values():
        shots = [float(results[i].deciding_score) for i in indexes
                 if results[i].deciding_score is not None]
        if len(shots) >= 2:
            series.append(shots)
    sigma, parts, shots = pooled_sigma(series)
    if sigma is not None:
        note = f"Gage R&R repeatability (pooled within-part SD), {parts} part(s), {shots} shots"
        if cutoff:
            note += f"; {sigma / cutoff * 100:.1f}% of the cutoff"
        rows.append(Row("Score repeatability σ", f"{sigma:.4f}", note))
    else:
        rows.append(Row("Score repeatability σ", "-",
                        "needs repeated shots of the same part: name them <part>_1.jpg, "
                        "<part>_2.jpg or list them in anomaly/ic/repeats.csv (image,part)",
                        available=False))

    # Latency -- per part, on this machine.
    times = [float(r.seconds) for r in results if getattr(r, "seconds", None) is not None]
    if times:
        median = float(np.median(times)) * 1000
        rows.append(Row("Inference latency", f"{median:.0f} ms/unit",
                        f"median over {len(times)} photo(s), models warm; {device}"))
    else:
        rows.append(Row("Inference latency", "-", f"no timed inspection yet; {device}", available=False))
    return ReportTable(rows)
