"""The Stats tab's charts.

One committed light look, drawn on the app's own surface: thin marks, hairline
solid grid, recessive axes, and text in ink rather than in the series colour.
Series colours are slots 1 and 2 of the reference categorical palette, which
validate together on this surface (worst adjacent CVD Delta E 24.7, normal-vision
33.6, both clear of the floors); the threshold rule is a *status* colour and
never doubles as a series.

Each function draws into an axes it is handed, so the layout lives in the tab
and the charts stay testable on a headless Agg figure.
"""

from __future__ import annotations

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
SERIES = ("#2a78d6", "#eb6834")     # slot 1 blue, slot 2 orange
CRITICAL = "#d03b3b"                # status: the threshold, never a series
# One hue, light to dark, for magnitude in the confusion matrix. Steps 100 and
# 450 of the reference sequential ramp -- a rainbow here would make a count of
# three look like a different kind of thing from a count of thirty.
SEQUENTIAL = ("#cde2fb", "#256abf")

TITLE_SIZE = 9
LABEL_SIZE = 8


def style_axes(ax, title: str = "", xlabel: str = "", ylabel: str = "") -> None:
    """Recessive chrome: no box, hairline horizontal grid, muted tick text."""
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
        ax.spines[side].set_linewidth(0.8)
    ax.tick_params(colors=MUTED, labelsize=LABEL_SIZE, length=3, width=0.8)
    ax.grid(axis="y", color=GRID, linewidth=0.8, linestyle="-")   # solid, never dashed
    ax.set_axisbelow(True)
    if title:
        ax.set_title(title, color=INK, fontsize=TITLE_SIZE, loc="left", pad=6)
    if xlabel:
        ax.set_xlabel(xlabel, color=INK_2, fontsize=LABEL_SIZE)
    if ylabel:
        ax.set_ylabel(ylabel, color=INK_2, fontsize=LABEL_SIZE)


def empty(ax, message: str, title: str = "") -> None:
    """An empty chart says what would fill it, rather than showing bare axes."""
    ax.clear()
    ax.set_facecolor(SURFACE)
    ax.axis("off")
    if title:
        ax.set_title(title, color=INK, fontsize=TITLE_SIZE, loc="left", pad=6)
    ax.text(0.5, 0.5, message, ha="center", va="center", color=MUTED,
            fontsize=LABEL_SIZE, transform=ax.transAxes, wrap=True)


def score_histogram(ax, scores: list[float], threshold: float = 0.5) -> None:
    """Where this run's anomaly scores fell, against the deciding threshold.

    One hue: the bars are a single series (photos), so darkening them by height
    would encode the height twice. The threshold is the other thing on the
    chart, and it is a status colour with its own label.
    """
    ax.clear()
    if not scores:
        empty(ax, "Run an inspection with the defect check on.", "Anomaly scores")
        return
    # The threshold moves with the calibration, so the rule is labelled with
    # its own value rather than the axis claiming a fixed one.
    style_axes(ax, "Anomaly scores", "score", "photos")
    span = max(max(scores), threshold * 1.2)
    ax.hist(scores, bins=min(24, max(6, len(scores) // 2)), range=(0, span),
            color=SERIES[0], edgecolor=SURFACE, linewidth=1.2)
    ax.axvline(threshold, color=CRITICAL, linewidth=1.6)
    ax.annotate(f"threshold {threshold:.2f}", xy=(threshold, 1.0), xycoords=("data", "axes fraction"),
                xytext=(4, -10), textcoords="offset points", color=CRITICAL, fontsize=LABEL_SIZE)


def confidence_histogram(ax, confidences: list[float], floor: float = 0.0) -> None:
    """How sure the reader was, per photo."""
    ax.clear()
    if not confidences:
        empty(ax, "Run an inspection with the marking check on.", "OCR confidence")
        return
    style_axes(ax, "OCR confidence", "mean confidence per photo", "photos")
    ax.hist(confidences, bins=min(20, max(5, len(confidences) // 2)), range=(0, 1),
            color=SERIES[0], edgecolor=SURFACE, linewidth=1.2)
    if floor > 0:
        ax.axvline(floor, color=CRITICAL, linewidth=1.6)
        ax.annotate(f"reject below {floor:g}", xy=(floor, 1.0), xycoords=("data", "axes fraction"),
                    xytext=(4, -10), textcoords="offset points", color=CRITICAL, fontsize=LABEL_SIZE)


def accuracy_curves(ax, epochs: list[int], char: list[float], seq: list[float]) -> None:
    """The reader's validation accuracy per epoch -- two series on one scale."""
    ax.clear()
    if not epochs:
        empty(ax, "Train the marking reader to get a history.", "Reader accuracy per epoch")
        return
    style_axes(ax, "Reader accuracy per epoch", "epoch", "accuracy")
    ax.plot(epochs, char, color=SERIES[0], linewidth=2, label="character")
    ax.plot(epochs, seq, color=SERIES[1], linewidth=2, label="sequence")
    ax.set_ylim(0, 1.02)
    # Direct-label the endpoints only: a number on every epoch would be noise.
    # The two curves converge to within a thousandth, so the labels are pushed
    # apart by hand -- otherwise they print on top of each other.
    for values, color, dy in ((char, SERIES[0], 5), (seq, SERIES[1], -11)):
        if values:
            ax.annotate(f"{values[-1]:.3f}", xy=(epochs[-1], values[-1]),
                        xytext=(4, dy), textcoords="offset points", color=color,
                        fontsize=LABEL_SIZE)
    legend = ax.legend(loc="lower right", frameon=False, fontsize=LABEL_SIZE)
    for text in legend.get_texts():
        text.set_color(INK_2)          # identity comes from the mark, not the text
    ax.set_xlim(min(epochs), max(epochs) + (max(epochs) - min(epochs)) * 0.12 + 0.5)


def loss_curve(ax, epochs: list[int], loss: list[float]) -> None:
    """CTC loss per epoch. Log scale: it drops three orders of magnitude."""
    ax.clear()
    if not epochs:
        empty(ax, "Train the marking reader to get a history.", "Reader training loss")
        return
    style_axes(ax, "Reader training loss", "epoch", "CTC loss (log)")
    ax.plot(epochs, loss, color=SERIES[0], linewidth=2)
    positive = [v for v in loss if v > 0]
    if positive:
        ax.set_yscale("log")
    if loss:
        ax.annotate(f"{loss[-1]:.4f}", xy=(epochs[-1], loss[-1]), xytext=(4, 2),
                    textcoords="offset points", color=SERIES[0], fontsize=LABEL_SIZE)
    ax.set_xlim(min(epochs), max(epochs) + (max(epochs) - min(epochs)) * 0.12 + 0.5)


def _blend(low: str, high: str, amount: float) -> str:
    """A step of the sequential ramp, `amount` of the way along it."""
    def channels(value: str) -> tuple[int, int, int]:
        return tuple(int(value[i:i + 2], 16) for i in (1, 3, 5))   # type: ignore[return-value]

    a, b = channels(low), channels(high)
    amount = max(0.0, min(1.0, amount))
    return "#" + "".join(f"{round(x + (y - x) * amount):02x}" for x, y in zip(a, b))


def confusion_matrix(ax, grid: list[list[int]], labels=("good", "defect")) -> None:
    """Verdict against truth, as the 2x2 everyone reads it as.

    Four counts is a table, so it is drawn as one -- cells shaded by magnitude
    in a single hue, with the number written in each. The two cells that matter
    are named: a missed defect ships, a false alarm scraps a good part.
    """
    from matplotlib.patches import Rectangle

    ax.clear()
    total = sum(sum(row) for row in grid)
    if not total:
        empty(ax, "Inspect photos from test/good and test/defect\n"
                  "to score the verdicts against their folders.", "Confusion matrix")
        return

    ax.set_facecolor(SURFACE)
    ax.set_title(f"Confusion matrix (n={total})", color=INK, fontsize=TITLE_SIZE,
                 loc="left", pad=6)
    biggest = max(max(row) for row in grid) or 1
    # The row and column say which cell this is; the word only has to say what
    # it costs -- a missed part ships, a false alarm scraps a good one.
    names = [["correct", "false alarm"], ["missed", "caught"]]
    for row in range(2):
        for column in range(2):
            count = grid[row][column]
            shade = _blend(SEQUENTIAL[0], SEQUENTIAL[1], count / biggest)
            ax.add_patch(Rectangle((column, 1 - row), 0.96, 0.96,
                                   facecolor=shade, edgecolor=SURFACE, linewidth=2))
            # White ink on the dark end of the ramp, dark ink on the light end.
            ink = "#ffffff" if count / biggest > 0.55 else INK
            ax.text(column + 0.48, 1 - row + 0.60, f"{count}", ha="center", va="center",
                    color=ink, fontsize=15)
            ax.text(column + 0.48, 1 - row + 0.26, names[row][column], ha="center", va="center",
                    color=ink, fontsize=LABEL_SIZE - 2)
    ax.set_xlim(0, 2)
    ax.set_ylim(0, 2)
    ax.set_xticks([0.48, 1.48])
    ax.set_yticks([1.48, 0.48])
    ax.set_xticklabels(["passed", "failed"], color=MUTED, fontsize=LABEL_SIZE)
    ax.set_yticklabels(labels, color=MUTED, fontsize=LABEL_SIZE)
    ax.set_xlabel("the window said", color=INK_2, fontsize=LABEL_SIZE)
    ax.set_ylabel("the part was", color=INK_2, fontsize=LABEL_SIZE)
    ax.tick_params(length=0, colors=MUTED, labelsize=LABEL_SIZE)
    for side in ax.spines.values():
        side.set_visible(False)
    ax.grid(False)


def iou_histogram(ax, values: list[float]) -> None:
    """Overlap with the hand-drawn masks, per photo."""
    ax.clear()
    if not values:
        empty(ax, "No ground-truth masks.\nAdd them to ground_truth/defect\n"
                  "(same name, white = defect).", "Damage overlap (IoU)")
        return
    style_axes(ax, "Damage overlap (IoU)", "intersection over union", "photos")
    ax.hist(values, bins=min(20, max(5, len(values) // 2)), range=(0, 1),
            color=SERIES[0], edgecolor=SURFACE, linewidth=1.2)
    mean = sum(values) / len(values)
    ax.axvline(mean, color=INK_2, linewidth=1.4)
    ax.annotate(f"mean {mean:.3f}", xy=(mean, 1.0), xycoords=("data", "axes fraction"),
                xytext=(4, -10), textcoords="offset points", color=INK_2, fontsize=LABEL_SIZE)


def new_figure(width: float, height: float, dpi: int = 100):
    """A 2x3 grid of charts on the app's surface: this run on top, the models below."""
    from matplotlib.figure import Figure

    figure = Figure(figsize=(width, height), dpi=dpi, facecolor=SURFACE, layout="constrained")
    axes = figure.subplots(2, 3)
    for ax in axes.flat:
        ax.set_facecolor(SURFACE)
    return figure, axes
