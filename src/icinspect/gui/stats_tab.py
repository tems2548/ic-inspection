"""The Stats tab: how the models were trained, and how this run went.

Two halves. The left is about *this* inspection -- score distribution, reader
confidence, what failed and why -- and fills in from whatever is currently on
the Inspect tab. The right is about the models on disk: the training history the
reader wrote, and every hyperparameter both models actually carry, which is the
thing to check first when a number looks wrong.

matplotlib is imported when the tab is first shown, not at startup: the window
should open in under a second even though this tab costs a second on its own.
"""

from __future__ import annotations

import tkinter as tk
from tkinter import messagebox, ttk

from icinspect.gui import charts
from icinspect.gui.heatmap import THRESHOLD
from icinspect.gui.stats import RunStats, ocr_facts, ocr_history, patchcore_facts, run_stats


class StatTile(ttk.Frame):
    """One number, said plainly. A two-slice pie chart in disguise, without the pie."""

    def __init__(self, master: tk.Misc, caption: str, **kwargs) -> None:
        super().__init__(master, padding=(10, 6), **kwargs)
        self.value = ttk.Label(self, text="-", font=("Segoe UI", 17), foreground=charts.INK)
        self.value.pack(anchor="w")
        self.caption = ttk.Label(self, text=caption, foreground=charts.MUTED, font=("Segoe UI", 8))
        self.caption.pack(anchor="w")

    def set(self, text: str, color: str | None = None) -> None:
        self.value.configure(text=text, foreground=color or charts.INK)


class StatsTab(ttk.Frame):
    def __init__(self, master: tk.Misc, app) -> None:
        super().__init__(master, padding=8)
        self.app = app
        self.inspector = app.inspector
        self._canvas = None          # the matplotlib canvas, built on first show
        self._axes = None
        self._drawn_once = False

        self.columnconfigure(0, weight=1)
        self.rowconfigure(2, weight=1)
        self._build_tiles()
        self._build_actions()
        self._build_body()

    # -- layout ------------------------------------------------------------

    def _build_tiles(self) -> None:
        bar = ttk.Frame(self)
        bar.grid(row=0, column=0, sticky="ew")
        self.tiles: dict[str, StatTile] = {}
        for column, (key, caption) in enumerate((
            ("inspected", "photos inspected"),
            ("pass_rate", "passed"),
            ("failed", "failed"),
            ("accuracy", "accuracy vs labels"),
            ("recall", "defects caught"),
            ("iou", "mean IoU"),
            ("auroc", "detector test AUROC"),
            ("seq", "reader sequence accuracy"),
        )):
            tile = StatTile(bar, caption)
            tile.grid(row=0, column=column, sticky="w", padx=(0, 18))
            self.tiles[key] = tile

    def _build_actions(self) -> None:
        bar = ttk.Frame(self)
        bar.grid(row=1, column=0, sticky="ew", pady=(6, 6))
        bar.columnconfigure(2, weight=1)
        ttk.Button(bar, text="Refresh", command=self.refresh_state).grid(row=0, column=0)
        ttk.Button(bar, text="Save charts as PNG...", command=self._export).grid(row=0, column=1, padx=6)
        self.note = ttk.Label(bar, text="", foreground=charts.MUTED)
        self.note.grid(row=0, column=2, sticky="w", padx=8)

    def _build_body(self) -> None:
        panes = ttk.PanedWindow(self, orient="horizontal")
        panes.grid(row=2, column=0, sticky="nsew")

        self.chart_frame = ttk.Frame(panes)
        self.chart_frame.rowconfigure(0, weight=1)
        self.chart_frame.columnconfigure(0, weight=1)
        self.placeholder = ttk.Label(
            self.chart_frame, text="Charts appear when this tab is opened.",
            foreground=charts.MUTED, anchor="center",
        )
        self.placeholder.grid(row=0, column=0, sticky="nsew")
        panes.add(self.chart_frame, weight=3)

        right = ttk.Frame(panes)
        right.rowconfigure(0, weight=1)
        right.columnconfigure(0, weight=1)
        tree = ttk.Treeview(right, columns=("value",), show="tree headings", selectmode="browse")
        tree.heading("#0", text="model")
        tree.heading("value", text="value")
        tree.column("#0", width=205, stretch=False)
        tree.column("value", width=230)
        tree.tag_configure("group", foreground=charts.INK)
        scroll = ttk.Scrollbar(right, orient="vertical", command=tree.yview)
        tree.configure(yscrollcommand=scroll.set)
        tree.grid(row=0, column=0, sticky="nsew")
        scroll.grid(row=0, column=1, sticky="ns")
        self.facts = tree
        panes.add(right, weight=2)
        # Six panels need most of the width; ttk would otherwise size the sash
        # from the table's requested width and squeeze them.
        self._panes = panes
        self.after(200, self._place_sash)

    def _place_sash(self) -> None:
        width = self.winfo_width()
        if width < 400:
            self.after(200, self._place_sash)
            return
        try:
            self._panes.sashpos(0, int(width * 0.64))
        except tk.TclError:
            pass

    # -- charts ------------------------------------------------------------

    def _ensure_canvas(self) -> bool:
        """Build the matplotlib canvas the first time the tab is actually looked at."""
        if self._canvas is not None:
            return True
        try:
            from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
        except ImportError:
            self.placeholder.configure(
                text="matplotlib is not installed, so the charts cannot be drawn.\n"
                     "pip install matplotlib -- the numbers on the right still work.")
            return False
        figure, axes = charts.new_figure(10.2, 5.4)
        canvas = FigureCanvasTkAgg(figure, master=self.chart_frame)
        widget = canvas.get_tk_widget()
        self.placeholder.grid_forget()
        widget.grid(row=0, column=0, sticky="nsew")
        self._canvas, self._axes, self._figure = canvas, axes, figure
        # The canvas asks for a lot of width the moment it appears, which shoves
        # the sash over and hides the table -- put it back once it is in place.
        self.after(50, self._place_sash)
        return True

    def _draw(self, stats: RunStats) -> None:
        if not self._ensure_canvas():
            return
        (scores, matrix, overlap), (confidence, accuracy, loss) = self._axes
        history = ocr_history(self.inspector.ocr_dir())
        # Top row is this run, bottom row is the models behind it.
        charts.score_histogram(scores, stats.scores, self._score_cutoff())
        charts.confusion_matrix(matrix, stats.confusion.as_grid())
        charts.iou_histogram(overlap, stats.ious)
        charts.confidence_histogram(confidence, stats.confidences, self._confidence_floor())
        charts.accuracy_curves(accuracy, history.epochs, history.char_accuracy,
                               history.seq_accuracy)
        charts.loss_curve(loss, history.epochs, history.loss)
        self._canvas.draw_idle()
        self._drawn_once = True

    def _score_cutoff(self) -> float:
        """The line the run was judged against -- the calibrated part cutoff, or 0.50."""
        tab = getattr(self.app, "inspect_tab", None)
        if tab is None or not getattr(tab, "ignore_background", None) or not tab.ignore_background.get():
            return THRESHOLD
        return self.inspector.part_cutoff(tab.options())

    def _confidence_floor(self) -> float:
        """The reject line the Inspect tab is set to, so the chart shows the same rule."""
        tab = getattr(self.app, "inspect_tab", None)
        if tab is None:
            return 0.0
        from icinspect.gui.widgets import as_float

        return as_float(tab.min_confidence.get(), 0.0) or 0.0

    def _export(self) -> None:
        if self._canvas is None:
            messagebox.showinfo("Nothing to save", "Open this tab first so the charts are drawn.",
                                parent=self)
            return
        from tkinter import filedialog

        chosen = filedialog.asksaveasfilename(
            parent=self, title="Save charts", defaultextension=".png",
            initialfile="ic-inspection-stats.png",
            filetypes=[("PNG image", "*.png"), ("All files", "*.*")],
        )
        if not chosen:
            return
        self._figure.savefig(chosen, dpi=150, facecolor=charts.SURFACE)
        self.app.log(f"[stats] charts saved: {chosen}", "ok")
        self.note.configure(text=f"Saved {chosen}")

    # -- numbers -----------------------------------------------------------

    def _fill_tiles(self, stats: RunStats, facts: dict[str, str]) -> None:
        self.tiles["inspected"].set(str(stats.total) if stats else "-")
        rate = stats.pass_rate
        self.tiles["pass_rate"].set(
            f"{rate * 100:.0f}%" if rate is not None else "-",
            "#0ca30c" if rate == 1 else None,
        )
        self.tiles["failed"].set(str(stats.failed) if stats else "-",
                                 charts.CRITICAL if stats.failed else None)
        matrix = stats.confusion
        accuracy = matrix.accuracy
        self.tiles["accuracy"].set(
            f"{accuracy * 100:.1f}%" if accuracy is not None else "-",
            charts.CRITICAL if accuracy is not None and accuracy < 1 else None,
        )
        recall = matrix.recall
        # A missed defect is the failure that ships, so it is called out in red.
        self.tiles["recall"].set(
            f"{matrix.tp}/{matrix.tp + matrix.fn}" if recall is not None else "-",
            charts.CRITICAL if matrix.fn else None,
        )
        iou = stats.mean_iou
        self.tiles["iou"].set(f"{iou:.3f}" if iou is not None else "-")
        self.tiles["auroc"].set(facts.get("test image AUROC", "-"))
        self.tiles["seq"].set(facts.get("val sequence accuracy", "-"))

    def _run_rows(self, stats: RunStats) -> list:
        if not stats:
            return [("photos", "no inspection run yet")]
        rows = [
            ("photos", str(stats.total)),
            ("passed", str(stats.passed)),
            ("failed", str(stats.failed)),
        ]
        if stats.scores:
            rows.append(("score range", f"{min(stats.scores):.4f} - {max(stats.scores):.4f}"))
        if stats.areas:
            flagged = [a for a in stats.areas if a > 0]
            rows.append(("showed damage", f"{len(flagged)} of {len(stats.areas)} photos"))
            if flagged:
                rows.append(("worst damage", f"{max(flagged) * 100:.1f}% of the part"))
                rows.append(("median damage",
                             f"{sorted(flagged)[len(flagged) // 2] * 100:.1f}% of the part"))
        if stats.mean_confidence is not None:
            rows.append(("mean OCR confidence", f"{stats.mean_confidence:.4f}"))
        tab = getattr(self.app, "inspect_tab", None)
        if tab is not None and tab.ignore_background.get():
            cutoff, source = self.inspector.part_cutoff_source(tab.options())
            rows.append(("scored region", tab.region_mode.get()))
            rows.append(("part cutoff", f"{cutoff:.4f} ({source})"))
        for reason, count in stats.reasons.most_common():
            rows.append((f"failed: {reason}", str(count)))

        matrix = stats.confusion
        if matrix:
            rows.append(("labelled photos", f"{matrix.labelled} of {stats.total}"))
            rows.append(("caught defects (TP)", str(matrix.tp)))
            rows.append(("missed defects (FN)", str(matrix.fn)))
            rows.append(("false alarms (FP)", str(matrix.fp)))
            rows.append(("correct passes (TN)", str(matrix.tn)))
            for name, value in (("accuracy", matrix.accuracy), ("precision", matrix.precision),
                                ("recall", matrix.recall), ("specificity", matrix.specificity),
                                ("F1", matrix.f1)):
                if value is not None:
                    rows.append((name, f"{value:.4f}"))
        if stats.ious:
            rows.append(("mean IoU", f"{stats.mean_iou:.4f}"))
            rows.append(("worst IoU", f"{min(stats.ious):.4f}"))
            rows.append(("masks compared", f"{len(stats.ious)} of {stats.total}"))
        return rows

    def _fill_facts(self, ocr: list, patchcore: list, stats: RunStats) -> dict[str, str]:
        self.facts.delete(*self.facts.get_children())
        flat: dict[str, str] = {}
        for group, rows in (("This run", self._run_rows(stats)),
                            ("Defect detector (PatchCore)", patchcore),
                            ("Marking reader (CRNN + CTC)", ocr)):
            parent = self.facts.insert("", "end", text=group, values=("",), open=True, tags=("group",))
            for name, value in rows:
                self.facts.insert(parent, "end", text=f"   {name}", values=(value,))
                flat[name] = value
        return flat

    # -- state -------------------------------------------------------------

    def refresh_state(self) -> None:
        """Recompute from the current inspection and the files on disk.

        Called on every tab switch, so it must stay cheap -- the checkpoints are
        read for their metadata, the models are never loaded.
        """
        results = getattr(self.app.inspect_tab, "results", []) if hasattr(self.app, "inspect_tab") else []
        stats = run_stats(results)
        try:
            ocr = ocr_facts(self.inspector.ocr_checkpoint(), self.inspector.ocr_config)
            patchcore = patchcore_facts(
                self.inspector.anomaly_checkpoint(),
                self.inspector.patchcore_dir(),
                self.inspector.patchcore_config,
            )
        except Exception as exc:                      # a half-written checkpoint
            ocr, patchcore = [("error", str(exc))], []
        flat = self._fill_facts(ocr, patchcore, stats)
        self._fill_tiles(stats, flat)
        self.note.configure(
            text="No inspection yet -- the training charts still show."
            if not stats else
            f"{stats.total} photo(s) from the last inspection"
        )
        if self._canvas is not None or self._drawn_once:
            self._draw(stats)

    def on_shown(self) -> None:
        """The tab became visible: build the canvas now, not at startup."""
        self.refresh_state()
        self._draw(run_stats(getattr(self.app.inspect_tab, "results", [])))
