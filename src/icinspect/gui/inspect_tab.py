"""The Inspect tab: pick photos, run both checks, see where the reader looked."""

from __future__ import annotations

import re
import tkinter as tk
from datetime import datetime
from pathlib import Path
from tkinter import messagebox, ttk

from icinspect.gui import widgets as w
from icinspect.gui.runner import Progress
from icinspect.gui.service import (
    REGION_MODES,
    REGION_REFERENCE,
    Inspector,
    Options,
    Result,
    write_report,
)
from icinspect.config import CONFIG_DIR
from icinspect.gui import heatmap
from icinspect.gui.viewer import (
    CHIP_COLOR,
    IGNORED_COLOR,
    READ_COLOR,
    HeatLegend,
    ImageView,
    Overlay,
)
from icinspect.ocr.dataset import list_images
from icinspect.ocr.segment import MODES

VERSION = r"v\d+"   # anomalib's run folders: v1, v2, ...
PART_SCORE_LINE = re.compile(r"^(\s*part_score\s*:\s*).*$", re.MULTILINE)
REGION_COLOR = "#4a3aa7"    # the window that was scored, distinct from package and reads


def _percent(text: str) -> float | None:
    """A percentage typed into a box, as a share (0..1). Blank means no limit."""
    value = w.as_float(text, None)
    return None if value is None else max(0.0, value) / 100.0


def _pixels(region, image_path):
    """A region given as fractions of the frame, in this photo's pixels."""
    if region is None:
        return None
    from PIL import Image

    try:
        with Image.open(image_path) as img:
            width, height = img.size
    except Exception:
        return None
    x1, y1, x2, y2 = region
    return (int(x1 * width), int(y1 * height), int(x2 * width), int(y2 * height))
VIEW_PHOTO = "photo only"
VIEW_HEAT = "heat over photo"
VIEW_JET = "anomalib map (JET)"
VIEW_PANELS = "anomalib panels"
VIEW_MODES = (VIEW_PHOTO, VIEW_HEAT, VIEW_JET, VIEW_PANELS)

SPLIT_HELP = {
    "auto": "auto - find the package, then its text lines (use this for photos of a chip)",
    "half": "half - find the package, cut it into equal strips top to bottom",
    "none": "none - the input is already a single-line crop",
}


class InspectTab(ttk.Frame):
    def __init__(self, master: tk.Misc, app) -> None:
        super().__init__(master, padding=8)
        self.app = app
        self.inspector: Inspector = app.inspector
        self.worker = app.worker   # one heavy job at a time, window-wide
        self.results: list[Result] = []
        self._chip_offset = 0   # the chip box sits before the segments in the overlay list

        self.columnconfigure(0, weight=1)
        self.rowconfigure(3, weight=1)

        self._build_source()
        self._build_options()
        self._build_actions()
        self._build_results()
        self._build_footer()
        self.refresh_state()

    # -- layout ------------------------------------------------------------

    def _build_source(self) -> None:
        box = ttk.LabelFrame(self, text="1. What to inspect", padding=6)
        box.grid(row=0, column=0, sticky="ew")
        box.columnconfigure(0, weight=1)

        self.source = tk.StringVar()
        entry = ttk.Entry(box, textvariable=self.source)
        entry.grid(row=0, column=0, sticky="ew", padx=(0, 6))
        entry.bind("<Return>", lambda _e: self.run())

        ttk.Button(box, text="Photo...", width=10, command=self._pick_image).grid(row=0, column=1)
        ttk.Button(box, text="Folder...", width=10, command=self._pick_folder).grid(row=0, column=2, padx=4)

        quick = ttk.Menubutton(box, text="Dataset", width=10)
        menu = tk.Menu(quick, tearoff=False)
        for label, getter in (
            ("raw captures", lambda p: p.raw_root),
            ("anomaly / train / good", lambda p: p.anomaly_root / "train" / "good"),
            ("anomaly / test / good", lambda p: p.anomaly_root / "test" / "good"),
            ("anomaly / test / defect", lambda p: p.anomaly_root / "test" / "defect"),
            ("ocr / images", lambda p: p.ocr_images),
        ):
            menu.add_command(label=label, command=lambda g=getter: self._use_dataset_folder(g))
        quick["menu"] = menu
        quick.grid(row=0, column=3)

        self.count_label = ttk.Label(box, text="", foreground="#8b949e")
        self.count_label.grid(row=1, column=0, columnspan=4, sticky="w", pady=(4, 0))
        self.source.trace_add("write", lambda *_: self._update_count())

    def _build_options(self) -> None:
        box = ttk.LabelFrame(self, text="2. Checks", padding=6)
        box.grid(row=1, column=0, sticky="ew", pady=(6, 0))

        self.check_anomaly = tk.BooleanVar(value=True)
        self.check_ocr = tk.BooleanVar(value=True)
        self.threshold = tk.StringVar()
        self.split = tk.StringVar(value="auto")
        self.expect = tk.StringVar()
        self.min_confidence = tk.StringVar(value="0.0")

        row = ttk.Frame(box)
        row.grid(row=0, column=0, sticky="ew")
        ttk.Checkbutton(row, text="Surface defects (PatchCore)", variable=self.check_anomaly,
                        command=self.refresh_state).pack(side="left")
        self.anomaly_model = tk.StringVar()
        self.anomaly_picker = ttk.Combobox(row, textvariable=self.anomaly_model,
                                           state="readonly", width=26)
        self.anomaly_picker.pack(side="left", padx=(8, 0))
        self.anomaly_picker.bind("<<ComboboxSelected>>", lambda _e: self._model_chosen("anomaly"))
        ttk.Label(row, text="score over").pack(side="left", padx=(12, 4))
        ttk.Entry(row, textvariable=self.threshold, width=7).pack(side="left")
        ttk.Label(row, text="= FAIL  (blank: the model's own)",
                  foreground="#8b949e").pack(side="left", padx=4)

        part = ttk.Frame(box)
        part.grid(row=1, column=0, sticky="ew", pady=(2, 0))
        self.ignore_background = tk.BooleanVar(value=True)
        ttk.Checkbutton(part, text="ignore the background (score the part only)",
                        variable=self.ignore_background,
                        command=self._background_toggled).pack(side="left", padx=(18, 0))
        ttk.Label(part, text="region").pack(side="left", padx=(10, 3))
        self.region_mode = tk.StringVar(value=REGION_REFERENCE)
        region_picker = ttk.Combobox(part, textvariable=self.region_mode, state="readonly",
                                     width=12, values=list(REGION_MODES))
        region_picker.pack(side="left")
        region_picker.bind("<<ComboboxSelected>>", lambda _e: self._background_toggled())
        self.calibrate_button = ttk.Button(part, text="Calibrate", width=10,
                                           command=self._calibrate)
        self.calibrate_button.pack(side="left", padx=8)
        self.cutoff_label = ttk.Label(part, text="", foreground="#8b949e")
        self.cutoff_label.pack(side="left")

        row2 = ttk.Frame(box)
        row2.grid(row=2, column=0, sticky="ew", pady=(4, 0))
        ttk.Checkbutton(row2, text="Marking text (OCR)", variable=self.check_ocr,
                        command=self.refresh_state).pack(side="left")
        self.ocr_model = tk.StringVar()
        self.ocr_picker = ttk.Combobox(row2, textvariable=self.ocr_model,
                                       state="readonly", width=26)
        self.ocr_picker.pack(side="left", padx=(8, 0))
        self.ocr_picker.bind("<<ComboboxSelected>>", lambda _e: self._model_chosen("ocr"))
        ttk.Label(row2, text="expected part").pack(side="left", padx=(12, 4))
        ttk.Entry(row2, textvariable=self.expect, width=18).pack(side="left")
        ttk.Label(row2, text="min confidence").pack(side="left", padx=(12, 4))
        ttk.Entry(row2, textvariable=self.min_confidence, width=6).pack(side="left")

        row3 = ttk.Frame(box)
        row3.grid(row=3, column=0, sticky="ew", pady=(4, 0))
        ttk.Label(row3, text="how to find the marking").pack(side="left")
        combo = ttk.Combobox(row3, textvariable=self.split, values=list(MODES),
                             state="readonly", width=6)
        combo.pack(side="left", padx=4)
        self.split_hint = ttk.Label(row3, text=SPLIT_HELP["auto"], foreground="#8b949e")
        self.split_hint.pack(side="left", padx=4)
        combo.bind("<<ComboboxSelected>>",
                   lambda _e: self.split_hint.configure(text=SPLIT_HELP[self.split.get()]))

        advanced = w.Collapsible(box, text="More options")
        advanced.grid(row=4, column=0, sticky="ew", pady=(6, 0))
        body = advanced.body
        self.max_lines = tk.StringVar(value="0")
        self.min_char_confidence = tk.StringVar()
        self.max_cer = tk.StringVar(value="0.0")
        self.detect_chip = tk.BooleanVar(value=True)
        self.save_crops = tk.BooleanVar(value=False)
        w.labelled(body, "max lines (0 = all)", ttk.Entry(body, textvariable=self.max_lines, width=6), 0, 0)
        w.labelled(body, "min char confidence",
                   ttk.Entry(body, textvariable=self.min_char_confidence, width=6), 0, 2)
        w.labelled(body, "tolerated CER vs expected",
                   ttk.Entry(body, textvariable=self.max_cer, width=6), 0, 4)
        self.max_damage = tk.StringVar()
        w.labelled(body, "fail over __% of the part damaged",
                   ttk.Entry(body, textvariable=self.max_damage, width=6), 1, 0)
        ttk.Checkbutton(body, text="find the package first", variable=self.detect_chip).grid(
            row=2, column=0, columnspan=2, sticky="w", padx=8)
        ttk.Checkbutton(body, text="save line crops to results/ocr/crops",
                        variable=self.save_crops).grid(row=2, column=2, columnspan=3, sticky="w")

    def _build_actions(self) -> None:
        bar = ttk.Frame(self)
        bar.grid(row=2, column=0, sticky="ew", pady=6)
        bar.columnconfigure(3, weight=1)
        self.run_button = ttk.Button(bar, text="Inspect", command=self.run, style="Accent.TButton")
        self.run_button.grid(row=0, column=0)
        self.stop_button = ttk.Button(bar, text="Stop", command=self.worker.cancel, state="disabled")
        self.stop_button.grid(row=0, column=1, padx=6)
        self.progress = ttk.Progressbar(bar, mode="determinate", length=160)
        self.progress.grid(row=0, column=2, padx=6)
        self.status = ttk.Label(bar, text="Ready", foreground="#8b949e")
        self.status.grid(row=0, column=3, sticky="w")

    def _build_results(self) -> None:
        panes = ttk.PanedWindow(self, orient="horizontal")
        panes.grid(row=3, column=0, sticky="nsew")

        left = ttk.Frame(panes)
        left.rowconfigure(1, weight=1)
        left.columnconfigure(0, weight=1)

        filters = ttk.Frame(left)
        filters.grid(row=0, column=0, sticky="ew", pady=(0, 4))
        self.filter = tk.StringVar(value="all")
        for text, value in (("all", "all"), ("FAIL only", "fail"), ("PASS only", "pass")):
            ttk.Radiobutton(filters, text=text, value=value, variable=self.filter,
                            command=self._refill_tree).pack(side="left", padx=(0, 8))
        self.summary = ttk.Label(filters, text="", foreground="#8b949e")
        self.summary.pack(side="right")

        columns = ("verdict", "score", "damage", "iou", "marking", "confidence", "reasons")
        self.tree = ttk.Treeview(left, columns=columns, show="tree headings", selectmode="browse")
        self.tree.heading("#0", text="photo")
        self.tree.column("#0", width=180, stretch=True)
        for name, width, anchor in (
            ("verdict", 60, "center"), ("score", 70, "e"), ("damage", 70, "e"),
            ("iou", 60, "e"), ("marking", 130, "w"), ("confidence", 70, "e"),
            ("reasons", 120, "w"),
        ):
            self.tree.heading(name, text=name)
            self.tree.column(name, width=width, anchor=anchor, stretch=(name == "marking"))
        self.tree.tag_configure("PASS", foreground="#3fb950")
        self.tree.tag_configure("FAIL", foreground="#ff7b72")
        scroll = ttk.Scrollbar(left, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=scroll.set)
        self.tree.grid(row=1, column=0, sticky="nsew")
        scroll.grid(row=1, column=1, sticky="ns")
        self.tree.bind("<<TreeviewSelect>>", lambda _e: self._show_selected())
        panes.add(left, weight=1)

        right = ttk.Frame(panes)
        right.rowconfigure(0, weight=3)
        right.rowconfigure(4, weight=1)
        right.columnconfigure(0, weight=1)
        self.view = ImageView(right)
        self.view.grid(row=0, column=0, sticky="nsew")

        tools = ttk.Frame(right)
        tools.grid(row=1, column=0, sticky="ew", pady=2)
        ttk.Button(tools, text="Fit", width=5, command=self.view.reset_view).pack(side="left")
        ttk.Button(tools, text="Package", width=9, command=self._zoom_package).pack(side="left", padx=2)
        ttk.Button(tools, text="+", width=3, command=self.view.zoom_in).pack(side="left", padx=2)
        ttk.Button(tools, text="-", width=3, command=self.view.zoom_out).pack(side="left")
        self.show_boxes = tk.BooleanVar(value=True)
        ttk.Checkbutton(tools, text="boxes", variable=self.show_boxes,
                        command=self._show_selected).pack(side="left", padx=8)
        self.zoom_package = tk.BooleanVar(value=True)
        ttk.Checkbutton(tools, text="zoom to package", variable=self.zoom_package,
                        command=self._show_selected).pack(side="left")
        ttk.Button(tools, text="Open folder",
                   command=self._reveal_current).pack(side="right")

        heat = ttk.Frame(right)
        heat.grid(row=2, column=0, sticky="ew", pady=(0, 3))
        ttk.Label(heat, text="view").pack(side="left")
        self.view_mode = tk.StringVar(value=VIEW_HEAT)
        modes = ttk.Combobox(heat, textvariable=self.view_mode, state="readonly", width=20,
                             values=VIEW_MODES)
        modes.pack(side="left", padx=(4, 8))
        modes.bind("<<ComboboxSelected>>", lambda _e: self._show_selected())
        self.show_mask = tk.BooleanVar(value=True)
        ttk.Checkbutton(heat, text="defect outline", variable=self.show_mask,
                        command=self._show_selected).pack(side="left")
        self.sensitivity = tk.StringVar(value="near threshold")
        self.sensitivity_picker = ttk.Combobox(heat, textvariable=self.sensitivity,
                                               state="readonly", width=14,
                                               values=list(heatmap.SENSITIVITY))
        self.sensitivity_picker.pack(side="left", padx=8)
        self.sensitivity_picker.bind("<<ComboboxSelected>>", lambda _e: self._show_selected())
        self.heat_legend = HeatLegend(heat)
        self.heat_legend.pack(side="left", padx=4)
        ttk.Button(heat, text="Save view...", command=self._save_view).pack(side="right")

        self.detail = ttk.Label(right, text="", foreground="#52514e")
        self.detail.grid(row=3, column=0, sticky="w", pady=(0, 3))

        reads = ttk.Treeview(right, columns=("text", "confidence", "note"),
                             show="headings", height=5, selectmode="browse")
        for name, width, anchor in (("text", 160, "w"), ("confidence", 80, "e"), ("note", 160, "w")):
            reads.heading(name, text=name)
            reads.column(name, width=width, anchor=anchor)
        reads.tag_configure("ignored", foreground="#8b949e")
        reads.grid(row=4, column=0, sticky="nsew")
        reads.bind("<<TreeviewSelect>>", lambda _e: self._highlight_segment())
        reads.bind("<Double-Button-1>", lambda _e: self._zoom_segment())
        self.reads = reads
        panes.add(right, weight=2)
        # ttk places the sash by requested widths, which puts the photo in the
        # smaller half; nudge it once the frame has a real width.
        self.after(200, lambda: self._place_sash(panes))

    def _place_sash(self, panes: ttk.PanedWindow) -> None:
        width = self.winfo_width()
        if width < 400:
            self.after(200, lambda: self._place_sash(panes))
            return
        try:
            panes.sashpos(0, int(width * 0.44))
        except tk.TclError:
            pass

    def _build_footer(self) -> None:
        bar = ttk.Frame(self)
        bar.grid(row=4, column=0, sticky="ew", pady=(6, 0))
        ttk.Button(bar, text="Save report as CSV...", command=self._export).pack(side="left")
        ttk.Button(bar, text="Open results folder",
                   command=lambda: w.reveal(self.inspector.paths().results_dir)).pack(side="left", padx=6)
        self.model_label = ttk.Label(bar, text="", foreground="#8b949e")
        self.model_label.pack(side="right")

    # -- source pickers ----------------------------------------------------

    def _pick_image(self) -> None:
        chosen = w.ask_image(self, self.source.get() or str(self.inspector.paths().raw_root))
        if chosen:
            self.source.set(chosen)

    def _pick_folder(self) -> None:
        chosen = w.ask_folder(self, "Choose a folder of photos",
                              self.source.get() or str(self.inspector.paths().raw_root))
        if chosen:
            self.source.set(chosen)

    def _use_dataset_folder(self, getter) -> None:
        self.source.set(str(getter(self.inspector.paths())))

    def _images(self) -> list[Path]:
        raw = self.source.get().strip().strip('"')
        if not raw:
            return []
        path = Path(raw)
        return list_images(path) if path.exists() else []

    def _update_count(self) -> None:
        raw = self.source.get().strip()
        if not raw:
            self.count_label.configure(text="Pick a photo or a folder of photos.")
            return
        path = Path(raw.strip('"'))
        if not path.exists():
            self.count_label.configure(text="That path does not exist.")
            return
        found = len(self._images())
        self.count_label.configure(text=f"{found} image(s) found" if found else "No images here.")

    # -- running -----------------------------------------------------------

    def options(self) -> Options:
        return Options(
            check_anomaly=self.check_anomaly.get(),
            check_ocr=self.check_ocr.get(),
            split=self.split.get(),
            expect=self.expect.get(),
            max_cer=w.as_float(self.max_cer.get(), 0.0) or 0.0,
            min_confidence=w.as_float(self.min_confidence.get(), 0.0) or 0.0,
            max_lines=w.as_int(self.max_lines.get(), 0),
            min_char_confidence=w.as_float(self.min_char_confidence.get(), None),
            threshold=w.as_float(self.threshold.get(), None),
            detect_chip=self.detect_chip.get(),
            save_crops=self.save_crops.get(),
            ignore_background=self.ignore_background.get(),
            part_region_mode=self.region_mode.get(),
            max_damage=_percent(self.max_damage.get()),
        )

    def run(self) -> None:
        if self.worker.busy:
            messagebox.showinfo("Busy", "Another job is already running.", parent=self)
            return
        images = self._images()
        if not images:
            messagebox.showwarning("Nothing to inspect",
                                   "Choose a photo or a folder that contains photos.", parent=self)
            return
        opts = self.options()
        if not opts.check_anomaly and not opts.check_ocr:
            messagebox.showwarning("Nothing to check", "Tick at least one of the two checks.", parent=self)
            return

        self.results = []
        self.tree.delete(*self.tree.get_children())
        self.reads.delete(*self.reads.get_children())
        self.view.show(None)
        self.progress.configure(value=0, maximum=len(images))
        self._busy(True)
        self.app.log(f"[inspect] {len(images)} image(s) from {self.source.get()}")

        inspector = self.inspector

        def task(prog: Progress) -> list[Result]:
            return inspector.inspect(images, opts, prog)

        started = self.worker.start(
            task,
            on_log=self.app.log,
            on_status=lambda text: self.status.configure(text=text),
            on_progress=lambda done_total: self.progress.configure(
                value=done_total[0], maximum=max(done_total[1], 1)),
            on_partial=self._add_result,
            on_done=self._finished,
            on_error=self._failed,
            on_cancelled=lambda _p: (self._busy(False), self.status.configure(text="Stopped")),
        )
        if not started:
            self._busy(False)

    def _busy(self, running: bool) -> None:
        self.run_button.configure(state="disabled" if running else "normal")
        self.stop_button.configure(state="normal" if running else "disabled")

    def _finished(self, results: list[Result]) -> None:
        self._busy(False)
        self.results = results
        failed = sum(1 for r in results if r.verdict == "FAIL")
        self.status.configure(text=f"{len(results)} inspected: {len(results) - failed} PASS / {failed} FAIL")
        self.app.log(f"[inspect] {len(results) - failed} PASS / {failed} FAIL", "ok" if not failed else None)
        self._refill_tree()
        # The Stats tab plots this run, so it is stale the moment one finishes.
        stats_tab = getattr(self.app, "stats_tab", None)
        if stats_tab is not None:
            stats_tab.refresh_state()
        if self.tree.get_children():
            first = self.tree.get_children()[0]
            self.tree.selection_set(first)
            self.tree.focus(first)

    def _failed(self, payload) -> None:
        exc, trace = payload
        self._busy(False)
        self.status.configure(text="Failed")
        self.app.log(trace, "error")
        messagebox.showerror("Inspection failed", str(exc), parent=self)

    # -- results -----------------------------------------------------------

    def _add_result(self, result: Result) -> None:
        self.results.append(result)
        self._insert(result, len(self.results) - 1)
        self._update_summary()

    def _passes_filter(self, result: Result) -> bool:
        wanted = self.filter.get()
        return wanted == "all" or (wanted == "fail") == (result.verdict == "FAIL")

    def _insert(self, result: Result, index: int) -> None:
        if not self._passes_filter(result):
            return
        self.tree.insert(
            "", "end", iid=str(index), text=result.path.name,
            values=(
                result.verdict,
                "" if result.deciding_score is None else f"{result.deciding_score:.4f}",
                "" if result.damage_ratio is None else f"{result.damage_ratio * 100:.1f}%",
                "" if result.iou is None else f"{result.iou:.3f}",
                result.marking,
                f"{result.confidence:.3f}" if result.reads else "",
                ";".join(result.reasons),
            ),
            tags=(result.verdict,),
        )

    def _refill_tree(self) -> None:
        selected = self.tree.selection()
        self.tree.delete(*self.tree.get_children())
        for index, result in enumerate(self.results):
            self._insert(result, index)
        self._update_summary()
        if selected and self.tree.exists(selected[0]):
            self.tree.selection_set(selected[0])

    def _update_summary(self) -> None:
        total = len(self.results)
        failed = sum(1 for r in self.results if r.verdict == "FAIL")
        self.summary.configure(text=f"{total - failed} PASS / {failed} FAIL" if total else "")

    def _selected_result(self) -> Result | None:
        selection = self.tree.selection()
        if not selection:
            return None
        index = int(selection[0])
        return self.results[index] if index < len(self.results) else None

    def _show_selected(self) -> None:
        result = self._selected_result()
        self.reads.delete(*self.reads.get_children())
        if result is None:
            self.view.show(None)
            self.view.set_heat(None, None)
            self.detail.configure(text="")
            return

        overlays: list[Overlay] = []
        if result.part:
            # What was actually scored. On a part with a chunk missing this is
            # the whole answer to "why did that count as a defect".
            with_size = _pixels(result.part, result.path)
            if with_size:
                overlays.append(Overlay(with_size, REGION_COLOR, "scored region", dashed=True))
        if result.chip:
            overlays.append(Overlay(result.chip, CHIP_COLOR, "package"))
        for position, seg in enumerate(result.segments):
            overlays.append(Overlay(
                seg.box,
                IGNORED_COLOR if seg.ignored else READ_COLOR,
                f"{seg.label} {seg.text}" if not seg.ignored else f"{seg.label} ignored",
                dashed=seg.ignored,
            ))
            note = ""
            if seg.ignored:
                note = f"ignored (read {seg.raw_text or '-'})"
            elif seg.dropped:
                note = f"{seg.dropped} unsure char(s) dropped, raw {seg.raw_text}"
            self.reads.insert(
                "", "end", iid=str(position),
                values=(seg.text or "-", f"{seg.confidence:.3f}", note),
                tags=("ignored",) if seg.ignored else (),
            )
        mode = self.view_mode.get()
        if mode == VIEW_PANELS and result.heat is not None:
            # A composite of four captioned panels: there is no single photo to
            # draw the marking boxes on, because the panels are the picture.
            self.view.show_composite(self._panels(result), key=str(result.path))
            self.view.set_heat(None, None)
        else:
            self.view.show(result.path, overlays if self.show_boxes.get() else [])
            # scored_heat, not heat: the picture has to be what was counted, so
            # an excluded background is visibly excluded.
            # The outline is the damaged area the ratio counts, so the picture
            # and the percentage under it are the same thing.
            self.view.set_heat(result.scored_heat,
                               result.damage if result.damage is not None else result.scored_mask,
                               limit=result.part)
        self._apply_heat_view()
        self._show_detail(result)
        if result.chip and self.zoom_package.get() and mode != VIEW_PANELS:
            self.view.zoom_to(result.chip)
        # Segments come after the region and package boxes in the overlay list.
        self._chip_offset = (1 if result.part else 0) + (1 if result.chip else 0)

    def _panels(self, result: Result):
        """anomalib's visualizer grid for this photo: image, image + map, map, mask."""
        from PIL import Image

        with Image.open(result.path) as photo:
            photo.load()
            # Wider than it needs to be at fit zoom: the panels are worth
            # zooming into, and the grid is only built when the photo changes.
            return heatmap.panel_view(
                photo, result.scored_heat,
                result.damage if result.damage is not None else result.scored_mask, style="jet",
                panel_width=900, floor=self._floor(), limit=result.part,
            )

    def _floor(self) -> float:
        return heatmap.SENSITIVITY.get(self.sensitivity.get(), heatmap.FLOOR)

    def _apply_heat_view(self) -> None:
        """Push the view mode onto the canvas and the legend.

        The JET view paints the whole frame, so the sensitivity floor means
        nothing there -- its box is disabled rather than left looking connected.
        """
        mode = self.view_mode.get()
        floor = self._floor()
        jet = mode in (VIEW_JET, VIEW_PANELS)
        self.view.set_heat_view(
            visible=mode != VIEW_PHOTO,
            mask=self.show_mask.get(),
            floor=floor,
            style="jet" if jet else "threshold",
            opacity=0.5 if jet else 0.75,
        )
        self.sensitivity_picker.configure(state="disabled" if jet else "readonly")
        self.heat_legend.draw(floor, style="jet" if jet else "threshold")

    def _show_detail(self, result: Result) -> None:
        """The numbers behind the picture, for the selected photo."""
        if result.score is None:
            self.detail.configure(text="anomaly check not run")
            return
        if result.part_score is not None and result.part_decides:
            cutoff = self.inspector.part_cutoff(self.options())
            parts = [f"part score {result.part_score:.4f} (cutoff {cutoff:.4f}, decides)",
                     f"whole frame {result.score:.4f}"]
        elif result.part_score is not None:
            parts = [f"whole frame {result.score:.4f} (threshold {heatmap.THRESHOLD:g}, decides)",
                     f"part only {result.part_score:.4f} - not fitted yet"]
        else:
            parts = [f"score {result.score:.4f} (threshold {heatmap.THRESHOLD:g})"]
            if self.ignore_background.get():
                parts.append("part not found - whole frame scored")
        if result.peak_score is not None:
            parts.append(f"hottest pixel {result.peak_score:.3f}")
        if result.damage_ratio is not None:
            where = "of the part" if result.part else "of the frame"
            parts.append(f"damaged {result.damage_ratio * 100:.1f}% {where}")
        self.detail.configure(text="   ".join(parts))

    def _zoom_package(self) -> None:
        result = self._selected_result()
        if result is None:
            return
        # No package box (split none, or the search gave up): show it all.
        self.view.zoom_to(result.chip) if result.chip else self.view.reset_view()

    def _zoom_segment(self) -> None:
        result = self._selected_result()
        selection = self.reads.selection()
        if result is None or not selection:
            return
        index = int(selection[0])
        if index < len(result.segments):
            self.view.zoom_to(result.segments[index].box, margin=0.6)

    def _highlight_segment(self) -> None:
        selection = self.reads.selection()
        offset = getattr(self, "_chip_offset", 0)
        self.view.set_selected(int(selection[0]) + offset if selection else None)

    def _save_view(self) -> None:
        """Write what is on screen -- photo, boxes, heat map -- at full resolution.

        The screenshot people actually want for a report is this picture, not
        the window around it, and at the source photo's own size rather than
        whatever the pane happened to be.
        """
        from tkinter import filedialog

        result = self._selected_result()
        if result is None:
            messagebox.showinfo("Nothing to save", "Select a photo in the results list first.",
                                parent=self)
            return
        chosen = filedialog.asksaveasfilename(
            parent=self, title="Save the annotated photo", defaultextension=".png",
            initialfile=f"{result.path.stem}_inspected.png",
            filetypes=[("PNG image", "*.png"), ("All files", "*.*")],
        )
        if not chosen:
            return
        written = self._render_view(result, Path(chosen))
        self.app.log(f"[inspect] view saved: {written}", "ok")
        self.status.configure(text=f"Saved {written.name}")

    def _render_view(self, result: Result, out: Path) -> Path:
        from PIL import Image, ImageDraw

        mode = self.view_mode.get()
        if mode == VIEW_PANELS and result.heat is not None:
            self._panels(result).save(out)
            return out

        photo = Image.open(result.path)
        photo.load()
        canvas = photo.convert("RGBA")
        if mode != VIEW_PHOTO and result.heat is not None:
            tint = heatmap.region_overlay(
                result.scored_heat,
                result.damage if result.damage is not None else result.scored_mask,
                (0, 0, 1, 1), canvas.size,
                style=self.view.heat_style, opacity=self.view.heat_opacity,
                show_mask=self.show_mask.get(), floor=self._floor(), limit=result.part,
            )
            canvas = Image.alpha_composite(canvas, tint)
        if self.show_boxes.get():
            draw = ImageDraw.Draw(canvas)
            line = max(2, canvas.size[0] // 900)
            region = _pixels(result.part, result.path)
            if region:
                draw.rectangle(region, outline=REGION_COLOR, width=line)
            if result.chip:
                draw.rectangle(result.chip, outline=CHIP_COLOR, width=line)
            for seg in result.segments:
                draw.rectangle(seg.box, outline=IGNORED_COLOR if seg.ignored else READ_COLOR,
                               width=line)
        canvas.convert("RGB").save(out)
        return out

    def _reveal_current(self) -> None:
        result = self._selected_result()
        w.reveal(result.path if result else self.inspector.paths().results_dir)

    def _export(self) -> None:
        if not self.results:
            messagebox.showinfo("Nothing to save", "Run an inspection first.", parent=self)
            return
        default = self.inspector.paths().results_dir / "inspect" / "inspection_report.csv"
        chosen = w.ask_save_csv(self, default.name)
        if not chosen:
            return
        written = write_report(self.results, Path(chosen), self.expect.get().strip().upper())
        self.app.log(f"[inspect] report: {written}", "ok")
        self.status.configure(text=f"Saved {written.name}")

    # -- state -------------------------------------------------------------

    def _model_label(self, ckpt: Path) -> str:
        """A name short enough for the box but specific enough to tell runs apart.

        anomalib buries every run at the same depth under a `vN` folder, so the
        file name alone ("model.ckpt") names five different detectors.
        """
        stamp = datetime.fromtimestamp(ckpt.stat().st_mtime).strftime("%d %b %H:%M")
        if ckpt.suffix == ".ckpt":
            version = next(
                (part for part in reversed(ckpt.parts) if re.fullmatch(VERSION, part)),
                ckpt.parent.name,
            )
            return f"{version}   {stamp}"
        return f"{ckpt.name}   {stamp}"

    def _fill_model_pickers(self) -> None:
        """List every checkpoint on disk, keeping the current choice if it survived."""
        self._anomaly_choices = {self._model_label(p): p
                                 for p in self.inspector.anomaly_checkpoints()}
        self._ocr_choices = {self._model_label(p): p for p in self.inspector.ocr_checkpoints()}
        for picker, choices, var, current in (
            (self.anomaly_picker, self._anomaly_choices, self.anomaly_model,
             self.inspector.anomaly_checkpoint()),
            (self.ocr_picker, self._ocr_choices, self.ocr_model, self.inspector.ocr_checkpoint()),
        ):
            labels = list(choices)
            picker.configure(values=labels, state="readonly" if labels else "disabled")
            if not labels:
                var.set("not trained yet")
                continue
            if var.get() not in labels:
                match = next((name for name, path in choices.items() if path == current), labels[0])
                var.set(match)

    def _background_toggled(self) -> None:
        self._show_cutoff()
        self._show_selected()
        if self.results:
            self.status.configure(text="Run again to score with the new setting")

    def _show_cutoff(self) -> None:
        """Say which line a part-only score is being judged against, and where it came from."""
        if not self.ignore_background.get():
            self.cutoff_label.configure(text="the whole frame is scored, fixture included")
            return
        cutoff, source = self.inspector.part_cutoff_source(self.options())
        wording = {
            "run": f"cutoff {cutoff:.4f} (typed above)",
            "session": f"cutoff {cutoff:.4f} (calibrated this session)",
            "config": f"cutoff {cutoff:.4f} (from patchcore.yaml)",
            "default": "no cutoff fitted yet - the part is measured, "
                       "but PASS/FAIL still comes from the whole frame. Calibrate to change that.",
        }
        self.cutoff_label.configure(text=wording[source],
                                    foreground="#9a6700" if source == "default" else "#8b949e")

    def _calibrate(self) -> None:
        """Fit the part-only cutoff on test/good + test/defect."""
        if self.worker.busy:
            messagebox.showinfo("Busy", "Another job is already running.", parent=self)
            return
        if not self.ignore_background.get():
            messagebox.showinfo(
                "Nothing to calibrate",
                "Calibration fits the cutoff for part-only scoring.\n"
                "Tick \"ignore the background\" first.", parent=self)
            return
        self.app.log("")
        self.app.log("=== Calibrating the part-only cutoff ===")
        self.app.show_log()
        inspector, opts = self.inspector, self.options()

        def task(prog: Progress) -> dict:
            return inspector.calibrate_part_threshold(prog, opts)

        started = self.worker.start(
            task,
            on_log=self.app.log,
            on_status=lambda text: self.status.configure(text=text),
            on_progress=lambda done_total: self.progress.configure(
                value=done_total[0], maximum=max(done_total[1], 1)),
            on_done=self._calibrated,
            on_error=self._failed,
            on_cancelled=lambda _p: (self._busy(False), self.status.configure(text="Stopped")),
        )
        if started:
            self._busy(True)

    def _calibrated(self, best: dict) -> None:
        self._busy(False)
        self._show_cutoff()
        self.status.configure(text=f"Cutoff {best['threshold']:.4f}  F1 {best['f1']:.4f}")
        good = [s for s, y in best["scores"] if y == 0]
        bad = [s for s, y in best["scores"] if y == 1]
        summary = (
            f"Fitted on {best['good']} good and {best['defect']} defective part(s).\n\n"
            f"cutoff      {best['threshold']:.4f}\n"
            f"F1          {best['f1']:.4f}   (precision {best['precision']:.3f}, "
            f"recall {best['recall']:.3f})\n"
            f"AUROC       {best['auroc']:.4f}\n\n"
            f"good parts    {min(good):.4f} - {max(good):.4f}\n"
            f"defective     {min(bad):.4f} - {max(bad):.4f}\n"
            f"part found on {best['found_part']} of {len(best['scores'])} photos\n\n"
            "Save it to configs/patchcore.yaml so the next session starts here?"
        )
        if messagebox.askyesno("Calibrated", summary, parent=self):
            self._pin_cutoff(best["threshold"])
        stats_tab = getattr(self.app, "stats_tab", None)
        if stats_tab is not None:
            stats_tab.refresh_state()

    def _pin_cutoff(self, value: float) -> None:
        """Write `threshold.part_score` into the config, comments intact."""
        path = CONFIG_DIR / self.inspector.patchcore_config
        text = path.read_text(encoding="utf-8")
        new_text, count = PART_SCORE_LINE.subn(lambda m: f"{m.group(1)}{value:.4f}", text, count=1)
        if not count:
            messagebox.showerror("Could not save", f"No part_score line in {path}", parent=self)
            return
        path.write_text(new_text, encoding="utf-8")
        self.app.log(f"[calibrate] part_score {value:.4f} saved to {path}", "ok")
        self._show_cutoff()

    def _model_chosen(self, which: str) -> None:
        """Point the next run at the picked checkpoint. The old one is dropped."""
        if which == "anomaly":
            chosen = self._anomaly_choices.get(self.anomaly_model.get())
            self.inspector.select_anomaly(chosen)
        else:
            chosen = self._ocr_choices.get(self.ocr_model.get())
            self.inspector.select_ocr(chosen)
        if chosen is not None:
            self.app.log(f"[inspect] {which} model: {chosen}")
        stats_tab = getattr(self.app, "stats_tab", None)
        if stats_tab is not None:
            stats_tab.refresh_state()          # the facts table describes this one now

    def refresh_state(self) -> None:
        """Reflect which models exist, so a missing checkpoint is visible up front."""
        self._fill_model_pickers()
        ocr = self.inspector.ocr_checkpoint()
        anomaly = self.inspector.anomaly_checkpoint()
        parts = [
            f"OCR model: {'ready' if ocr else 'not trained'}",
            f"PatchCore: {'ready' if anomaly else 'not trained'}",
        ]
        self.model_label.configure(text="   ".join(parts))
        self._show_cutoff()
        if not self.source.get():
            raw = self.inspector.paths().raw_root
            if raw.exists():
                self.source.set(str(raw))
        self._update_count()
