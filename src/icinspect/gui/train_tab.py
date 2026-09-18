"""The Train tab: generate data and train either model without typing a command.

Each button shells out to the same module the README documents, so a run
started here and a run started in PowerShell are the same run -- and the exact
command is echoed into the log, which is how you learn the CLI from the GUI.
"""

from __future__ import annotations

import tkinter as tk
from tkinter import messagebox, ttk

from pathlib import Path

import re
import time

from icinspect.config import PROJECT_ROOT
from icinspect.gui import widgets as w
from icinspect.progress import hms
from icinspect.gui.runner import Progress, python_module, python_script, run_command
from icinspect.gui.service import Options, Result
from icinspect.ocr.dataset import list_images


def _segment_rows(result: Result) -> list[dict]:
    """A result's segments in the row shape `icinspect.ocr.harvest` reads."""
    return [
        {
            "image": str(result.path),
            "line": seg.line,
            "part": seg.part,
            "box": "{},{},{},{}".format(*seg.box),
            "text": seg.text,
            "raw_text": seg.raw_text,
        }
        for seg in result.segments
    ]


class TrainTab(ttk.Frame):
    def __init__(self, master: tk.Misc, app) -> None:
        super().__init__(master, padding=8)
        self.app = app
        self.worker = app.worker
        self.inspector = app.inspector
        self._buttons: list[ttk.Button] = []

        self.columnconfigure(0, weight=1)
        self._build_synth()
        self._build_ocr()
        self._build_patchcore()
        self._build_actions()
        self.refresh_state()

    # -- sections ----------------------------------------------------------

    def _build_synth(self) -> None:
        box = ttk.LabelFrame(self, text="1. Synthetic marking crops (no labelling needed)", padding=6)
        box.grid(row=0, column=0, sticky="ew")
        ttk.Label(
            box,
            text="Renders part numbers and date codes with Windows fonts, then blurs, "
                 "tilts and dirties them. Train on these first; fine-tune on real crops after.",
            foreground="#8b949e", wraplength=760, justify="left",
        ).grid(row=0, column=0, columnspan=6, sticky="w", pady=(0, 6))

        self.synth_count = tk.StringVar(value="20000")
        self.synth_val = tk.StringVar(value="0.1")
        self.synth_split = tk.BooleanVar(value=True)
        w.labelled(box, "how many", ttk.Entry(box, textvariable=self.synth_count, width=10), 1, 0)
        w.labelled(box, "validation share", ttk.Entry(box, textvariable=self.synth_val, width=6), 1, 2)
        ttk.Checkbutton(box, text="write train.txt / val.txt", variable=self.synth_split).grid(
            row=1, column=4, sticky="w", padx=8)
        self._button(box, "Generate crops", self.generate_synth).grid(row=1, column=5, sticky="e", padx=4)
        box.columnconfigure(5, weight=1)

    def _build_ocr(self) -> None:
        box = ttk.LabelFrame(self, text="2. Marking reader (CRNN + CTC)", padding=6)
        box.grid(row=1, column=0, sticky="ew", pady=(8, 0))
        box.columnconfigure(5, weight=1)

        self.ocr_labels = tk.StringVar(value="train.txt")
        self.ocr_epochs = tk.StringVar(value="")
        self.ocr_batch = tk.StringVar(value="")
        self.ocr_lr = tk.StringVar(value="")
        self.ocr_device = tk.StringVar(value="auto")
        self.ocr_resume = tk.BooleanVar(value=False)

        w.labelled(box, "labels", ttk.Entry(box, textvariable=self.ocr_labels, width=22), 0, 0)
        ttk.Button(box, text="...", width=3, command=self._pick_labels).grid(row=0, column=2, sticky="w")
        w.labelled(box, "epochs", ttk.Entry(box, textvariable=self.ocr_epochs, width=6), 0, 3)
        ttk.Label(box, text="blank = the value in configs/ocr.yaml", foreground="#8b949e").grid(
            row=0, column=5, sticky="w", padx=8)

        w.labelled(box, "batch size", ttk.Entry(box, textvariable=self.ocr_batch, width=6), 1, 0)
        w.labelled(box, "learning rate", ttk.Entry(box, textvariable=self.ocr_lr, width=8), 1, 2)
        w.labelled(box, "device",
                   ttk.Combobox(box, textvariable=self.ocr_device, values=["auto", "cuda", "cpu"],
                                state="readonly", width=6), 1, 4)

        ttk.Checkbutton(
            box, text="fine-tune the existing model instead of starting over",
            variable=self.ocr_resume,
        ).grid(row=2, column=0, columnspan=4, sticky="w", padx=8, pady=(4, 0))
        self._button(box, "Train reader", self.train_ocr).grid(row=2, column=5, sticky="e", padx=4, pady=(4, 0))

        self.ocr_state = ttk.Label(box, text="", foreground="#8b949e")
        self.ocr_state.grid(row=3, column=0, columnspan=6, sticky="w", padx=8, pady=(4, 0))

        # Real crops, labelled by position from photos that all carry one
        # marking: the reader fails on lighting it has not seen, and these are
        # the crops it is missing. See `icinspect.ocr.harvest`.
        ttk.Separator(box, orient="horizontal").grid(row=4, column=0, columnspan=6, sticky="ew", pady=6)
        ttk.Label(box, text="Add real crops from photos that all carry the same marking "
                            "(the crops the reader cannot read yet are the ones worth adding):",
                  foreground="#8b949e").grid(row=5, column=0, columnspan=6, sticky="w", padx=8)
        self.harvest_folder = tk.StringVar()
        self.harvest_marking = tk.StringVar()
        self.harvest_good_only = tk.BooleanVar(value=True)
        w.labelled(box, "photos", ttk.Entry(box, textvariable=self.harvest_folder, width=22), 6, 0)
        ttk.Button(box, text="...", width=3, command=self._pick_harvest_folder).grid(row=6, column=2, sticky="w")
        w.labelled(box, "marking", ttk.Entry(box, textvariable=self.harvest_marking, width=26), 6, 3)
        ttk.Label(box, text="lines with |, segments with spaces:  NE555P | 93M | DNJW26 G4",
                  foreground="#8b949e").grid(row=7, column=3, columnspan=3, sticky="w", padx=8)
        ttk.Checkbutton(box, text="only from parts that pass the defect check",
                        variable=self.harvest_good_only).grid(row=7, column=0, columnspan=3, sticky="w", padx=8)
        self._button(box, "Harvest crops", self.harvest_crops).grid(row=6, column=5, sticky="e", padx=4)

    def _build_patchcore(self) -> None:
        box = ttk.LabelFrame(self, text="3. Defect detector (PatchCore)", padding=6)
        box.grid(row=2, column=0, sticky="ew", pady=(8, 0))
        box.columnconfigure(3, weight=1)
        ttk.Label(
            box,
            text="Trains on good parts only, in one pass over anomaly/ic/train/good. "
                 "Seconds to a couple of minutes -- max_epochs: 1 is correct, not a typo.",
            foreground="#8b949e", wraplength=760, justify="left",
        ).grid(row=0, column=0, columnspan=4, sticky="w", pady=(0, 6))

        self.pc_no_test = tk.BooleanVar(value=False)
        ttk.Checkbutton(box, text="skip the evaluation pass (no test images yet)",
                        variable=self.pc_no_test).grid(row=1, column=0, sticky="w", padx=8)
        self._button(box, "Train detector", self.train_patchcore).grid(row=1, column=3, sticky="e", padx=4)

        self.pc_state = ttk.Label(box, text="", foreground="#8b949e")
        self.pc_state.grid(row=2, column=0, columnspan=4, sticky="w", padx=8, pady=(4, 0))

    def _build_actions(self) -> None:
        bar = ttk.Frame(self)
        bar.grid(row=3, column=0, sticky="ew", pady=(10, 0))
        bar.columnconfigure(3, weight=1)
        self.stop_button = ttk.Button(bar, text="Stop", command=self.worker.cancel, state="disabled")
        self.stop_button.grid(row=0, column=0)
        self.progress = ttk.Progressbar(bar, mode="indeterminate", length=260)
        self.progress.grid(row=0, column=1, padx=8)
        self.percent = ttk.Label(bar, text="", width=16)
        self.percent.grid(row=0, column=2, sticky="w")
        self.status = ttk.Label(bar, text="Ready", foreground="#8b949e")
        self.status.grid(row=0, column=3, sticky="w")
        ttk.Button(bar, text="Open results folder",
                   command=lambda: w.reveal(self.inspector.paths().results_dir)).grid(row=0, column=4)

    def _button(self, master: tk.Misc, text: str, command) -> ttk.Button:
        button = ttk.Button(master, text=text, command=command, width=16)
        self._buttons.append(button)
        return button

    # -- jobs --------------------------------------------------------------

    def _pick_labels(self) -> None:
        from tkinter import filedialog

        chosen = filedialog.askopenfilename(
            parent=self, title="Choose a label file",
            initialdir=str(self.inspector.paths().ocr_root),
            filetypes=[("Label files", "*.txt"), ("All files", "*.*")],
        )
        if chosen:
            self.ocr_labels.set(chosen)

    def _pick_harvest_folder(self) -> None:
        chosen = w.ask_folder(self, "Choose a folder of photos of one marking",
                              self.harvest_folder.get() or str(self.inspector.paths().raw_root))
        if chosen:
            self.harvest_folder.set(chosen)

    def harvest_crops(self) -> None:
        """Segment the photos, label the crops by position, append to the label file.

        Runs the same inspection the Inspect tab does, so the crops are the
        window's own segments and -- with the box ticked -- only from parts
        PatchCore passed: a scratched-through `NE555P` labelled `NE555P` would
        teach the reader the wrong thing.
        """
        from icinspect.ocr.harvest import harvest, parse_expected

        if self.worker.busy:
            messagebox.showinfo("Busy", "Another job is already running.", parent=self)
            return
        folder = Path(self.harvest_folder.get().strip().strip('"'))
        images = list_images(folder) if folder.exists() else []
        if not images:
            messagebox.showwarning("No photos", "Pick a folder of photos first.", parent=self)
            return
        try:
            expected = parse_expected(self.harvest_marking.get())
        except ValueError as exc:
            messagebox.showwarning("No marking", str(exc), parent=self)
            return
        if self.inspector.ocr_checkpoint() is None:
            messagebox.showwarning("No reader yet",
                                   "Train a reader on synthetic crops first; the harvest needs "
                                   "its reads to tell a logo from text.", parent=self)
            return
        good_only = self.harvest_good_only.get() and self.inspector.anomaly_checkpoint() is not None
        paths = self.inspector.paths()
        labels = Path(self.ocr_labels.get().strip() or "train.txt")
        if not labels.is_absolute():
            labels = paths.ocr_root / labels
        out_dir = paths.ocr_root / "images" / "harvest"
        inspect_tab = getattr(self.app, "inspect_tab", None)
        opts = inspect_tab.options() if inspect_tab is not None else Options()
        opts.check_ocr = True
        opts.check_anomaly = good_only
        opts.expect = ""
        opts.save_crops = False
        inspector = self.inspector
        title = "Harvesting crops"
        self.app.log("")
        self.app.log(f"=== {title}: {len(images)} photo(s), marking "
                     f"{' | '.join(' '.join(line) for line in expected)} ===")
        self.app.show_log()

        def task(prog: Progress) -> tuple[int, int, int]:
            results = inspector.inspect(images, opts, prog)
            kept = [r for r in results if not (good_only and r.verdict == "FAIL")]
            rejected = len(results) - len(kept)
            if rejected:
                prog.log(f"[harvest] {rejected} photo(s) left out: failed the defect check")
            rows = [row for r in kept for row in _segment_rows(r)]
            prog.status("Writing crops...")
            written, skipped = harvest([r.path for r in kept], expected, out_dir, labels,
                                       predict=lambda _images: rows, log=prog.log)
            return written, skipped, rejected

        def done(counts: tuple[int, int, int]) -> None:
            written, skipped, rejected = counts
            self._settle(f"{title} - {written} crop(s)")
            self.app.log(f"[harvest] {written} crop(s) -> {out_dir}", "ok")
            self.app.log(f"[harvest] labels appended to {labels}; {skipped} photo(s) skipped, "
                         f"{rejected} rejected as defective")
            if written:
                self.app.log("[harvest] next: tick fine-tune and press Train reader")

        started = self.worker.start(
            task,
            on_log=self.app.log,
            on_status=lambda text: self.status.configure(text=text),
            on_done=done,
            on_error=self._failed,
            on_cancelled=lambda _p: self._settle("Stopped"),
        )
        if started:
            self._busy(True, title)

    def generate_synth(self) -> None:
        args = ["--count", str(w.as_int(self.synth_count.get(), 20000)),
                "--val-ratio", str(w.as_float(self.synth_val.get(), 0.1) or 0.1)]
        if self.synth_split.get():
            args.append("--split")
        script = PROJECT_ROOT / "scripts" / "gen_synth_ocr.py"
        self._launch("Generating synthetic crops", python_script(script, *args))

    def train_ocr(self) -> None:
        labels = self.ocr_labels.get().strip()
        args: list[str] = []
        if labels:
            args += ["--labels", labels]
        for flag, var, cast in (
            ("--epochs", self.ocr_epochs, int), ("--batch-size", self.ocr_batch, int),
            ("--lr", self.ocr_lr, float),
        ):
            raw = var.get().strip()
            if raw:
                args += [flag, str(cast(float(raw)))]
        if self.ocr_device.get() != "auto":
            args += ["--device", self.ocr_device.get()]
        if self.ocr_resume.get():
            ckpt = self.inspector.ocr_checkpoint()
            if ckpt is None:
                messagebox.showwarning("Nothing to fine-tune",
                                       "There is no trained reader yet -- untick fine-tune.", parent=self)
                return
            args += ["--resume", str(ckpt)]
        self._launch("Training the marking reader", python_module("icinspect.ocr.train", *args))

    def train_patchcore(self) -> None:
        good = self.inspector.paths().anomaly_root / "train" / "good"
        count = len(list(good.glob("*"))) if good.exists() else 0
        if count < 10 and not messagebox.askyesno(
            "Few training images",
            f"{count} image(s) in {good}.\n\nPatchCore wants 30-300 good parts. Train anyway?",
            parent=self,
        ):
            return
        args = ["--no-test"] if self.pc_no_test.get() else []
        self._launch("Training the defect detector",
                     python_module("icinspect.anomaly.train_patchcore", *args))

    def _launch(self, title: str, command: list[str]) -> None:
        if self.worker.busy:
            messagebox.showinfo("Busy", "Another job is already running.", parent=self)
            return
        self.app.log("")
        self.app.log(f"=== {title} ===")
        self.app.show_log()

        def task(prog: Progress) -> int:
            return run_command(prog, command, cwd=PROJECT_ROOT)

        started = self.worker.start(
            task,
            on_log=self.app.log,
            on_status=self._status_text,
            on_progress=self._advance,
            on_done=lambda code: self._finished(title, code),
            on_error=self._failed,
            on_cancelled=lambda _p: self._settle("Stopped"),
        )
        if started:
            self._busy(True, title)

    def _busy(self, running: bool, title: str = "") -> None:
        for button in self._buttons:
            button.configure(state="disabled" if running else "normal")
        self.stop_button.configure(state="normal" if running else "disabled")
        self._title = title
        self._started = time.time()
        self._stage_started = self._started
        self._stage = ""
        self._last_step: tuple[int, int] | None = None
        if running:
            # Pulsing until the job says how far along it is; a bar with a
            # number the moment it does.
            self.progress.configure(mode="indeterminate")
            self.progress.start(12)
            self.status.configure(text=f"{title}...")
            self.percent.configure(text="")
        else:
            self.progress.stop()
            self.progress.configure(mode="determinate", value=0, maximum=1)
            self.percent.configure(text="")

    def _advance(self, done_total: tuple[int, int]) -> None:
        """A step from the job: switch the bar to a real one and move it."""
        done, total = done_total
        if total <= 0:
            return
        if self._last_step is None or done < self._last_step[0] or total != self._last_step[1]:
            # A new stage (lightning goes Fitting -> Validation -> Testing, each
            # with its own counter), so the ETA starts over from here.
            self._stage_started = time.time()
            self._stage_offset = done
        self._last_step = (done, total)
        if str(self.progress.cget("mode")) != "determinate":
            self.progress.stop()
            self.progress.configure(mode="determinate")
        self.progress.configure(maximum=total, value=done)
        advanced = done - getattr(self, "_stage_offset", 0)
        remaining = ""
        if advanced > 0 and done < total:
            rate = (time.time() - self._stage_started) / advanced
            remaining = f"   {hms(rate * (total - done))} left"
        self.percent.configure(text=f"{done / total * 100:.0f}%{remaining}")

    def _status_text(self, text: str) -> None:
        """The job's own words, kept short: the bar and the percentage do the rest.

        A child's `eta` is for whoever runs it from PowerShell; here the label
        beside the bar already says how long is left, so it is not repeated.
        """
        text = re.sub(r"\s+eta\s+\S+\s*$", "", text)
        self.status.configure(text=text if len(text) < 90 else text[:87] + "...")

    def _settle(self, message: str) -> None:
        self._busy(False)
        self.status.configure(text=message)
        # New weights on disk: drop the loaded ones so Inspect picks them up.
        self.inspector.unload()
        self.app.models_changed()

    def _finished(self, title: str, code: int) -> None:
        if code == 0:
            self._settle(f"{title} - done")
            self.app.log(f"[done] {title}", "ok")
        else:
            self._settle(f"{title} - failed (exit {code})")
            self.app.log(f"[failed] {title}: exit code {code}", "error")
            messagebox.showerror(
                "Training failed",
                f"{title} exited with code {code}.\nThe log at the bottom of the window has the details.",
                parent=self,
            )

    def _failed(self, payload) -> None:
        exc, trace = payload
        self._settle("Failed")
        self.app.log(trace, "error")
        messagebox.showerror("Could not run", str(exc), parent=self)

    # -- state -------------------------------------------------------------

    def refresh_state(self) -> None:
        ckpt = self.inspector.ocr_checkpoint()
        self.ocr_state.configure(
            text=f"current model: {ckpt}" if ckpt else "no reader trained yet"
        )
        anomaly = self.inspector.anomaly_checkpoint()
        good = self.inspector.paths().anomaly_root / "train" / "good"
        count = len([p for p in good.glob("*") if p.is_file()]) if good.exists() else 0
        self.pc_state.configure(
            text=(f"current model: {anomaly}" if anomaly else "no detector trained yet")
                 + f"   |   {count} good training image(s)"
        )
