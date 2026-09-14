"""The Train tab: generate data and train either model without typing a command.

Each button shells out to the same module the README documents, so a run
started here and a run started in PowerShell are the same run -- and the exact
command is echoed into the log, which is how you learn the CLI from the GUI.
"""

from __future__ import annotations

import tkinter as tk
from tkinter import messagebox, ttk

from icinspect.config import PROJECT_ROOT
from icinspect.gui import widgets as w
from icinspect.gui.runner import Progress, python_module, python_script, run_command


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
        bar.columnconfigure(2, weight=1)
        self.stop_button = ttk.Button(bar, text="Stop", command=self.worker.cancel, state="disabled")
        self.stop_button.grid(row=0, column=0)
        self.progress = ttk.Progressbar(bar, mode="indeterminate", length=160)
        self.progress.grid(row=0, column=1, padx=8)
        self.status = ttk.Label(bar, text="Ready", foreground="#8b949e")
        self.status.grid(row=0, column=2, sticky="w")
        ttk.Button(bar, text="Open results folder",
                   command=lambda: w.reveal(self.inspector.paths().results_dir)).grid(row=0, column=3)

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
            on_status=lambda text: self.status.configure(text=text),
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
        if running:
            self.progress.start(12)
            self.status.configure(text=f"{title}...")
        else:
            self.progress.stop()

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
