"""The Setup tab: where the dataset is, what is in it, and does the machine work.

The three questions that come first on a fresh machine, and the three that come
back whenever something behaves oddly -- almost always because the images are
not where the config says they are.
"""

from __future__ import annotations

import os
import re
import tkinter as tk
from datetime import datetime
from pathlib import Path
from tkinter import messagebox, ttk

from icinspect.config import CONFIG_DIR, PROJECT_ROOT
from icinspect.gui import widgets as w
from icinspect.gui.runner import Progress, python_script, run_command
from icinspect.ocr.dataset import list_images

DATASET_ROOT_LINE = re.compile(r'^(\s*dataset_root\s*:\s*).*$', re.MULTILINE)


class SetupTab(ttk.Frame):
    def __init__(self, master: tk.Misc, app) -> None:
        super().__init__(master, padding=8)
        self.app = app
        self.worker = app.worker
        self.inspector = app.inspector

        self.columnconfigure(0, weight=1)
        self.rowconfigure(2, weight=1)
        self._build_root()
        self._build_actions()
        self._build_summary()
        self.refresh_state()

    # -- sections ----------------------------------------------------------

    def _build_root(self) -> None:
        box = ttk.LabelFrame(self, text="Dataset location", padding=6)
        box.grid(row=0, column=0, sticky="ew")
        box.columnconfigure(0, weight=1)

        self.root_var = tk.StringVar()
        ttk.Entry(box, textvariable=self.root_var).grid(row=0, column=0, sticky="ew", padx=(0, 6))
        ttk.Button(box, text="Browse...", width=10, command=self._pick_root).grid(row=0, column=1)
        ttk.Button(box, text="Use now", width=10, command=self._apply_root).grid(row=0, column=2, padx=4)
        ttk.Button(box, text="Save as default", width=14, command=self._save_root).grid(row=0, column=3)
        ttk.Label(
            box,
            text="Images live outside the project so they never end up in version control. "
                 "\"Use now\" applies until you close the window; \"Save as default\" writes "
                 "configs/paths.yaml.",
            foreground="#8b949e", wraplength=760, justify="left",
        ).grid(row=1, column=0, columnspan=4, sticky="w", pady=(4, 0))

    def _build_actions(self) -> None:
        bar = ttk.Frame(self)
        bar.grid(row=1, column=0, sticky="ew", pady=8)
        bar.columnconfigure(3, weight=1)
        self.create_button = ttk.Button(bar, text="Create the folders", command=self.create_folders)
        self.create_button.grid(row=0, column=0)
        self.check_button = ttk.Button(bar, text="Check this machine", command=self.check_env)
        self.check_button.grid(row=0, column=1, padx=6)
        ttk.Button(bar, text="Refresh counts", command=self.refresh_state).grid(row=0, column=2)
        self.status = ttk.Label(bar, text="", foreground="#8b949e")
        self.status.grid(row=0, column=3, sticky="w", padx=8)
        ttk.Button(bar, text="Open dataset folder", command=self._reveal_root).grid(row=0, column=4)

    def _build_summary(self) -> None:
        box = ttk.LabelFrame(self, text="What is on disk", padding=6)
        box.grid(row=2, column=0, sticky="nsew")
        box.rowconfigure(0, weight=1)
        box.columnconfigure(0, weight=1)

        tree = ttk.Treeview(box, columns=("count", "path"), show="tree headings", selectmode="browse")
        tree.heading("#0", text="what")
        tree.heading("count", text="count")
        tree.heading("path", text="where")
        tree.column("#0", width=220, stretch=False)
        tree.column("count", width=80, anchor="e", stretch=False)
        tree.column("path", width=520)
        tree.tag_configure("missing", foreground="#ff7b72")
        tree.tag_configure("empty", foreground="#d29922")
        tree.bind("<Double-Button-1>", self._open_row)
        scroll = ttk.Scrollbar(box, orient="vertical", command=tree.yview)
        tree.configure(yscrollcommand=scroll.set)
        tree.grid(row=0, column=0, sticky="nsew")
        scroll.grid(row=0, column=1, sticky="ns")
        self.tree = tree

    # -- dataset root ------------------------------------------------------

    def _pick_root(self) -> None:
        chosen = w.ask_folder(self, "Choose the dataset root", self.root_var.get())
        if chosen:
            self.root_var.set(chosen)

    def _apply_root(self) -> None:
        root = self.root_var.get().strip().strip('"')
        if not root:
            return
        if not Path(root).exists() and not messagebox.askyesno(
            "Folder does not exist",
            f"{root}\ndoes not exist yet. Use it anyway? "
            "\"Create the folders\" will build the tree there.",
            parent=self,
        ):
            return
        # Child processes inherit this, so training and the GUI agree on where
        # the data is without either of them editing the config file.
        os.environ["IC_DATASET_ROOT"] = root
        # Different dataset, different fixture: the scored region has to be
        # measured again rather than carried over from the last one.
        self.inspector.forget_region()
        self.inspector.unload()
        self.app.models_changed()
        self.status.configure(text="Using this dataset root for now")
        self.app.log(f"[setup] IC_DATASET_ROOT = {root}")

    def _save_root(self) -> None:
        root = self.root_var.get().strip().strip('"').replace("\\", "/")
        if not root:
            return
        paths_file = CONFIG_DIR / "paths.yaml"
        text = paths_file.read_text(encoding="utf-8")
        # Rewrite just the one value: the file's comments explain the layout and
        # dumping the parsed YAML back would throw them away.
        new_text, count = DATASET_ROOT_LINE.subn(lambda m: f'{m.group(1)}"{root}"', text, count=1)
        if not count:
            messagebox.showerror("Could not save",
                                 f"No dataset_root line found in {paths_file}", parent=self)
            return
        paths_file.write_text(new_text, encoding="utf-8")
        os.environ.pop("IC_DATASET_ROOT", None)   # the file is the answer again
        self.inspector.forget_region()
        self.inspector.unload()
        self.app.models_changed()
        self.status.configure(text=f"Saved to {paths_file.name}")
        self.app.log(f"[setup] dataset_root saved to {paths_file}", "ok")

    def _reveal_root(self) -> None:
        w.reveal(Path(self.root_var.get().strip() or str(self.inspector.paths().dataset_root)))

    # -- jobs --------------------------------------------------------------

    def create_folders(self) -> None:
        root = self.root_var.get().strip().strip('"')
        args = ["--root", root] if root else []
        self._launch("Creating the dataset folders",
                     python_script(PROJECT_ROOT / "scripts" / "init_dataset.py", *args))

    def check_env(self) -> None:
        self._launch("Checking this machine",
                     python_script(PROJECT_ROOT / "scripts" / "check_env.py"))

    def _launch(self, title: str, command: list[str]) -> None:
        if self.worker.busy:
            messagebox.showinfo("Busy", "Another job is already running.", parent=self)
            return
        self.app.log("")
        self.app.log(f"=== {title} ===")
        self.app.show_log()

        def task(prog: Progress) -> int:
            return run_command(prog, command, cwd=PROJECT_ROOT)

        if self.worker.start(
            task,
            on_log=self.app.log,
            on_done=lambda code: self._finished(title, code),
            on_error=self._failed,
            on_cancelled=lambda _p: self._enable(True),
        ):
            self._enable(False)
            self.status.configure(text=f"{title}...")

    def _enable(self, on: bool) -> None:
        state = "normal" if on else "disabled"
        self.create_button.configure(state=state)
        self.check_button.configure(state=state)

    def _finished(self, title: str, code: int) -> None:
        self._enable(True)
        self.status.configure(text=f"{title} - {'done' if code == 0 else f'exit {code}'}")
        self.refresh_state()
        self.app.models_changed()

    def _failed(self, payload) -> None:
        exc, trace = payload
        self._enable(True)
        self.status.configure(text="Failed")
        self.app.log(trace, "error")
        messagebox.showerror("Could not run", str(exc), parent=self)

    # -- state -------------------------------------------------------------

    def refresh_state(self) -> None:
        paths = self.inspector.paths()
        if not self.root_var.get().strip():
            self.root_var.set(str(paths.dataset_root))

        rows: list[tuple[str, Path, bool]] = [
            ("raw captures", paths.raw_root, True),
            ("anomaly / train / good", paths.anomaly_root / "train" / "good", True),
            ("anomaly / test / good", paths.anomaly_root / "test" / "good", True),
            ("anomaly / test / defect", paths.anomaly_root / "test" / "defect", True),
            ("anomaly / ground_truth", paths.anomaly_root / "ground_truth" / "defect", True),
            ("ocr / images", paths.ocr_images, True),
            ("ocr / synth", paths.ocr_synth, True),
        ]
        self.tree.delete(*self.tree.get_children())
        for name, path, _counts in rows:
            if not path.exists():
                self.tree.insert("", "end", text=name, values=("-", f"{path}  (missing)"),
                                 tags=("missing",))
                continue
            found = len(list_images(path))
            tags = () if found else ("empty",)
            self.tree.insert("", "end", text=name, values=(found, str(path)), tags=tags)

        for name, path in (
            ("labels / train.txt", paths.labels_file("train")),
            ("labels / val.txt", paths.labels_file("val")),
            ("labels / test.txt", paths.labels_file("test")),
        ):
            if path.exists():
                lines = sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())
                self.tree.insert("", "end", text=name, values=(lines, str(path)))
            else:
                self.tree.insert("", "end", text=name, values=("-", f"{path}  (missing)"),
                                 tags=("missing",))

        for name, ckpt in (
            ("model / marking reader", self.inspector.ocr_checkpoint()),
            ("model / defect detector", self.inspector.anomaly_checkpoint()),
        ):
            if ckpt is None:
                self.tree.insert("", "end", text=name, values=("-", "not trained yet"), tags=("empty",))
            else:
                when = datetime.fromtimestamp(ckpt.stat().st_mtime).strftime("%Y-%m-%d %H:%M")
                self.tree.insert("", "end", text=name, values=("", f"{ckpt}   ({when})"))

    def _open_row(self, _event: tk.Event) -> None:
        selection = self.tree.selection()
        if not selection:
            return
        where = str(self.tree.item(selection[0], "values")[1]).split("  (")[0]
        candidate = Path(where)
        if candidate.exists():
            w.reveal(candidate)
