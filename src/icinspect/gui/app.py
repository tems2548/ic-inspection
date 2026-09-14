"""The IC Inspection window.

    python -m icinspect.gui

Four tabs -- Inspect, Stats, Train, Setup -- over one shared worker thread and
one shared log, so only one heavy job runs at a time and its output always lands
in the same place.
"""

from __future__ import annotations

import sys
import tkinter as tk
from pathlib import Path
from tkinter import messagebox, ttk

from icinspect.config import CONFIG_DIR, PROJECT_ROOT
from icinspect.gui import widgets as w
from icinspect.gui.runner import Worker
from icinspect.gui.service import Inspector

QUICK_START = """\
1. Setup tab - point at your dataset folder and press "Create the folders",
   then "Check this machine" to confirm torch sees the GPU.

2. Put photos in place:
   - 30-300 good parts in  anomaly\\ic\\train\\good
   - defective ones in     anomaly\\ic\\test\\defect

3. Train tab - "Generate crops" (synthetic markings), then "Train reader",
   then "Train detector". The first two need no labelling at all.

4. Inspect tab - choose a photo or a folder and press Inspect. Green boxes are
   what the reader read; grey dashed boxes are logos and marks it skipped, and
   the heat wash is where PatchCore saw a defect. A wrong read is usually a
   wrong cut: look at the boxes before the model.

5. Stats tab - the distribution of this run's scores, the reader's training
   curves, and every hyperparameter both models were trained with.
"""


class App(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("IC Inspection")
        self.geometry("1360x900")
        self.minsize(980, 640)

        self.inspector = Inspector()
        self.worker = Worker(self)

        self._style()
        self._menu()

        outer = ttk.PanedWindow(self, orient="vertical")
        outer.pack(fill="both", expand=True, padx=6, pady=(6, 0))

        notebook = ttk.Notebook(outer)
        outer.add(notebook, weight=4)

        # Imported here so a failure in one tab still leaves a usable window.
        from icinspect.gui.inspect_tab import InspectTab
        from icinspect.gui.setup_tab import SetupTab
        from icinspect.gui.stats_tab import StatsTab
        from icinspect.gui.train_tab import TrainTab

        self.inspect_tab = InspectTab(notebook, self)
        self.stats_tab = StatsTab(notebook, self)
        self.train_tab = TrainTab(notebook, self)
        self.setup_tab = SetupTab(notebook, self)
        notebook.add(self.inspect_tab, text="  Inspect  ")
        notebook.add(self.stats_tab, text="  Stats  ")
        notebook.add(self.train_tab, text="  Train  ")
        notebook.add(self.setup_tab, text="  Setup  ")
        notebook.bind("<<NotebookTabChanged>>", self._on_tab_changed)
        self.notebook = notebook

        log_frame = ttk.Frame(outer)
        log_frame.rowconfigure(1, weight=1)
        log_frame.columnconfigure(0, weight=1)
        header = ttk.Frame(log_frame)
        header.grid(row=0, column=0, sticky="ew")
        ttk.Label(header, text="Output").pack(side="left")
        ttk.Button(header, text="Clear", width=8,
                   command=lambda: self.log_pane.clear()).pack(side="right")
        self.log_pane = w.LogPane(log_frame, height=10)
        self.log_pane.grid(row=1, column=0, sticky="nsew", pady=(2, 0))
        outer.add(log_frame, weight=1)
        self._outer = outer
        self._log_frame = log_frame

        status = ttk.Frame(self)
        status.pack(fill="x", padx=8, pady=4)
        self.status = ttk.Label(status, text=f"Project: {PROJECT_ROOT}", foreground="#8b949e")
        self.status.pack(side="left")
        self.dataset_label = ttk.Label(status, text="", foreground="#8b949e")
        self.dataset_label.pack(side="right")

        self.protocol("WM_DELETE_WINDOW", self._on_close)
        self.bind("<F5>", lambda _e: self.inspect_tab.run())
        self.log("IC Inspection - ready. Setup tab first if this is a new machine.")
        self.models_changed()

    # -- chrome ------------------------------------------------------------

    def _style(self) -> None:
        style = ttk.Style(self)
        if sys.platform == "win32" and "vista" in style.theme_names():
            style.theme_use("vista")
        base = style.lookup("TButton", "font") or "TkDefaultFont"
        style.configure("Accent.TButton", font=(base if isinstance(base, str) else "TkDefaultFont", 10, "bold"))
        style.configure("Treeview", rowheight=22)

    def _menu(self) -> None:
        bar = tk.Menu(self)

        file_menu = tk.Menu(bar, tearoff=False)
        file_menu.add_command(label="Inspect a photo...", command=self._open_image)
        file_menu.add_command(label="Inspect a folder...", command=self._open_folder)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self._on_close)
        bar.add_cascade(label="File", menu=file_menu)

        tools = tk.Menu(bar, tearoff=False)
        tools.add_command(label="Open results folder",
                          command=lambda: w.reveal(self.inspector.paths().results_dir))
        tools.add_command(label="Open dataset folder",
                          command=lambda: w.reveal(self.inspector.paths().dataset_root))
        tools.add_command(label="Open configs folder", command=lambda: w.reveal(CONFIG_DIR))
        tools.add_separator()
        tools.add_command(label="Reload configs and models", command=self._reload)
        tools.add_command(label="Relearn the scored region", command=self._relearn_region)
        bar.add_cascade(label="Tools", menu=tools)

        help_menu = tk.Menu(bar, tearoff=False)
        help_menu.add_command(label="Quick start", command=self._quick_start)
        help_menu.add_command(label="Open README", command=self._open_readme)
        bar.add_cascade(label="Help", menu=help_menu)

        self.config(menu=bar)

    # -- shared services ---------------------------------------------------

    def log(self, text: str, tag: str | None = None) -> None:
        for line in str(text).splitlines() or [""]:
            self.log_pane.write(line, tag)

    def show_log(self) -> None:
        """Make sure the output pane is not collapsed to nothing."""
        try:
            if self._outer.sashpos(0) > self.winfo_height() - 120:
                self._outer.sashpos(0, max(self.winfo_height() - 260, 200))
        except tk.TclError:
            pass

    def _on_tab_changed(self, event) -> None:
        """Stats builds its matplotlib canvas the first time it is looked at."""
        try:
            current = event.widget.nametowidget(event.widget.select())
        except (tk.TclError, KeyError):
            return
        if current is getattr(self, "stats_tab", None):
            self.stats_tab.on_shown()

    def models_changed(self) -> None:
        """A checkpoint or the dataset root moved: refresh whatever shows that."""
        for tab in ("inspect_tab", "stats_tab", "train_tab", "setup_tab"):
            widget = getattr(self, tab, None)
            if widget is not None:
                widget.refresh_state()
        paths = self.inspector.paths()
        self.dataset_label.configure(text=f"Dataset: {paths.dataset_root}")

    # -- menu actions ------------------------------------------------------

    def _open_image(self) -> None:
        chosen = w.ask_image(self, str(self.inspector.paths().raw_root))
        if chosen:
            self.inspect_tab.source.set(chosen)
            self.notebook.select(self.inspect_tab)

    def _open_folder(self) -> None:
        chosen = w.ask_folder(self, "Choose a folder of photos", str(self.inspector.paths().raw_root))
        if chosen:
            self.inspect_tab.source.set(chosen)
            self.notebook.select(self.inspect_tab)

    def _reload(self) -> None:
        self.inspector.unload()
        self.models_changed()
        self.log("[gui] configs re-read, models unloaded")

    def _relearn_region(self) -> None:
        """Measure the part window again -- after moving the camera or the fixture."""
        self.inspector.forget_region()
        self.log("[part] scored region forgotten; it is measured again on the next run")
        self.models_changed()

    def _quick_start(self) -> None:
        messagebox.showinfo("Quick start", QUICK_START, parent=self)

    def _open_readme(self) -> None:
        readme = PROJECT_ROOT / "README.md"
        if readme.exists():
            w.reveal(readme)
            try:
                import os

                os.startfile(readme)  # noqa: S606 - user asked for the README
            except (AttributeError, OSError):
                pass

    def _on_close(self) -> None:
        if self.worker.busy and not messagebox.askyesno(
            "A job is running",
            "Something is still running. Close anyway and stop it?",
            parent=self,
        ):
            return
        self.worker.cancel()
        self.destroy()


def main(argv: list[str] | None = None) -> int:
    """Entry point for `python -m icinspect.gui` and the ic-inspect-gui script."""
    args = list(sys.argv[1:] if argv is None else argv)
    app = App()
    if args:
        candidate = Path(args[0])
        if candidate.exists():
            app.inspect_tab.source.set(str(candidate))
    app.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
