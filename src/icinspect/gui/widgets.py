"""Small shared pieces: the log pane, file pickers, and number parsing."""

from __future__ import annotations

import os
import subprocess
import sys
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, ttk

IMAGE_TYPES = [
    ("Images", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff"),
    ("All files", "*.*"),
]


class LogPane(ttk.Frame):
    """Read-only output area. Autoscrolls unless the user has scrolled up."""

    MAX_LINES = 4000

    def __init__(self, master: tk.Misc, height: int = 12, **kwargs) -> None:
        super().__init__(master, **kwargs)
        self.text = tk.Text(
            self, height=height, wrap="none", state="disabled",
            background="#0f1116", foreground="#c9d1d9", insertbackground="#c9d1d9",
            font=("Consolas", 9), relief="flat", padx=6, pady=4,
        )
        yscroll = ttk.Scrollbar(self, orient="vertical", command=self.text.yview)
        xscroll = ttk.Scrollbar(self, orient="horizontal", command=self.text.xview)
        self.text.configure(yscrollcommand=yscroll.set, xscrollcommand=xscroll.set)
        self.text.grid(row=0, column=0, sticky="nsew")
        yscroll.grid(row=0, column=1, sticky="ns")
        xscroll.grid(row=1, column=0, sticky="ew")
        self.rowconfigure(0, weight=1)
        self.columnconfigure(0, weight=1)
        self.text.tag_configure("error", foreground="#ff7b72")
        self.text.tag_configure("ok", foreground="#3fb950")

    def write(self, line: str, tag: str | None = None) -> None:
        at_bottom = self.text.yview()[1] > 0.999
        self.text.configure(state="normal")
        self.text.insert("end", line.rstrip("\n") + "\n", tag or ())
        # Keep the buffer bounded: a training run prints tens of thousands of
        # lines and Tk slows to a crawl long before that matters to anyone.
        excess = int(self.text.index("end-1c").split(".")[0]) - self.MAX_LINES
        if excess > 0:
            self.text.delete("1.0", f"{excess + 1}.0")
        self.text.configure(state="disabled")
        if at_bottom:
            self.text.see("end")

    def clear(self) -> None:
        self.text.configure(state="normal")
        self.text.delete("1.0", "end")
        self.text.configure(state="disabled")


def ask_image(parent: tk.Misc, initial: str | None = None) -> str:
    return filedialog.askopenfilename(
        parent=parent, title="Choose an IC photo", filetypes=IMAGE_TYPES,
        initialdir=initial or None,
    )


def ask_folder(parent: tk.Misc, title: str = "Choose a folder", initial: str | None = None) -> str:
    return filedialog.askdirectory(parent=parent, title=title, initialdir=initial or None)


def ask_save_csv(parent: tk.Misc, initial_file: str = "inspection_report.csv") -> str:
    return filedialog.asksaveasfilename(
        parent=parent, title="Save report", defaultextension=".csv",
        initialfile=initial_file, filetypes=[("CSV", "*.csv"), ("All files", "*.*")],
    )


def reveal(path: Path) -> None:
    """Open a folder in the file manager (or the folder holding a file)."""
    target = path if path.is_dir() else path.parent
    target.mkdir(parents=True, exist_ok=True)
    if os.name == "nt":
        os.startfile(target)  # noqa: S606 - a user-initiated "open folder" action
    elif sys.platform == "darwin":
        subprocess.Popen(["open", str(target)])
    else:
        subprocess.Popen(["xdg-open", str(target)])


def as_float(value: str, default: float | None = None) -> float | None:
    """Parse a number typed into a box; blank or nonsense falls back to `default`."""
    try:
        return float(str(value).strip())
    except (TypeError, ValueError):
        return default


def as_int(value: str, default: int = 0) -> int:
    try:
        return int(float(str(value).strip()))
    except (TypeError, ValueError):
        return default


def labelled(master: tk.Misc, text: str, widget: tk.Widget, row: int, column: int, **grid) -> None:
    """Place `widget` after a label, in one grid cell pair."""
    ttk.Label(master, text=text).grid(row=row, column=column, sticky="e", padx=(8, 4), pady=2)
    widget.grid(row=row, column=column + 1, sticky="w", pady=2, **grid)


class Collapsible(ttk.Frame):
    """A section that folds away, for the knobs most runs never touch."""

    def __init__(self, master: tk.Misc, text: str, expanded: bool = False, **kwargs) -> None:
        super().__init__(master, **kwargs)
        self._expanded = tk.BooleanVar(value=expanded)
        self._button = ttk.Checkbutton(
            self, style="Toolbutton", variable=self._expanded,
            text=text, command=self._toggle,
        )
        self._button.grid(row=0, column=0, sticky="w")
        self.body = ttk.Frame(self)
        self.columnconfigure(0, weight=1)
        if expanded:
            self.body.grid(row=1, column=0, sticky="ew", padx=(12, 0))

    def _toggle(self) -> None:
        if self._expanded.get():
            self.body.grid(row=1, column=0, sticky="ew", padx=(12, 0))
        else:
            self.body.grid_forget()
