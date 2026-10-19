"""Small shared pieces: the log pane, file pickers, and number parsing."""

from __future__ import annotations

import os
import subprocess
import sys
import time
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


class ProgressMeter(ttk.Frame):
    """A progress bar with the percentage and the time left beside it.

    It pulses until the job first says how far along it is -- loading a model
    has no count -- and becomes a real bar the moment it does. The time left
    is the rate so far applied to what remains. A count that goes backwards or
    changes its total is a new stage (lightning: fit, then validation, then
    test), so the rate is measured afresh from there.
    """

    def __init__(self, master: tk.Misc, length: int = 260, **kwargs) -> None:
        super().__init__(master, **kwargs)
        self.bar = ttk.Progressbar(self, mode="determinate", length=length, maximum=1, value=0)
        self.bar.grid(row=0, column=0)
        self.label = ttk.Label(self, text="", width=18)
        self.label.grid(row=0, column=1, sticky="w", padx=(8, 0))
        self._started = time.monotonic()
        self._stage_started = self._started
        self._stage_offset = 0
        self._last: tuple[int, int] | None = None

    def start(self) -> None:
        """A job began: pulse until its first step."""
        self._started = time.monotonic()
        self._stage_started = self._started
        self._stage_offset = 0
        self._last = None
        self.bar.configure(mode="indeterminate")
        self.bar.start(12)
        self.label.configure(text="")

    def advance(self, done_total: tuple[int, int]) -> None:
        """A step from the job: `(done, total)`."""
        done, total = done_total
        if total <= 0:
            return
        done = max(0, min(done, total))
        now = time.monotonic()
        if self._last is None or done < self._last[0] or total != self._last[1]:
            self._stage_started = now
            self._stage_offset = done
        self._last = (done, total)
        if str(self.bar.cget("mode")) != "determinate":
            self.bar.stop()
            self.bar.configure(mode="determinate")
        self.bar.configure(maximum=total, value=done)
        advanced = done - self._stage_offset
        remaining = ""
        if advanced > 0 and done < total:
            rate = (now - self._stage_started) / advanced
            remaining = f"   {_hms(rate * (total - done))} left"
        self.label.configure(text=f"{done / total * 100:.0f}%{remaining}")

    def stop(self, finished: bool = False) -> None:
        """The job ended. A finished one shows a full bar and how long it took."""
        self.bar.stop()
        if finished:
            self.bar.configure(mode="determinate", maximum=1, value=1)
            self.label.configure(text=f"100%   in {_hms(time.monotonic() - self._started)}")
        else:
            self.bar.configure(mode="determinate", maximum=1, value=0)
            self.label.configure(text="")

    @property
    def fraction(self) -> float:
        """How far the bar is, 0..1 (0 while it pulses)."""
        if self._last is None:
            return 0.0
        done, total = self._last
        return done / total


def _hms(seconds: float) -> str:
    from icinspect.progress import hms

    return hms(seconds)
