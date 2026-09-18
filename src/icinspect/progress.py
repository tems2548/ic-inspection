"""Progress lines a training script prints and the window turns into a bar.

The scripts run as child processes of the window (see `gui.runner`), so the
only channel back is stdout. One line shape carries a step:

    [progress] 1234/11300 epoch 3/20  batch 104/565  loss 0.0421

`done/total` drives the bar; whatever follows is the status text. Lines are
rate-limited here so a fast loop does not flood the log, and a run started
from PowerShell sees the same lines -- a bar there would be nicer, but this
is honest about what the window shows.
"""

from __future__ import annotations

import re
import sys
import time

PREFIX = "[progress]"
LINE = re.compile(r"^\[progress\]\s+(\d+)\s*/\s*(\d+)\s*(.*)$")

# What lightning and rich print for a stage that has a bar: "Epoch 0/0 ━━━ 44/44 ..."
# or "Predicting ---- 7/7 0:00:02". The stage word leads, the counter follows.
STAGE = re.compile(
    r"^\s*(Epoch \d+/\d+|Fitting|Training|Validation|Validating|Testing|Predicting|Sanity Checking)"
    r"\b.*?(\d+)/(\d+)"
)


def format_progress(done: int, total: int, text: str = "") -> str:
    return f"{PREFIX} {done}/{total} {text}".rstrip()


def parse_progress(line: str) -> tuple[int, int, str] | None:
    """(done, total, text) from a `[progress]` line or a lightning stage line, else None."""
    m = LINE.match(line)
    if m:
        return int(m.group(1)), int(m.group(2)), m.group(3).strip()
    m = STAGE.match(line)
    if m:
        return int(m.group(2)), int(m.group(3)), m.group(1).strip()
    return None


def eta(started: float, done: int, total: int) -> str:
    """`mm:ss` (or `h:mm:ss`) left, from the rate so far; blank before it can tell."""
    if done <= 0 or total <= 0:
        return ""
    left = (time.time() - started) / done * (total - done)
    return hms(left)


def hms(seconds: float) -> str:
    seconds = max(int(round(seconds)), 0)
    hours, rest = divmod(seconds, 3600)
    minutes, secs = divmod(rest, 60)
    return f"{hours}:{minutes:02d}:{secs:02d}" if hours else f"{minutes}:{secs:02d}"


class Reporter:
    """Prints `[progress]` lines, at most every `every` seconds plus the last one."""

    def __init__(self, total: int, every: float = 1.0, stream=None) -> None:
        self.total = max(int(total), 1)
        self.every = every
        self.stream = stream or sys.stdout
        self.started = time.time()
        self._last = 0.0

    def step(self, done: int, text: str = "", force: bool = False) -> None:
        now = time.time()
        if not force and done < self.total and now - self._last < self.every:
            return
        self._last = now
        remaining = eta(self.started, done, self.total)
        tail = f"{text}  eta {remaining}" if remaining and done < self.total else text
        print(format_progress(done, self.total, tail), file=self.stream, flush=True)
