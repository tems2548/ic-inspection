"""Background work for the GUI: one task at a time, UI updates via a queue.

Tkinter widgets may only be touched from the thread that created them, so a
task runs on a worker thread and talks back through `Progress` -- log lines,
status text, a counter, finished items -- which the main thread drains on a
timer. Nothing in this module imports tkinter beyond the `after` scheduler it
is handed.
"""

from __future__ import annotations

import os
import queue
import subprocess
import sys
import threading
import traceback
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from icinspect.config import PROJECT_ROOT


class Cancelled(Exception):
    """Raised inside a task when the user pressed Stop."""


class Progress:
    """A running task's handle on the UI. Every method is thread-safe."""

    def __init__(self, q: "queue.Queue[tuple[str, Any]]", cancel: threading.Event) -> None:
        self._q = q
        self._cancel = cancel

    def log(self, text: str = "") -> None:
        self._q.put(("log", text))

    def status(self, text: str) -> None:
        self._q.put(("status", text))

    def step(self, done: int, total: int) -> None:
        self._q.put(("progress", (done, total)))

    def partial(self, payload: Any) -> None:
        """Hand a finished item to the UI before the whole task ends."""
        self._q.put(("partial", payload))

    @property
    def cancelled(self) -> bool:
        return self._cancel.is_set()

    def check(self) -> None:
        """Abort the task if Stop was pressed. Call between units of work."""
        if self._cancel.is_set():
            raise Cancelled


class Worker:
    """Runs one task at a time, off the UI thread.

    A second `start` while a task is running is refused rather than queued --
    the buttons that feed it are disabled meanwhile, and silently stacking two
    model loads would be worse than doing nothing.
    """

    def __init__(self, root: Any, poll_ms: int = 60) -> None:
        self._root = root
        self._poll = poll_ms
        self._q: queue.Queue[tuple[str, Any]] = queue.Queue()
        self._cancel = threading.Event()
        self._thread: threading.Thread | None = None
        self._handlers: dict[str, Callable[[Any], None] | None] = {}
        self._draining = False

    @property
    def busy(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def start(
        self,
        fn: Callable[[Progress], Any],
        *,
        on_log: Callable[[str], None] | None = None,
        on_status: Callable[[str], None] | None = None,
        on_progress: Callable[[tuple[int, int]], None] | None = None,
        on_partial: Callable[[Any], None] | None = None,
        on_done: Callable[[Any], None] | None = None,
        on_error: Callable[[Any], None] | None = None,
        on_cancelled: Callable[[Any], None] | None = None,
    ) -> bool:
        if self.busy:
            return False
        self._cancel.clear()
        self._handlers = {
            "log": on_log,
            "status": on_status,
            "progress": on_progress,
            "partial": on_partial,
            "done": on_done,
            "error": on_error,
            "cancelled": on_cancelled,
        }
        progress = Progress(self._q, self._cancel)

        def run() -> None:
            try:
                result = fn(progress)
            except Cancelled:
                self._q.put(("cancelled", None))
            except BaseException as exc:  # noqa: BLE001 - reported in the UI, not swallowed
                self._q.put(("error", (exc, traceback.format_exc())))
            else:
                self._q.put(("done", result))

        self._thread = threading.Thread(target=run, daemon=True, name="icinspect-worker")
        self._thread.start()
        if not self._draining:
            self._draining = True
            self._root.after(self._poll, self._drain)
        return True

    def cancel(self) -> None:
        self._cancel.set()

    def _drain(self) -> None:
        finished = False
        while True:
            try:
                kind, payload = self._q.get_nowait()
            except queue.Empty:
                break
            handler = self._handlers.get(kind)
            if handler is not None:
                handler(payload)
            if kind in ("done", "error", "cancelled"):
                finished = True
        if finished and not self.busy:
            self._draining = False
            return
        self._root.after(self._poll, self._drain)


def child_env(extra: dict[str, str] | None = None) -> dict[str, str]:
    """Environment for a child process: unbuffered, and able to import icinspect."""
    env = dict(os.environ)
    env["PYTHONUNBUFFERED"] = "1"
    src = str(PROJECT_ROOT / "src")
    existing = env.get("PYTHONPATH", "")
    if src not in existing.split(os.pathsep):
        env["PYTHONPATH"] = os.pathsep.join([src, existing]) if existing else src
    env.update(extra or {})
    return env


def run_command(
    prog: Progress,
    args: Sequence[str],
    *,
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
) -> int:
    """Run a child process, streaming its output into the log. Returns the exit code.

    Stop has to reach a process that has gone quiet -- a training epoch prints
    nothing for a minute -- so a watchdog terminates it on cancel rather than
    the read loop noticing between lines.
    """
    cmd = [str(a) for a in args]
    prog.log("$ " + subprocess.list2cmdline(cmd))
    creationflags = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0) if os.name == "nt" else 0
    proc = subprocess.Popen(
        cmd,
        cwd=str(cwd) if cwd else None,
        env=child_env(env),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        stdin=subprocess.DEVNULL,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
        creationflags=creationflags,
    )

    stop = threading.Event()

    def watchdog() -> None:
        while not stop.wait(0.2):
            if prog.cancelled:
                if proc.poll() is None:
                    proc.terminate()
                return

    guard = threading.Thread(target=watchdog, daemon=True, name="icinspect-watchdog")
    guard.start()
    try:
        assert proc.stdout is not None
        _stream(prog, proc.stdout)
    finally:
        stop.set()
        if proc.stdout is not None:
            proc.stdout.close()
        code = proc.wait()
    if prog.cancelled:
        prog.log("[stopped]")
        raise Cancelled
    return code


def _stream(prog: Progress, stdout) -> None:
    """Feed a child's output to the log, turning progress into bar steps.

    A `[progress] done/total text` line (see `icinspect.progress`) becomes a
    step and a status instead of a log line. lightning's own bars arrive the
    same way -- "Fitting ━━━ 12/44" -- only they redraw with a carriage
    return rather than a newline, so the read is character-wise and splits
    on both: on `\r` the chunk is a bar update and only its counter is used;
    on `\n` a real line is logged.
    """
    from icinspect.progress import parse_progress

    buffer: list[str] = []
    while True:
        char = stdout.read(1)
        if not char:
            break
        if char not in ("\r", "\n"):
            buffer.append(char)
            continue
        text = "".join(buffer).rstrip()
        buffer.clear()
        found = parse_progress(text) if text else None
        if found is not None:
            done, total, label = found
            prog.step(done, total)
            if label:
                prog.status(label)
            if text.startswith("[progress]") or char == "\r":
                continue                         # the bar shows it; the log need not
        if char == "\n" and text:
            prog.log(text)
    if buffer:
        prog.log("".join(buffer).rstrip())


def python_module(module: str, *args: str) -> list[str]:
    """`python -m <module> ...` using the interpreter the GUI is running under."""
    return [sys.executable, "-m", module, *args]


def python_script(script: Path, *args: str) -> list[str]:
    return [sys.executable, str(script), *args]
