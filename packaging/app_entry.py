"""Entry point of the packaged app ("IC Inspection.exe").

With no special arguments it opens the GUI. The GUI runs training and the
setup scripts as child processes (`gui.runner`); in the packaged app there is
no python.exe for that, so the exe re-runs itself in one of two modes:

    "IC Inspection.exe" --run-module icinspect.ocr.train --epochs 20
    "IC Inspection.exe" --run-script <app>\\scripts\\init_dataset.py --force
"""

from __future__ import annotations

import multiprocessing
import os
import runpy
import sys


def _reattach_stdio() -> None:
    """A windowed exe starts with sys.stdout/stderr = None even when the parent
    handed it pipes. Reopen them on the inherited handles so a child job's
    output reaches the GUI log."""
    for fd, name in ((1, "stdout"), (2, "stderr")):
        if getattr(sys, name) is not None:
            continue
        try:
            stream = open(fd, "w", encoding="utf-8", errors="replace", buffering=1, closefd=False)
        except OSError:
            stream = open(os.devnull, "w", encoding="utf-8")
        setattr(sys, name, stream)


def _use_bundled_backbones() -> None:
    """PatchCore builds its backbone with timm `pretrained=True`, which fetches the
    ImageNet weights from HuggingFace -- on an offline PC that hangs. build_app.ps1
    ships them in hf_cache\\ beside the exe; read them from there, never the net.
    Set by environment so training/inspection child processes inherit it."""
    cache = os.path.join(os.path.dirname(os.path.abspath(sys.executable)), "hf_cache", "hub")
    if os.path.isdir(cache) and "HF_HUB_CACHE" not in os.environ:
        os.environ["HF_HUB_CACHE"] = cache
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")


def main() -> int:
    # torch DataLoader workers re-launch this exe; let them through first.
    multiprocessing.freeze_support()
    _use_bundled_backbones()

    args = sys.argv[1:]
    if args[:1] == ["--run-module"] and len(args) >= 2:
        _reattach_stdio()
        sys.argv = [args[1], *args[2:]]
        runpy.run_module(args[1], run_name="__main__", alter_sys=True)
        return 0
    if args[:1] == ["--run-script"] and len(args) >= 2:
        _reattach_stdio()
        sys.argv = [args[1], *args[2:]]
        runpy.run_path(args[1], run_name="__main__")
        return 0

    _reattach_stdio()
    from icinspect.gui.app import main as gui_main

    return gui_main(args)


if __name__ == "__main__":
    raise SystemExit(main())
