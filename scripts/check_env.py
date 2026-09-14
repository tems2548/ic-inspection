"""Verify the environment before a long training run.

    python scripts/check_env.py
"""

from __future__ import annotations

import importlib
import platform
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

OK, BAD, WARN = "[ok]  ", "[FAIL]", "[warn]"


def check_python() -> bool:
    v = sys.version_info
    ok = (3, 10) <= (v.major, v.minor) < (3, 13)
    print(f"{OK if ok else BAD} python {platform.python_version()}"
          + ("" if ok else "  -- anomalib needs 3.10-3.12; create the conda env (see README)"))
    return ok


def check_package(name: str, required: bool = True, attr: str = "__version__") -> bool:
    try:
        mod = importlib.import_module(name)
    except ImportError as exc:
        print(f"{BAD if required else WARN} {name}: not installed ({exc.msg})")
        return not required
    print(f"{OK} {name} {getattr(mod, attr, '?')}")
    return True


def check_torch() -> bool:
    try:
        import torch
    except ImportError:
        print(f"{BAD} torch: not installed")
        return False

    print(f"{OK} torch {torch.__version__} (cuda build: {torch.version.cuda or 'cpu-only'})")
    if not torch.cuda.is_available():
        print(f"{WARN} CUDA not available -- training will run on CPU (slow but correct)")
        return True

    name = torch.cuda.get_device_name(0)
    cap = torch.cuda.get_device_capability(0)
    print(f"{OK} GPU: {name}  sm_{cap[0]}{cap[1]}  "
          f"{torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GiB")

    arch_list = torch.cuda.get_arch_list()
    if f"sm_{cap[0]}{cap[1]}" not in arch_list:
        print(f"{BAD} this torch build has no kernels for sm_{cap[0]}{cap[1]} "
              f"(built for: {', '.join(arch_list)})")
        print("       reinstall: pip install --force-reinstall torch torchvision "
              "--index-url https://download.pytorch.org/whl/cu128")
        return False

    try:  # a real kernel launch catches driver/arch mismatches the version strings hide
        (torch.randn(64, 64, device="cuda") @ torch.randn(64, 64, device="cuda")).sum().item()
        print(f"{OK} CUDA matmul smoke test passed")
    except Exception as exc:
        print(f"{BAD} CUDA compute failed: {exc}")
        return False
    return True


def check_dataset() -> bool:
    try:
        from icinspect.config import resolve_paths
    except ImportError as exc:
        print(f"{BAD} icinspect not importable: {exc}")
        return False

    paths = resolve_paths()
    print(f"{OK if paths.dataset_root.exists() else WARN} dataset root: {paths.dataset_root}"
          + ("" if paths.dataset_root.exists() else "  -- run: python scripts/init_dataset.py"))
    if not paths.dataset_root.exists():
        return True

    exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

    def count(p: Path) -> int:
        return sum(1 for f in p.rglob("*") if f.is_file() and f.suffix.lower() in exts) if p.is_dir() else 0

    train_good = count(paths.anomaly_root / "train/good")
    test_good = count(paths.anomaly_root / "test/good")
    test_bad = count(paths.anomaly_root / "test/defect")
    print(f"      anomaly: train/good={train_good}  test/good={test_good}  test/defect={test_bad}")
    if train_good == 0:
        print(f"{WARN} no training images yet -- PatchCore cannot build a memory bank")
    elif train_good < 20:
        print(f"{WARN} only {train_good} good images; 30+ gives a noticeably steadier threshold")

    for split in ("train", "val", "test"):
        f = paths.labels_file(split)
        if f.exists():
            n = sum(1 for ln in f.read_text(encoding="utf-8").splitlines()
                    if ln.strip() and not ln.lstrip().startswith("#"))
            print(f"      ocr {split}.txt: {n} labelled sample(s)")
    return True


def main() -> int:
    print("=== IC inspection environment check ===")
    results = [
        check_python(),
        check_torch(),
        check_package("torchvision"),
        check_package("anomalib"),
        check_package("PIL", attr="__version__"),
        check_package("numpy"),
        check_package("yaml", attr="__version__"),
        check_package("matplotlib", required=False),
        check_dataset(),
    ]
    failed = results.count(False)
    print("\n" + (f"{failed} check(s) failed -- see above." if failed else "All checks passed."))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
