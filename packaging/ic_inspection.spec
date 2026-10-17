# PyInstaller spec for the standalone "IC Inspection" app. Built by build_app.ps1:
#
#   pyinstaller --noconfirm --distpath dist --workpath build packaging\ic_inspection.spec
#
# Output: dist\IC Inspection\  (the exe + _internal\). build_app.ps1 then adds
# configs\, scripts\, README.md and the shipped models beside the exe.

from PyInstaller.utils.hooks import collect_all, collect_submodules, copy_metadata

datas, binaries, hiddenimports = [], [], []

# Imported by name at runtime (jsonargparse / lightning / anomalib model registry),
# or read their own source for torch.jit -- take them whole.
for package in ("anomalib", "lightning", "lightning_fabric", "pytorch_lightning",
                "lightning_utilities", "torchmetrics",
                "kornia", "timm", "jsonargparse", "FrEIA", "omegaconf"):
    d, b, h = collect_all(package)
    datas += d
    binaries += b
    hiddenimports += h

# Version checks via importlib.metadata.
for dist in ("torch", "torchvision", "lightning", "pytorch-lightning", "anomalib", "torchmetrics",
             "timm", "kornia", "jsonargparse", "numpy", "pillow", "scikit-learn", "scikit-image",
             "opencv-python-headless", "huggingface_hub", "safetensors", "tqdm", "rich",
             "packaging", "filelock", "einops", "matplotlib", "lightning-utilities"):
    try:
        datas += copy_metadata(dist)
    except Exception:
        pass

# The GUI starts these as `--run-module ...`, so nothing imports them statically.
hiddenimports += collect_submodules("icinspect")

a = Analysis(
    ["app_entry.py"],
    pathex=["../src"],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    excludes=["tensorboard", "IPython", "jupyter", "notebook", "pytest", "django", "sqlalchemy"],
    module_collection_mode={
        "anomalib": "pyz+py",
        "lightning": "pyz+py",
        "lightning_fabric": "pyz+py",
        "pytorch_lightning": "pyz+py",
        "kornia": "pyz+py",
        "timm": "pyz+py",
        "torchvision": "pyz+py",
        "torchmetrics": "pyz+py",
        "jsonargparse": "pyz+py",
    },
    noarchive=False,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="IC Inspection",
    console=False,
    upx=False,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    name="IC Inspection",
)
