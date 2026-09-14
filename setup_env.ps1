# One-shot environment setup for the IC inspection project.
#
#   powershell -ExecutionPolicy Bypass -File .\setup_env.ps1
#
# Creates a conda env with Python 3.11 (anomalib does not support 3.13), installs
# a CUDA 12.8 torch build (RTX 50-series / Blackwell is sm_120 and needs cu128+),
# then anomalib and this package.

param(
    [string]$EnvName = "icinspect",
    [string]$CudaIndex = "https://download.pytorch.org/whl/cu128",
    [switch]$Cpu
)

$ErrorActionPreference = "Stop"
$ProjectRoot = $PSScriptRoot

function Step($msg) { Write-Host "`n=== $msg ===" -ForegroundColor Cyan }

if (-not (Get-Command conda -ErrorAction SilentlyContinue)) {
    throw "conda not found on PATH. Open an Anaconda/Miniconda prompt, or install Miniconda first."
}

Step "Creating conda env '$EnvName' (python 3.11)"
$existing = conda env list | Select-String -Pattern "^\s*$EnvName\s"
if ($existing) {
    Write-Host "env '$EnvName' already exists - reusing it"
} else {
    conda create -y -n $EnvName python=3.11
    if ($LASTEXITCODE -ne 0) { throw "conda create failed" }
}

# Resolve the env's own python so we never depend on `conda activate` inside a script.
# `conda run` is unreliable for value capture in a non-interactive shell, so read the
# env prefix from conda's own listing and fall back to <base>\envs\<name>.
$Py = $null
$row = (conda env list) | Where-Object { $_ -match "^\s*$([regex]::Escape($EnvName))\s+\*?\s*(?<p>[A-Za-z]:\\.+)$" }
if ($row -and $Matches.p) {
    $candidate = Join-Path $Matches.p.Trim() "python.exe"
    if (Test-Path $candidate) { $Py = $candidate }
}
if (-not $Py) {
    $candidate = Join-Path (Join-Path ((conda info --base).Trim()) "envs\$EnvName") "python.exe"
    if (Test-Path $candidate) { $Py = $candidate }
}
if (-not $Py) { throw "Could not locate python.exe for conda env '$EnvName'" }
Write-Host "python: $Py"

Step "Upgrading pip"
# setuptools is deliberately left alone: torch pins setuptools<82 and
# upgrading it here only to have torch downgrade it again is noise.
& $Py -m pip install --upgrade pip wheel

Step "Installing PyTorch"
if ($Cpu) {
    & $Py -m pip install torch torchvision
} else {
    & $Py -m pip install torch torchvision --index-url $CudaIndex
}
if ($LASTEXITCODE -ne 0) { throw "torch install failed" }

Step "Installing anomalib + project dependencies"
# anomalib can pull a different torch; install it without letting pip swap the
# CUDA build we just placed.
& $Py -m pip install "anomalib>=2.0" numpy pillow pyyaml matplotlib pytest
if ($LASTEXITCODE -ne 0) { throw "anomalib install failed" }

Step "Verifying the torch build survived"
# Single line on purpose: a multi-line script passed via -c loses its newlines here.
$probe = "import torch; print('cpu' if not torch.cuda.is_available() else ('ok' if 'sm_%d%d' % torch.cuda.get_device_capability(0) in torch.cuda.get_arch_list() else 'mismatch'))"
$archOk = (& $Py -c $probe | Out-String).Trim()
Write-Host "cuda arch check: $archOk"
if ($archOk -eq "mismatch") {
    Write-Host "torch lost its CUDA kernels - reinstalling from $CudaIndex" -ForegroundColor Yellow
    & $Py -m pip install --force-reinstall torch torchvision --index-url $CudaIndex
}

Step "Installing icinspect (editable)"
& $Py -m pip install -e "$ProjectRoot"

Step "Environment check"
& $Py (Join-Path $ProjectRoot "scripts\check_env.py")

Write-Host "`nDone. Activate with:  conda activate $EnvName" -ForegroundColor Green
Write-Host "Then:                 python scripts\init_dataset.py"
