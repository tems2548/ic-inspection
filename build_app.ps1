# Builds the standalone IC Inspection app and its installer. The result runs on
# any Windows 10/11 PC with no Python or conda installed.
#
#   powershell -ExecutionPolicy Bypass -File .\build_app.ps1
#
# Output:
#   dist\IC-Inspection-Setup.exe           <- give this to other people
#   dist\IC Inspection\IC Inspection.exe   <- the app itself, to try it here
#   dist\IC-Inspection.zip                 <- only with -Zip
#
# The app carries a CPU-only PyTorch. It is built in its own conda env
# ("icinspect-build"), created on the first run, so the CUDA build in the
# "icinspect" development env is never touched.

param(
    [string]$EnvName = "icinspect-build",
    # Which PatchCore runs to ship: "v24", "v24,v8", or "all"; empty = the highest vN.
    # The first one listed becomes the app's default detector.
    [string]$PatchcoreVersion = "",
    [switch]$NoModels,                # ship without trained models
    [switch]$NoInstaller,             # skip Setup.exe (just the folder)
    [switch]$Zip                      # also make a portable zip
)

$ErrorActionPreference = "Stop"
$ProjectRoot = $PSScriptRoot
Set-Location $ProjectRoot

function Step($msg) { Write-Host "`n=== $msg ===" -ForegroundColor Cyan }

# -- 1. build env -------------------------------------------------------------
$condaBase = @("$env:USERPROFILE\miniconda3", "$env:USERPROFILE\anaconda3",
               "$env:LOCALAPPDATA\miniconda3", "$env:ProgramData\miniconda3") |
             Where-Object { Test-Path "$_\condabin\conda.bat" } | Select-Object -First 1
if (-not $condaBase) { throw "conda not found (looked for miniconda3/anaconda3)" }
$Py = "$condaBase\envs\$EnvName\python.exe"

if (-not (Test-Path $Py)) {
    Step "Creating build env '$EnvName' (python 3.11, CPU torch)"
    & "$condaBase\condabin\conda.bat" create -y -n $EnvName python=3.11
    if ($LASTEXITCODE -ne 0) { throw "conda create failed" }
    & $Py -m pip install --upgrade pip wheel
    & $Py -m pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cpu
    if ($LASTEXITCODE -ne 0) { throw "torch install failed" }
    & $Py -m pip install -c packaging\constraints.txt "anomalib==2.7.0" numpy pillow pyyaml matplotlib pyinstaller
    if ($LASTEXITCODE -ne 0) { throw "dependency install failed" }
}
# Always refresh the project itself so the build sees the current code.
& $Py -m pip install --no-deps -e $ProjectRoot | Out-Null

# -- 2. PyInstaller -------------------------------------------------------------
Step "Bundling with PyInstaller (takes several minutes)"
$App = Join-Path $ProjectRoot "dist\IC Inspection"
& $Py -m PyInstaller --noconfirm --clean --distpath dist --workpath build packaging\ic_inspection.spec
if ($LASTEXITCODE -ne 0) { throw "PyInstaller failed" }

# -- 3. files the app reads from beside the exe --------------------------------
Step "Adding configs, scripts and models"
Copy-Item -Recurse -Force configs, scripts $App
Copy-Item -Force README.md $App
Get-ChildItem "$App\scripts" -Recurse -Include __pycache__ | Remove-Item -Recurse -Force
New-Item -ItemType Directory -Force "$App\results" | Out-Null

if (-not $NoModels) {
    $ocr = "results\ocr\crnn_best.pt"
    if (Test-Path $ocr) {
        New-Item -ItemType Directory -Force "$App\results\ocr" | Out-Null
        Copy-Item -Force $ocr "$App\results\ocr\"
        Write-Host "marking reader: $ocr"
    } else { Write-Host "no $ocr - shipping without a marking reader" -ForegroundColor Yellow }

    $pcRoot = "results\patchcore\Patchcore\ic"
    $versions = Get-ChildItem $pcRoot -Directory -ErrorAction SilentlyContinue |
                Where-Object { $_.Name -match '^v\d+$' -and (Test-Path "$($_.FullName)\weights\lightning\model.ckpt") }
    $byNumber = $versions | Sort-Object { [int]$_.Name.Substring(1) } -Descending
    if ($PatchcoreVersion -eq "all") { $picks = @($byNumber) }
    elseif ($PatchcoreVersion) {
        $picks = @()
        foreach ($name in ($PatchcoreVersion -split "[,\s]+" | Where-Object { $_ })) {
            $match = $versions | Where-Object Name -eq $name
            if (-not $match) { throw "PatchCore $name not found under $pcRoot (have: $($byNumber.Name -join ', '))" }
            $picks += $match
        }
    }
    else { $picks = @($byNumber | Select-Object -First 1) }

    if ($picks) {
        # The app defaults to the newest file, so date the copies in the order listed:
        # the first pick is the newest and is selected when the app opens.
        $stamp = Get-Date
        foreach ($pick in $picks) {
            $dest = "$App\$pcRoot\$($pick.Name)\weights\lightning"
            New-Item -ItemType Directory -Force $dest | Out-Null
            Copy-Item -Force "$($pick.FullName)\weights\lightning\model.ckpt" $dest
            (Get-Item "$dest\model.ckpt").LastWriteTime = $stamp
            $stamp = $stamp.AddMinutes(-1)
            Write-Host "defect detector: $pcRoot\$($pick.Name)"
        }
    } else { Write-Host "no PatchCore checkpoint found - shipping without a defect detector" -ForegroundColor Yellow }
}

# PatchCore's pretrained backbone (timm, from HuggingFace). Without it the app
# needs internet to load or train a detector. Copied from this PC's HF cache,
# which has it once a PatchCore has been trained or run here. Only refs\ and
# snapshots\ are copied (dereferenced), so the weights are not stored twice.
$hfCache = if ($env:HF_HUB_CACHE) { $env:HF_HUB_CACHE }
           elseif ($env:HF_HOME) { "$env:HF_HOME\hub" }
           else { "$env:USERPROFILE\.cache\huggingface\hub" }
$backbones = Get-ChildItem $hfCache -Directory -Filter "models--timm--*" -ErrorAction SilentlyContinue
if ($backbones) {
    foreach ($b in $backbones) {
        $dest = "$App\hf_cache\hub\$($b.Name)"
        New-Item -ItemType Directory -Force $dest | Out-Null
        Copy-Item -Recurse -Force "$($b.FullName)\refs" $dest
        foreach ($snap in Get-ChildItem "$($b.FullName)\snapshots" -Directory) {
            $snapDest = "$dest\snapshots\$($snap.Name)"
            New-Item -ItemType Directory -Force $snapDest | Out-Null
            # File.Copy follows a snapshot symlink to its blob.
            foreach ($f in Get-ChildItem $snap.FullName -File) {
                [System.IO.File]::Copy($f.FullName, "$snapDest\$($f.Name)", $true)
            }
        }
        Write-Host "backbone weights: $($b.Name)"
    }
} else {
    Write-Host "no timm backbone in $hfCache - the app will need internet for PatchCore" -ForegroundColor Yellow
}

# -- 4. installer -----------------------------------------------------------------
if (-not $NoInstaller) {
    Step "Building the installer (Inno Setup; compressing takes a while)"
    $iscc = @("$env:LOCALAPPDATA\Programs\Inno Setup 6\ISCC.exe",
              "${env:ProgramFiles(x86)}\Inno Setup 6\ISCC.exe",
              "$env:ProgramFiles\Inno Setup 6\ISCC.exe") | Where-Object { Test-Path $_ } | Select-Object -First 1
    if (-not $iscc) {
        Write-Host "Inno Setup 6 not found - installing it for this user (winget)"
        winget install --id JRSoftware.InnoSetup -e --scope user --accept-package-agreements --accept-source-agreements --disable-interactivity
        $iscc = "$env:LOCALAPPDATA\Programs\Inno Setup 6\ISCC.exe"
        if (-not (Test-Path $iscc)) { throw "Inno Setup install failed; install it from https://jrsoftware.org/isdl.php" }
    }

    $version = (Select-String -Path pyproject.toml -Pattern '^version\s*=\s*"(.+)"').Matches[0].Groups[1].Value
    $isccArgs = @("/Q", "/DAppVersion=$version")
    # One Setup.exe tops out near 2 GB compressed; above that Inno splits it into
    # Setup.exe + Setup-1.bin ... (the raw folder size is a safe proxy).
    $rawGB = (Get-ChildItem $App -Recurse -File | Measure-Object Length -Sum).Sum / 1GB
    if ($rawGB -gt 3.5) {
        $isccArgs += "/DDiskSpan"
        Write-Host ("app is {0:N1} GB - the installer will be split into Setup.exe + .bin files" -f $rawGB) -ForegroundColor Yellow
    }
    Get-ChildItem dist -Filter "IC-Inspection-Setup*" | Remove-Item -Force
    & $iscc @isccArgs packaging\installer.iss
    if ($LASTEXITCODE -ne 0) { throw "Inno Setup failed" }
    Get-ChildItem dist -Filter "IC-Inspection-Setup*" |
        ForEach-Object { Write-Host ("{0} ({1:N0} MB)" -f $_.FullName, ($_.Length / 1MB)) }
}

# -- 5. zip (optional) ------------------------------------------------------------
if ($Zip) {
    Step "Zipping"
    $zip = Join-Path $ProjectRoot "dist\IC-Inspection.zip"
    if (Test-Path $zip) { Remove-Item $zip }
    # Windows' bsdtar writes zip and copes with files over 2 GB (Compress-Archive does not).
    tar -a -c -f $zip -C dist "IC Inspection"
    if ($LASTEXITCODE -ne 0) { throw "zip failed" }
    Write-Host ("{0} ({1:N0} MB)" -f $zip, ((Get-Item $zip).Length / 1MB))
}

Write-Host "`nDone." -ForegroundColor Green
Write-Host "  app:       $App\IC Inspection.exe"
if (-not $NoInstaller) { Write-Host "  installer: $(Join-Path $ProjectRoot 'dist\IC-Inspection-Setup.exe')" }
