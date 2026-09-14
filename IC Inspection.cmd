@echo off
REM Double-click this to open the IC Inspection window.
REM It activates the conda environment first, so nothing has to be typed.

setlocal
cd /d "%~dp0"

set "CONDA_BAT=%USERPROFILE%\miniconda3\condabin\conda.bat"
if not exist "%CONDA_BAT%" set "CONDA_BAT=%USERPROFILE%\anaconda3\condabin\conda.bat"
if not exist "%CONDA_BAT%" set "CONDA_BAT=%LOCALAPPDATA%\miniconda3\condabin\conda.bat"

if not exist "%CONDA_BAT%" (
    echo Could not find conda. Open a terminal and run:
    echo     powershell -ExecutionPolicy Bypass -File .\setup_env.ps1
    pause
    exit /b 1
)

call "%CONDA_BAT%" activate icinspect
if errorlevel 1 (
    echo The "icinspect" environment is missing. Create it with:
    echo     powershell -ExecutionPolicy Bypass -File .\setup_env.ps1
    pause
    exit /b 1
)

python -m icinspect.gui %*
if errorlevel 1 pause
endlocal
