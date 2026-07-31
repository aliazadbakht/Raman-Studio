@echo off
REM Raman Studio - installer (Windows)
setlocal
cd /d "%~dp0"

echo === Raman Studio - Setup (Windows) ===

where python >nul 2>&1
if errorlevel 1 (
    echo ERROR: python not found on PATH. Install from https://python.org
    exit /b 1
)

for /f "delims=" %%v in ('python --version') do echo Using: %%v

if not exist "venv" (
    python -m venv venv
    echo Created virtual environment.
)

call "venv\Scripts\activate.bat"
python -m pip install --upgrade pip -q
python -m pip install -r requirements.txt

echo.
echo === Installation complete ===
echo Run the app with:  run.bat
echo.
echo Camera note: FLIR/PointGrey capture requires the Spinnaker SDK and PySpin
echo installed into the Python used by run.bat. Opening, processing, and exporting
echo spectrum files works without PySpin.
