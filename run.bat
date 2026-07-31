@echo off
REM Raman Studio - launcher (Windows)
setlocal
cd /d "%~dp0"

REM Point the optional FLIR Spinnaker SDK at its libraries.
set "SPIN_BIN=C:\Program Files\Teledyne\Spinnaker\bin64\vs2015"
set "SPIN_CTI=C:\Program Files\Teledyne\Spinnaker\cti64\vs2015"
if exist "%SPIN_BIN%" set "PATH=%SPIN_BIN%;%PATH%"
if exist "%SPIN_CTI%" if not defined GENICAM_GENTL64_PATH set "GENICAM_GENTL64_PATH=%SPIN_CTI%"

REM Interpreter: %RAMAN_PYTHON% wins, then the venv, then whatever is on PATH.
if defined RAMAN_PYTHON (
    "%RAMAN_PYTHON%" main_qt.py
    goto :eof
)

if exist "venv\Scripts\python.exe" (
    "venv\Scripts\python.exe" main_qt.py
) else (
    python main_qt.py
)
