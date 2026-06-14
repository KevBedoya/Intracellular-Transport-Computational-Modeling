@echo off
REM ===================================================================
REM  build_windows.bat -- build the Biophysics GUI into a Windows .exe
REM
REM  Usage (from a "Command Prompt" in this folder):
REM      build_windows.bat
REM
REM  Produces:
REM      dist\Project2025App\Project2025App.exe   (plus its support folder)
REM
REM  Requires: Python 3.9+ (64-bit) on PATH. Run on a WINDOWS machine --
REM  PyInstaller cannot cross-compile a Windows .exe from macOS/Linux.
REM ===================================================================
setlocal enabledelayedexpansion
cd /d "%~dp0"

set VENV=.venv-win

echo ==^> Setting up virtual environment (%VENV%)
if not exist "%VENV%" (
    python -m venv "%VENV%"
)
call "%VENV%\Scripts\activate.bat"

echo ==^> Installing dependencies
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

echo ==^> Building with PyInstaller
if exist build rmdir /s /q build
if exist dist rmdir /s /q dist
pyinstaller Project2025App.spec --noconfirm --clean
if errorlevel 1 (
    echo BUILD FAILED.
    exit /b 1
)

echo.
echo ==^> Done.
echo     Folder: dist\Project2025App\
echo     Run:    dist\Project2025App\Project2025App.exe
echo.
echo     To distribute, zip the entire dist\Project2025App\ folder.
echo     The .exe will not run without the files alongside it.
endlocal
