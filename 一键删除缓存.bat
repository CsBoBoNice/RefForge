@echo off
chcp 65001 >nul
setlocal
cd /d "%~dp0"
set "PYTHONUTF8=1"
echo ============================================
echo  Delete caches (clean folder for zip / copy)
echo    - __pycache__ / *.pyc / *.pyo
echo    - *.part / *.tmp / .downloads
echo    - models hidden HF .cache metadata
echo    - *.log / Thumbs.db / Desktop.ini
echo    - %%TEMP%%\ref_forge_* work dirs
echo  Keeps: source, models, python, bin, llama_cpp,
echo         input, output
echo ============================================
choice /C YN /N /M "Delete caches now? [Y/N] "
if errorlevel 2 (
    echo Cancelled.
    pause
    exit /b 0
)
echo.
"%~dp0python\python.exe" "%~dp0scripts\clean_cache.py" --yes
if errorlevel 1 (
    echo.
    echo [ERROR] Clean failed. Is the bundled Python ready?
    pause
    exit /b 1
)
pause
exit /b 0
