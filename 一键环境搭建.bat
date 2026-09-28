@echo off
chcp 65001 >nul
setlocal
cd /d "%~dp0"
set "PYTHONNOUSERSITE=1"
set "PYTHONUTF8=1"
set "PY=%~dp0python\python.exe"

echo ============================================
echo  RefForge one-click setup (first run, online, ~20GB)
echo    1) Bootstrap portable Python 3.10
echo    2) Install dependencies (base / speech / separation / persons)
echo    3) Download runtime assets (ffmpeg / llama.cpp / GGUF)
echo    4) Download models (ASR / separator / real / anime)
echo  Existing files are skipped; re-run to resume.
echo ============================================

if exist "%PY%" goto setup

echo.
echo [1/2] Bundled Python not found, bootstrapping Python 3.10 runtime...
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\bootstrap_python.ps1"
if not exist "%PY%" goto fail

:setup
echo.
echo [2/2] Installing dependencies and downloading assets/models...
"%PY%" "%~dp0scripts\setup_env.py" %*
if errorlevel 1 goto fail

echo.
echo ============================================
echo  Setup complete. Double-click the start script to run.
echo ============================================
pause
exit /b 0

:fail
echo.
echo [ERROR] Setup did not complete. Check your network and retry.
echo         Re-run this script to resume; downloaded content is skipped.
pause
exit /b 1
