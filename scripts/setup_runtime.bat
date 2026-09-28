@echo off
chcp 65001 >nul
cd /d "%~dp0.."
set "PY=%CD%\python\python.exe"
set "PYTHONNOUSERSITE=1"
set "PYTHONUTF8=1"

if not exist "%PY%" (
    echo ============================================
    echo  [ERROR] Bundled Python not found.
    echo  Run the root one-click setup script first to bootstrap the runtime.
    echo ============================================
    pause
    exit /b 1
)

echo ============================================
echo  Install/update runtime + download models (into bundled python\ and models\)
echo  Options: --skip-pip / --skip-runtime / --skip-models / --force
echo ============================================
"%PY%" "%CD%\scripts\setup_env.py" %*
if errorlevel 1 (
    echo.
    echo [ERROR] Setup failed. Check your network and retry.
    pause
    exit /b 1
)
pause
exit /b 0
