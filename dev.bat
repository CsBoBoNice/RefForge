@echo off
chcp 65001 >nul
cd /d "%~dp0"

set "PATH=%~dp0python;%~dp0python\Scripts;%~dp0bin;%PATH%"

if "%~1"=="" (
    echo Usage: dev.bat ^<script.py^> [args]  or  dev.bat -m ^<module^> [args]
    echo Example: dev.bat scripts\run_segmentation_test.py
    pause
    exit /b 0
)

"%~dp0python\python.exe" %*
if errorlevel 1 pause
