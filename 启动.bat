@echo off
chcp 65001 >nul
cd /d "%~dp0"

set "PATH=%~dp0python;%~dp0python\Scripts;%~dp0bin;%PATH%"

if "%~1"=="" (
    "%~dp0python\python.exe" "%~dp0app\main.py" --interactive
) else (
    "%~dp0python\python.exe" "%~dp0app\main.py" %*
)
if errorlevel 1 pause
pause
