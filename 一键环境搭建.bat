@echo off
chcp 65001 >nul
setlocal
cd /d "%~dp0"
set "PYTHONNOUSERSITE=1"
set "PYTHONUTF8=1"
set "PY=%~dp0python\python.exe"

echo ============================================
echo  RefForge 一键环境搭建（首次联网，约 20GB）
echo    1) 引导包内可移植 Python 3.10
echo    2) 安装依赖（基础 / 语音 / 人声分离 / 人物）
echo    3) 下载运行期资产（ffmpeg / llama.cpp / GGUF）
echo    4) 下载模型（ASR / 分离 / 真人 / 动漫）
echo  已存在的文件会自动跳过；可重复运行以断点续传。
echo ============================================

if exist "%PY%" goto setup

echo.
echo [1/2] 未检测到包内 Python，正在引导 Python 3.10 运行时...
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\bootstrap_python.ps1"
if not exist "%PY%" goto fail

:setup
echo.
echo [2/2] 安装依赖并下载资产与模型...
"%PY%" "%~dp0scripts\setup_env.py" %*
if errorlevel 1 goto fail

echo.
echo ============================================
echo  环境搭建完成。双击 启动.bat 即可运行。
echo ============================================
pause
exit /b 0

:fail
echo.
echo [错误] 环境搭建未完成，请检查网络后重试。
echo        重新双击本脚本即可续传，已下载内容会自动跳过。
pause
exit /b 1
