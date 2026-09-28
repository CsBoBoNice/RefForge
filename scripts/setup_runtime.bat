@echo off
chcp 65001 >nul
cd /d "%~dp0.."
set "PY=%CD%\python\python.exe"
set "PYTHONNOUSERSITE=1"
set "PYTHONUTF8=1"

if not exist "%PY%" (
    echo ============================================
    echo  [错误] 未找到包内 Python。
    echo  请先双击根目录的「一键环境搭建.bat」引导运行时。
    echo ============================================
    pause
    exit /b 1
)

echo ============================================
echo  安装/更新运行环境 + 下载模型（写入包内 python\ 与 models\）
echo  参数：--skip-pip / --skip-runtime / --skip-models / --force
echo ============================================
"%PY%" "%CD%\scripts\setup_env.py" %*
if errorlevel 1 (
    echo.
    echo [错误] 安装失败，请检查网络后重试。
    pause
    exit /b 1
)
pause
exit /b 0
