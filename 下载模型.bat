@echo off
chcp 65001 >nul
cd /d "%~dp0"

echo ============================================
echo  一键下载模型（不含 llama.cpp 的 gguf）
echo    - 语音识别 / 声纹 / VAD  ^<- ModelScope
echo    - 人声分离四档模型       ^<- GitHub（含国内镜像）
echo    - 人物提取模型           ^<- GitHub / hf-mirror
echo    - 动漫人物模型           ^<- GitHub（人脸级联）/ hf-mirror（CLIP）
echo  目标目录：models\asr\、models\separator\ 与 models\person\
echo  可选参数：--force 强制重下
echo ============================================

"%~dp0python\python.exe" "%~dp0scripts\download_models.py" %*
if errorlevel 1 goto fail

"%~dp0python\python.exe" "%~dp0scripts\download_separation_models.py" %*
if errorlevel 1 goto fail

"%~dp0python\python.exe" "%~dp0scripts\download_person_models.py" %*
if errorlevel 1 goto fail

"%~dp0python\python.exe" "%~dp0scripts\download_anime_person_models.py" %*
if errorlevel 1 goto fail

echo.
echo 全部模型下载完成。
pause
exit /b 0

:fail
echo.
echo [错误] 下载失败，请检查网络后重试。
pause
exit /b 1
