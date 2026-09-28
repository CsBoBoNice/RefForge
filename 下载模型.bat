@echo off
chcp 65001 >nul
cd /d "%~dp0"

echo ============================================
echo  One-click model download (excluding llama.cpp GGUF)
echo    - ASR / speaker / VAD   ^<- ModelScope
echo    - Separation models     ^<- GitHub (with CN mirrors)
echo    - Person models         ^<- GitHub / hf-mirror
echo    - Anime models          ^<- GitHub / hf-mirror (CLIP)
echo  Target dirs: models\asr\, models\separator\, models\person\
echo  Optional: --force to re-download
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
echo All models downloaded.
pause
exit /b 0

:fail
echo.
echo [ERROR] Download failed. Check your network and retry.
pause
exit /b 1
