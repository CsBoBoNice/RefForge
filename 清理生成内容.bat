@echo off
setlocal
cd /d "%~dp0"
echo ============================================
echo  Clean generated content
echo    - output\ (segments, voiceprint_center.json, run.log)
echo    - input\*.srt (generated subtitles)
echo    - %%TEMP%%\ref_forge_* (test work dirs)
echo  Protected: models, python, bin, app, scripts, input videos
echo ============================================
"%~dp0python\python.exe" "%~dp0scripts\clean_generated.py" --yes %*
if errorlevel 1 pause
pause
