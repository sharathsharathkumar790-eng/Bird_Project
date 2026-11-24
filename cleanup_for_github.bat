@echo off
REM Cleans the Avian AI folder before uploading to GitHub.
REM Nothing is deleted for good: unused files are MOVED to D:\Avian_AI_removed_files
cd /d "%~dp0"
set "DEST=D:\Avian_AI_removed_files"
if not exist "%DEST%\Models" mkdir "%DEST%\Models"
if not exist "%DEST%\test_samples" mkdir "%DEST%\test_samples"

echo Moving unused model files...
for %%F in (_downloads image audio audio_v2) do if exist "Models\%%F" move "Models\%%F" "%DEST%\Models\" >nul

echo Moving old audio samples (used only by the old audio model)...
if exist "test_samples\audio" move "test_samples\audio" "%DEST%\test_samples\" >nul

echo Moving backups, empty folders and the unfinished git setup...
if exist ".git" attrib -h ".git"
for %%F in (.git backups data docs tests) do if exist "%%F" move "%%F" "%DEST%\" >nul
if exist "restore_frontend_before_extras.bat" move "restore_frontend_before_extras.bat" "%DEST%\" >nul

echo Removing Python cache folders (Python recreates them automatically)...
for /d /r %%D in (__pycache__) do if exist "%%D" rd /s /q "%%D"

echo.
echo Done. Moved files are in %DEST%
pause
