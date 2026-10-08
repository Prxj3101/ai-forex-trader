@echo off
cd /d "%~dp0"
".venv\Scripts\python.exe" "src\data\download_ticks.py" %*
if errorlevel 1 exit /b 1
