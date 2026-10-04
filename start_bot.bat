@echo off
REM ============================================================
REM  FOREX BOT launcher (VPS ke liye)
REM  - project folder mein jaata hai, venv chalu karta hai
REM  - shadow.py chalata hai
REM  - bot band / crash ho jaye toh 60 sec baad khud dobara start
REM  Band karna ho: Ctrl+C, phir "Terminate batch job (Y/N)?" pe Y
REM ============================================================
title FOREX BOT - shadow mode
cd /d "%~dp0"
call .venv\Scripts\activate.bat
if not exist data\live mkdir data\live

:loop
echo [%date% %time%] bot start>> data\live\restarts.log
python src\live\shadow.py
echo [%date% %time%] bot band hua (code %errorlevel%), 60 sec baad restart>> data\live\restarts.log
echo.
echo Bot band ho gaya. 60 second baad dobara start hoga (rokna ho toh Ctrl+C, phir Y)...
timeout /t 60 /nobreak
goto loop
