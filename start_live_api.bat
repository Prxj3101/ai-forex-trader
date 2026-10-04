@echo off
REM ============================================================
REM  LIVE API launcher (VPS ke liye)
REM  - read-only server jo laptop dashboard ko bot ka live hisaab deta hai (port 8001)
REM  - band / crash ho jaye toh 30 sec baad khud dobara start
REM  Token dekhna ho: python src\api\live_api.py --show-token
REM ============================================================
title LIVE API - read only (port 8001)
cd /d "%~dp0"
call .venv\Scripts\activate.bat

:loop
python src\api\live_api.py
echo.
echo Live API band ho gaya. 30 second baad dobara start hoga (rokna ho toh Ctrl+C, phir Y)...
timeout /t 30 /nobreak
goto loop
