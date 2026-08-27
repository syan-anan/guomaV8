@echo off
chcp 65001 >nul
setlocal EnableDelayedExpansion

REM === syandaV8 Windows Launcher v2.2 ===
cd /d "%~dp0"

REM Create venv if missing
if not exist ".venv\Scripts\python.exe" (
    echo Creating virtual environment...
    python -m venv .venv
    .venv\Scripts\python.exe -m pip install --upgrade pip
    echo Installing dependencies...
    .venv\Scripts\python.exe -m pip install -r requirements.txt
)

REM Load .env vars
if exist ".env" (
    for /F "tokens=1,2 delims==" %%a in (.env) do (
        set "key=%%a"
        set "val=%%b"
        set !key!=!val!
    )
)

if not defined PORT set PORT=15666
if not defined HOST set HOST=0.0.0.0

echo Captcha Solver v2.2 on !HOST!:!PORT!
echo Checking solver registry...

.venv\Scripts\python.exe run.py

echo.
echo Service stopped.
pause >nul