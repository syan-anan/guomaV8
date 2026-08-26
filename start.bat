@echo off
set PORT=8000
cd /d %~dp0
.venv\Scripts\python.exe run.py
echo done
