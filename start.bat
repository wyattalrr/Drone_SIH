@echo off
title Autonomous Disaster Response Drone System - SIH Command Center
echo =====================================================================
echo  AI Autonomous Disaster Response Drone System - SIH Mission Runner
echo =====================================================================
cd /d "%~dp0"

if exist "sih_venv\Scripts\python.exe" (
    echo [OK] Using virtual environment: sih_venv
    "sih_venv\Scripts\python.exe" run_system.py %*
) else (
    echo [!] sih_venv not found, falling back to system python...
    python run_system.py %*
)

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo [ERROR] Application exited with error code %ERRORLEVEL%
    pause
)
