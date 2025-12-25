@echo off
REM Waits for current debugging/test to finish, then starts optimization loop
REM Run this and go to bed - it will start automatically when ready

echo ========================================
echo Optimizer Auto-Start Script
echo ========================================
echo.
echo This script will:
echo   1. Wait for current Python processes to finish
echo   2. Wait an extra 30 seconds (safety buffer)
echo   3. Start the optimization loop automatically
echo.
echo You can go to bed now!
echo ========================================
echo.

:wait_loop
echo [%time%] Checking for running Python processes...

REM Check if any python process is running train_time_travel.py
tasklist /FI "IMAGENAME eq python.exe" 2>NUL | find /I "python.exe" >NUL
if %ERRORLEVEL%==0 (
    echo [%time%] Python still running. Waiting 60 seconds...
    timeout /t 60 /nobreak >nul
    goto :wait_loop
)

echo.
echo [%time%] No Python processes detected!
echo [%time%] Waiting 30 seconds safety buffer...
timeout /t 30 /nobreak >nul

REM Double-check no new process started
tasklist /FI "IMAGENAME eq python.exe" 2>NUL | find /I "python.exe" >NUL
if %ERRORLEVEL%==0 (
    echo [%time%] Python started again. Continuing to wait...
    goto :wait_loop
)

echo.
echo ========================================
echo [%time%] Starting Optimization Loop!
echo ========================================
echo.

REM Log start time
echo Optimization started at %date% %time% >> optimization_autostart_log.txt

REM Start the optimization loop
call optimize_loop.bat

echo.
echo ========================================
echo Optimization loop completed at %time%
echo ========================================
pause
