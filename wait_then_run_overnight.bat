@echo off
REM Waits for current debugging to finish, then runs overnight tests

echo ========================================
echo Waiting for debugging agent to finish...
echo ========================================
echo.
echo Monitoring for Python processes...
echo You can go to bed - tests will start automatically!
echo.

:wait_loop
cmd /c "tasklist | findstr /i python.exe" >nul 2>&1
if %ERRORLEVEL%==0 (
    echo [%time%] Python still running... waiting 60 seconds
    timeout /t 60 /nobreak >nul
    goto :wait_loop
)

echo.
echo [%time%] No Python detected. Waiting 30 sec safety buffer...
timeout /t 30 /nobreak >nul

REM Double check
cmd /c "tasklist | findstr /i python.exe" >nul 2>&1
if %ERRORLEVEL%==0 (
    echo [%time%] Python restarted. Continuing to wait...
    goto :wait_loop
)

echo.
echo ========================================
echo [%time%] Debugging complete! Starting overnight tests...
echo ========================================
echo.

call run_overnight_tests.bat
