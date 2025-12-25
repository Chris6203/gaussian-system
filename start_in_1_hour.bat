@echo off
echo ========================================
echo Overnight tests will start in 1 HOUR
echo ========================================
echo Current time: %time%
echo Tests will begin at approximately %time% + 1 hour
echo.
echo Leave this window open and go to bed!
echo ========================================
timeout /t 3600 /nobreak
echo.
echo Starting overnight tests now!
call run_overnight_tests.bat
