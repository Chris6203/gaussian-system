@echo off
REM Trading Bot Optimization Loop - Agent-Driven
REM The agent decides what to test, when to continue, and when to stop

setlocal enabledelayedexpansion

REM Configuration
set CYCLES_PER_TEST=5000
set PRINT_EVERY=100
set MAX_ITERATIONS=50
set PAUSE_BETWEEN_RUNS=5

REM Create optimization log
set "OPTLOG=optimization_log.txt"
echo ======================================== >> %OPTLOG%
echo Optimization Session Started: %date% %time% >> %OPTLOG%
echo ======================================== >> %OPTLOG%

echo ========================================
echo Trading Bot Optimization Loop (Agent-Driven)
echo ========================================
echo The agent will decide what to test and when to stop.
echo Max iterations: %MAX_ITERATIONS% (safety limit)
echo ========================================
echo.

set ITERATION=0

:loop
set /a ITERATION+=1

if %ITERATION% GTR %MAX_ITERATIONS% (
    echo.
    echo [SAFETY] Reached max iterations (%MAX_ITERATIONS%). Stopping.
    echo To continue, run this script again.
    goto :end
)

echo.
echo ========================================
echo ITERATION %ITERATION%
echo ========================================
echo.

REM Ask agent what to test next (creates next_test_config.json)
echo [%ITERATION%] Asking agent what to test next...
claude "Run /trading-bot-test-optimizer in PLANNING mode. Analyze RESULTS_TRACKER.md and recent test results. Decide what improvement to try next. Create or update next_test_config.json with: {\"continue\": true/false, \"change_description\": \"what you changed\", \"hypothesis\": \"why this might help\", \"cycles\": 5000}. If you believe we've found a good solution or hit diminishing returns, set continue to false. Make the code/config changes now, then commit them with a descriptive message."

REM Check if agent wants to continue
if not exist next_test_config.json (
    echo [%ITERATION%] Agent did not create next_test_config.json. Assuming continue...
    set SHOULD_CONTINUE=true
    set TEST_CYCLES=%CYCLES_PER_TEST%
) else (
    for /f "tokens=2 delims=:," %%a in ('findstr "continue" next_test_config.json') do (
        set "CONTINUE_RAW=%%a"
        set "SHOULD_CONTINUE=!CONTINUE_RAW: =!"
    )
    for /f "tokens=2 delims=:," %%a in ('findstr "cycles" next_test_config.json') do (
        set "CYCLES_RAW=%%a"
        set "TEST_CYCLES=!CYCLES_RAW: =!"
    )
    if "!TEST_CYCLES!"=="" set TEST_CYCLES=%CYCLES_PER_TEST%
)

echo [%ITERATION%] Continue: %SHOULD_CONTINUE%
echo [%ITERATION%] Cycles: %TEST_CYCLES%

if "%SHOULD_CONTINUE%"=="false" (
    echo.
    echo [%ITERATION%] Agent decided to stop optimization.
    echo Check RESULTS_TRACKER.md for final recommendations.
    goto :end
)

REM Generate unique run directory name
for /f "tokens=2 delims==" %%a in ('wmic OS Get localdatetime /value') do set "dt=%%a"
set "TIMESTAMP=!dt:~0,8!_!dt:~8,6!"
set "RUN_DIR=models/opt_iter%ITERATION%_!TIMESTAMP!"

echo [%ITERATION%] Starting test run: !RUN_DIR!
echo [%ITERATION%] Time: %date% %time%
echo Iteration %ITERATION%: !RUN_DIR! >> %OPTLOG%

REM Run the test
set MODEL_RUN_DIR=!RUN_DIR!
set PAPER_TRADING=True
set TT_MAX_CYCLES=%TEST_CYCLES%
set TT_PRINT_EVERY=%PRINT_EVERY%

python scripts/train_time_travel.py

if errorlevel 1 (
    echo [%ITERATION%] ERROR: Test failed with error code %errorlevel%
    echo ERROR at iteration %ITERATION% >> %OPTLOG%
    echo [%ITERATION%] Asking agent to diagnose...
    claude "Run /trading-bot-test-optimizer - The test at !RUN_DIR! failed. Check the error, diagnose the issue, fix it, and update next_test_config.json to retry."
    goto :loop
)

echo.
echo [%ITERATION%] Test complete. Invoking agent for analysis...
echo.

REM Analyze results, update docs, commit
claude "Run /trading-bot-test-optimizer in ANALYSIS mode. The test at !RUN_DIR! just completed. Read the SUMMARY.txt, analyze results, update RESULTS_TRACKER.md with findings, and git commit the results update. Then prepare the next improvement."

echo.
echo [%ITERATION%] Iteration complete.
timeout /t %PAUSE_BETWEEN_RUNS% /nobreak >nul

goto :loop

:end
echo.
echo ========================================
echo OPTIMIZATION SESSION COMPLETE
echo Ran %ITERATION% iterations
echo Check RESULTS_TRACKER.md for full results
echo Check %OPTLOG% for session log
echo ========================================

REM Final summary from agent
claude "Run /trading-bot-test-optimizer in SUMMARY mode. Create a final summary of this optimization session. What worked? What didn't? What are the recommended settings? Update RESULTS_TRACKER.md with an 'Optimization Summary' section and commit it."

pause
