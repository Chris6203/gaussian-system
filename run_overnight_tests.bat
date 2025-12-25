@echo off
REM Overnight Test Runner - Full Documentation & Auto-Commit
REM Saves all configs, results, updates scoreboard, commits improvements

setlocal enabledelayedexpansion

set LOGFILE=overnight_log.txt
set RESULTS_FILE=RESULTS_TRACKER.md

echo ========================================
echo Overnight Test Runner (Full Documentation)
echo ========================================
echo Started: %date% %time%
echo ========================================
echo.

echo Overnight Test Session Started: %date% %time% > %LOGFILE%
echo. >> %LOGFILE%

REM Save starting config
echo [SETUP] Saving baseline config...
copy config.json config_backup_%date:~-4,4%%date:~-10,2%%date:~-7,2%.json >nul 2>&1

REM Create overnight results directory
set OVERNIGHT_DIR=models\overnight_%date:~-4,4%%date:~-10,2%%date:~-7,2%
mkdir %OVERNIGHT_DIR% 2>nul

REM ============================================
REM TEST 1: Baseline (current settings)
REM ============================================
echo.
echo [1/5] TEST: Baseline (current settings)
echo ============================================

set MODEL_RUN_DIR=%OVERNIGHT_DIR%\test1_baseline
set PAPER_TRADING=True
set TT_MAX_CYCLES=5000
set TT_PRINT_EVERY=500

REM Save test config
echo Test 1: Baseline > %MODEL_RUN_DIR%_config.txt
echo Settings: Default config.json >> %MODEL_RUN_DIR%_config.txt
copy config.json %OVERNIGHT_DIR%\test1_config.json >nul 2>&1

python scripts/train_time_travel.py

echo [1/5] Baseline complete >> %LOGFILE%
if exist %MODEL_RUN_DIR%\SUMMARY.txt (
    echo --- Test 1 Results --- >> %LOGFILE%
    type %MODEL_RUN_DIR%\SUMMARY.txt >> %LOGFILE%
    echo. >> %LOGFILE%
)

REM ============================================
REM TEST 2: Tighter stop loss (-5%)
REM ============================================
echo.
echo [2/5] TEST: Tight Stop Loss (-5%%)
echo ============================================

set MODEL_RUN_DIR=%OVERNIGHT_DIR%\test2_tight_stop

REM Modify config for this test
powershell -Command "(Get-Content config.json) -replace '\"hard_stop_loss_pct\": -8', '\"hard_stop_loss_pct\": -5' | Set-Content config.json"

echo Test 2: Tight Stop Loss > %MODEL_RUN_DIR%_config.txt
echo Change: hard_stop_loss_pct -8 to -5 >> %MODEL_RUN_DIR%_config.txt
copy config.json %OVERNIGHT_DIR%\test2_config.json >nul 2>&1

python scripts/train_time_travel.py

echo [2/5] Tight stop complete >> %LOGFILE%
if exist %MODEL_RUN_DIR%\SUMMARY.txt (
    echo --- Test 2 Results --- >> %LOGFILE%
    type %MODEL_RUN_DIR%\SUMMARY.txt >> %LOGFILE%
    echo. >> %LOGFILE%
)

REM Restore config
powershell -Command "(Get-Content config.json) -replace '\"hard_stop_loss_pct\": -5', '\"hard_stop_loss_pct\": -8' | Set-Content config.json"

REM ============================================
REM TEST 3: Wider take profit (+18%)
REM ============================================
echo.
echo [3/5] TEST: Wide Take Profit (+18%%)
echo ============================================

set MODEL_RUN_DIR=%OVERNIGHT_DIR%\test3_wide_tp

REM Modify config for this test
powershell -Command "(Get-Content config.json) -replace '\"hard_take_profit_pct\": 12', '\"hard_take_profit_pct\": 18' | Set-Content config.json"

echo Test 3: Wide Take Profit > %MODEL_RUN_DIR%_config.txt
echo Change: hard_take_profit_pct 12 to 18 >> %MODEL_RUN_DIR%_config.txt
copy config.json %OVERNIGHT_DIR%\test3_config.json >nul 2>&1

python scripts/train_time_travel.py

echo [3/5] Wide TP complete >> %LOGFILE%
if exist %MODEL_RUN_DIR%\SUMMARY.txt (
    echo --- Test 3 Results --- >> %LOGFILE%
    type %MODEL_RUN_DIR%\SUMMARY.txt >> %LOGFILE%
    echo. >> %LOGFILE%
)

REM Restore config
powershell -Command "(Get-Content config.json) -replace '\"hard_take_profit_pct\": 18', '\"hard_take_profit_pct\": 12' | Set-Content config.json"

REM ============================================
REM TEST 4: Stricter entry thresholds
REM ============================================
echo.
echo [4/5] TEST: Strict Entry (0.70/0.30)
echo ============================================

set MODEL_RUN_DIR=%OVERNIGHT_DIR%\test4_strict_entry
set HMM_STRONG_BULLISH=0.70
set HMM_STRONG_BEARISH=0.30

echo Test 4: Strict Entry Thresholds > %MODEL_RUN_DIR%_config.txt
echo Change: HMM_STRONG_BULLISH=0.70, HMM_STRONG_BEARISH=0.30 >> %MODEL_RUN_DIR%_config.txt

python scripts/train_time_travel.py

echo [4/5] Strict entry complete >> %LOGFILE%
if exist %MODEL_RUN_DIR%\SUMMARY.txt (
    echo --- Test 4 Results --- >> %LOGFILE%
    type %MODEL_RUN_DIR%\SUMMARY.txt >> %LOGFILE%
    echo. >> %LOGFILE%
)

REM Reset env vars
set HMM_STRONG_BULLISH=
set HMM_STRONG_BEARISH=

REM ============================================
REM TEST 5: Combined best + validation (10K)
REM ============================================
echo.
echo [5/5] TEST: 10K Validation Run
echo ============================================

set MODEL_RUN_DIR=%OVERNIGHT_DIR%\test5_10k_validation
set TT_MAX_CYCLES=10000
set TT_PRINT_EVERY=1000

echo Test 5: 10K Validation > %MODEL_RUN_DIR%_config.txt
echo Change: Extended 10K cycle validation >> %MODEL_RUN_DIR%_config.txt
copy config.json %OVERNIGHT_DIR%\test5_config.json >nul 2>&1

python scripts/train_time_travel.py

echo [5/5] 10K validation complete >> %LOGFILE%
if exist %MODEL_RUN_DIR%\SUMMARY.txt (
    echo --- Test 5 Results --- >> %LOGFILE%
    type %MODEL_RUN_DIR%\SUMMARY.txt >> %LOGFILE%
    echo. >> %LOGFILE%
)

REM ============================================
REM UPDATE RESULTS TRACKER
REM ============================================
echo.
echo [DOCS] Updating RESULTS_TRACKER.md...

echo. >> %RESULTS_FILE%
echo ## Overnight Test Session - %date% >> %RESULTS_FILE%
echo. >> %RESULTS_FILE%
echo ^| Test ^| Config Change ^| Win Rate ^| P/L ^| Trades ^| Per-Trade P/L ^| >> %RESULTS_FILE%
echo ^|---^|---^|---^|---^|---^|---^| >> %RESULTS_FILE%

REM Parse each test result and add to tracker
for %%t in (1 2 3 4 5) do (
    set "TEST_DIR=%OVERNIGHT_DIR%\test%%t_*"
    for /d %%d in (!TEST_DIR!) do (
        if exist "%%d\SUMMARY.txt" (
            echo Found results in %%d >> %LOGFILE%
        )
    )
)

echo. >> %RESULTS_FILE%
echo See overnight_log.txt and %OVERNIGHT_DIR% for full details. >> %RESULTS_FILE%
echo. >> %RESULTS_FILE%

REM ============================================
REM GIT COMMIT RESULTS
REM ============================================
echo.
echo [GIT] Committing results...

git add %RESULTS_FILE%
git add %LOGFILE%
git add %OVERNIGHT_DIR%\*.txt 2>nul
git add %OVERNIGHT_DIR%\*.json 2>nul

git commit -m "Overnight tests: %date% - 5 configuration tests

Tests run:
1. Baseline (current settings)
2. Tight stop loss (-5%%)
3. Wide take profit (+18%%)
4. Strict entry (0.70/0.30)
5. 10K validation

See overnight_log.txt and %OVERNIGHT_DIR% for full results.

Generated with Claude Code automated testing" 2>nul

if %ERRORLEVEL%==0 (
    echo [GIT] Results committed successfully!
    echo Git commit successful >> %LOGFILE%
) else (
    echo [GIT] Nothing to commit or commit failed
    echo Git commit skipped >> %LOGFILE%
)

REM ============================================
REM COMPLETION
REM ============================================
echo.
echo ========================================
echo OVERNIGHT TESTS COMPLETE
echo ========================================
echo Finished: %date% %time%
echo.
echo Results saved to:
echo   - %OVERNIGHT_DIR%\
echo   - %LOGFILE%
echo   - %RESULTS_FILE%
echo.
echo Run the optimizer agent tomorrow to analyze:
echo   /trading-bot-test-optimizer analyze overnight results
echo ========================================

echo. >> %LOGFILE%
echo Completed: %date% %time% >> %LOGFILE%

pause
