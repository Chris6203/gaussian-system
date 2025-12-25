@echo off
REM Comparative tests for all 4 improvement approaches
REM Run each approach individually to compare win rates

cd /d E:\gaussian\output3

echo ========================================
echo APPROACH COMPARISON TESTS
echo ========================================
echo.

REM Common settings
set ENTRY_CONTROLLER=consensus
set TT_PRINT_EVERY=100
set PAPER_TRADING=True
set TT_MAX_CYCLES=500

REM -----------------------------------------
echo.
echo [TEST 1/5] BASELINE - Consensus only (no new approaches)
echo -----------------------------------------
set MODEL_RUN_DIR=models\test_baseline
set CONSENSUS_ENABLE_MEAN_REVERSION=0
set CONSENSUS_ENABLE_REGIME_FILTER=0
set CONSENSUS_ENABLE_ADVANCED_FEATURES=0
set XGBOOST_MIN_PROFIT_EXIT=0
python scripts/train_time_travel.py 2>&1 | findstr /C:"Win Rate" /C:"P&L" /C:"Trades"
echo.

REM -----------------------------------------
echo.
echo [TEST 2/5] APPROACH 1 - Fix Exit Policy (min 3% profit for exit)
echo -----------------------------------------
set MODEL_RUN_DIR=models\test_exit_fix
set CONSENSUS_ENABLE_MEAN_REVERSION=0
set CONSENSUS_ENABLE_REGIME_FILTER=0
set CONSENSUS_ENABLE_ADVANCED_FEATURES=0
set XGBOOST_MIN_PROFIT_EXIT=3.0
python scripts/train_time_travel.py 2>&1 | findstr /C:"Win Rate" /C:"P&L" /C:"Trades"
echo.

REM -----------------------------------------
echo.
echo [TEST 3/5] APPROACH 2 - Mean Reversion (trade extremes)
echo -----------------------------------------
set MODEL_RUN_DIR=models\test_mean_reversion
set CONSENSUS_ENABLE_MEAN_REVERSION=1
set CONSENSUS_ENABLE_REGIME_FILTER=0
set CONSENSUS_ENABLE_ADVANCED_FEATURES=0
set XGBOOST_MIN_PROFIT_EXIT=0
python scripts/train_time_travel.py 2>&1 | findstr /C:"Win Rate" /C:"P&L" /C:"Trades"
echo.

REM -----------------------------------------
echo.
echo [TEST 4/5] APPROACH 3 - Regime Filter (only favorable HMM)
echo -----------------------------------------
set MODEL_RUN_DIR=models\test_regime_filter
set CONSENSUS_ENABLE_MEAN_REVERSION=0
set CONSENSUS_ENABLE_REGIME_FILTER=1
set CONSENSUS_ENABLE_ADVANCED_FEATURES=0
set XGBOOST_MIN_PROFIT_EXIT=0
python scripts/train_time_travel.py 2>&1 | findstr /C:"Win Rate" /C:"P&L" /C:"Trades"
echo.

REM -----------------------------------------
echo.
echo [TEST 5/5] APPROACH 4 - Advanced Features (VIX term + skew)
echo -----------------------------------------
set MODEL_RUN_DIR=models\test_advanced
set CONSENSUS_ENABLE_MEAN_REVERSION=0
set CONSENSUS_ENABLE_REGIME_FILTER=0
set CONSENSUS_ENABLE_ADVANCED_FEATURES=1
set XGBOOST_MIN_PROFIT_EXIT=0
python scripts/train_time_travel.py 2>&1 | findstr /C:"Win Rate" /C:"P&L" /C:"Trades"
echo.

REM -----------------------------------------
echo.
echo [BONUS] ALL APPROACHES COMBINED
echo -----------------------------------------
set MODEL_RUN_DIR=models\test_all_combined
set CONSENSUS_ENABLE_MEAN_REVERSION=1
set CONSENSUS_ENABLE_REGIME_FILTER=1
set CONSENSUS_ENABLE_ADVANCED_FEATURES=1
set XGBOOST_MIN_PROFIT_EXIT=3.0
python scripts/train_time_travel.py 2>&1 | findstr /C:"Win Rate" /C:"P&L" /C:"Trades"
echo.

echo ========================================
echo TESTS COMPLETE - Compare results above
echo ========================================
pause
