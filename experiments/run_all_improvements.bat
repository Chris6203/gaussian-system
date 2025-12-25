@echo off
REM Run with all 4 improvement approaches enabled
REM This is the recommended configuration for best win rate

cd /d E:\gaussian\output3

echo ========================================
echo RUNNING ALL IMPROVEMENTS COMBINED
echo ========================================
echo.
echo Approach 1: Exit Policy Fix (min 3% profit for exit)
echo Approach 2: Mean Reversion (fade RSI/BB extremes)
echo Approach 3: Regime Filter (only favorable HMM regimes)
echo Approach 4: Advanced Features (VIX term + options skew)
echo.
echo ========================================

set ENTRY_CONTROLLER=consensus
set TT_MAX_CYCLES=100
set TT_PRINT_EVERY=20
set PAPER_TRADING=True
set MODEL_RUN_DIR=models\all_improvements

REM Enable all approaches
set CONSENSUS_ENABLE_MEAN_REVERSION=1
set CONSENSUS_ENABLE_REGIME_FILTER=1
set CONSENSUS_ENABLE_ADVANCED_FEATURES=1
set XGBOOST_MIN_PROFIT_EXIT=3.0

REM Tight exit settings
set TT_MAX_HOLD_MINUTES=45

REM Debug: show environment
echo.
echo Environment variables:
echo   ENTRY_CONTROLLER=%ENTRY_CONTROLLER%
echo   TT_MAX_CYCLES=%TT_MAX_CYCLES%
echo.

python scripts/train_time_travel.py

echo.
echo ========================================
echo TEST COMPLETE
echo ========================================
