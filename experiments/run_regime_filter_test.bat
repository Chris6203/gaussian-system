@echo off
REM Phase 12: Regime Filter Test
REM Tests the new regime quality filter that gates entries based on:
REM 1. Regime Quality Score (trend clarity, vol sweet spot, confidence, VIX)
REM 2. Regime Transition Detection (stability check)
REM 3. VIX-HMM Reconciliation (implied vs realized vol alignment)

echo ========================================
echo Phase 12: Regime Filter Test
echo ========================================

REM Test configuration
set REGIME_FILTER_ENABLED=1
set REGIME_MIN_QUALITY=0.35
set REGIME_FULL_SIZE_QUALITY=0.65
set MODEL_RUN_DIR=models/regime_filter_test
set TT_MAX_CYCLES=5000
set TT_PRINT_EVERY=500
set PAPER_TRADING=True

echo.
echo Configuration:
echo   REGIME_FILTER_ENABLED=%REGIME_FILTER_ENABLED%
echo   REGIME_MIN_QUALITY=%REGIME_MIN_QUALITY%
echo   MODEL_RUN_DIR=%MODEL_RUN_DIR%
echo   TT_MAX_CYCLES=%TT_MAX_CYCLES%
echo.

python scripts/train_time_travel.py

echo.
echo ========================================
echo Test Complete. Check %MODEL_RUN_DIR%\SUMMARY.txt
echo ========================================
