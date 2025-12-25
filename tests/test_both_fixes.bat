@echo off
REM Test C: Both fixes (1 min hold + $50 cap)
cd /d E:\gaussian\output3
set TT_MAX_CYCLES=5000
set TT_PRINT_EVERY=100
set PAPER_TRADING=True
set MODEL_RUN_DIR=models/test_both_fixes

REM Fix A: Reduced min hold from 5 min to 1 min
set STOP_LOSS_MIN_HOLD_MINUTES=1.0
REM Fix B: $50 max loss cap
set MAX_LOSS_PER_TRADE=-50.0

echo ========================================
echo TEST C: BOTH FIXES
echo ========================================
echo STOP_LOSS_MIN_HOLD_MINUTES=1.0 (was 5.0)
echo MAX_LOSS_PER_TRADE=-50.0
echo ========================================

python scripts/train_time_travel.py
