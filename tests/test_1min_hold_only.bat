@echo off
REM Test A: 1 minute hold time only (no $50 cap)
cd /d E:\gaussian\output3
set TT_MAX_CYCLES=5000
set TT_PRINT_EVERY=100
set PAPER_TRADING=True
set MODEL_RUN_DIR=models/test_1min_hold

REM Fix A: Reduced min hold from 5 min to 1 min
set STOP_LOSS_MIN_HOLD_MINUTES=1.0
REM Disable max loss cap
set MAX_LOSS_PER_TRADE=-99999

echo ========================================
echo TEST A: 1 MINUTE HOLD ONLY
echo ========================================
echo STOP_LOSS_MIN_HOLD_MINUTES=1.0 (was 5.0)
echo MAX_LOSS_PER_TRADE=disabled
echo ========================================

python scripts/train_time_travel.py
