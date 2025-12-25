@echo off
REM Test B: $50 max loss cap only (with original 5 min hold)
cd /d E:\gaussian\output3
set TT_MAX_CYCLES=5000
set TT_PRINT_EVERY=100
set PAPER_TRADING=True
set MODEL_RUN_DIR=models/test_50_cap

REM Keep original 5 min hold
set STOP_LOSS_MIN_HOLD_MINUTES=5.0
REM Fix B: $50 max loss cap
set MAX_LOSS_PER_TRADE=-50.0

echo ========================================
echo TEST B: $50 MAX LOSS CAP ONLY
echo ========================================
echo STOP_LOSS_MIN_HOLD_MINUTES=5.0 (original)
echo MAX_LOSS_PER_TRADE=-50.0
echo ========================================

python scripts/train_time_travel.py
