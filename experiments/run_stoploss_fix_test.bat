@echo off
REM Stop Loss Fix Test - 5000 cycles
REM Changes tested:
REM   1. Reduced min hold for stop loss: 5 min -> 1 min
REM   2. Added $50 max loss cap per trade
REM   3. Using bandit entry controller (best baseline)

cd /d E:\gaussian\output3
set TT_MAX_CYCLES=5000
set TT_PRINT_EVERY=100
set PAPER_TRADING=True
set MODEL_RUN_DIR=models/stoploss_fix_test

echo ========================================
echo STOP LOSS FIX TEST
echo ========================================
echo Config changes:
echo   - min_hold_for_stop: 1 min (was 5 min)
echo   - max_loss_per_trade: $50
echo   - entry_controller: bandit
echo ========================================

python scripts/train_time_travel.py
