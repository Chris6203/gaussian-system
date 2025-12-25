@echo off
REM Contrarian Mode Test - 5000 cycles
REM Changes:
REM   1. Contrarian mode enabled (trade against consensus)
REM   2. Max hold: 20 min (from 45)
REM   3. Stop loss: 5% (from 8%)
REM   4. Take profit: 18% (from 12%)

cd /d E:\gaussian\output3
set TT_MAX_CYCLES=5000
set TT_PRINT_EVERY=100
set PAPER_TRADING=True
set MODEL_RUN_DIR=models/contrarian_test

echo ========================================
echo CONTRARIAN MODE TEST
echo ========================================
echo Config changes:
echo   - contrarian_mode: true
echo   - max_hold_minutes: 20
echo   - stop_loss: 5%%
echo   - take_profit: 18%%
echo ========================================

python scripts/train_time_travel.py
