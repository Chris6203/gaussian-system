@echo off
cd /d E:\gaussian-system
set HARD_STOP_LOSS_PCT=50
set HARD_TAKE_PROFIT_PCT=10
set TRAIN_MAX_CONF=0.25
set DAY_OF_WEEK_FILTER=1
set SKIP_MONDAY=1
set SKIP_FRIDAY=1
set MODEL_RUN_DIR=models/combo_dow_validation
set TT_MAX_CYCLES=5000
set TT_PRINT_EVERY=500
set PAPER_TRADING=True
echo ========================================
echo COMBO_DOW VALIDATION TEST
echo Config: 50%% stop, 10%% TP, max_conf=0.25
echo Skip Monday/Friday
echo ========================================
python scripts/train_time_travel.py
