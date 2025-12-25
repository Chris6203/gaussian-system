@echo off
cd /d E:\gaussian\output3
set ENTRY_CONTROLLER=rl
set MODEL_RUN_DIR=models/best_rl_35pct_win
set TT_MAX_CYCLES=2000
set TT_PRINT_EVERY=100
set PAPER_TRADING=True
set TT_REGIME_FILTER=1
echo Starting with TT_REGIME_FILTER=%TT_REGIME_FILTER%
python scripts/train_time_travel.py
echo Completed with errorlevel %errorlevel%
