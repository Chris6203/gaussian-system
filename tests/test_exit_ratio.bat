@echo off
cd /d E:\gaussian\output3
set ENTRY_CONTROLLER=bandit
set MODEL_RUN_DIR=models/exit_test_2pct_4pct
set TT_MAX_CYCLES=1000
set TT_PRINT_EVERY=100
set PAPER_TRADING=True
python scripts/train_time_travel.py
