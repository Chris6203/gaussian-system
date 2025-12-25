@echo off
cd /d E:\gaussian\output3
set ENTRY_CONTROLLER=bandit
set MODEL_RUN_DIR=models/arch_fix_test
set TT_MAX_CYCLES=500
set TT_PRINT_EVERY=50
set PAPER_TRADING=True
python scripts/train_time_travel.py
