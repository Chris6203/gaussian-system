@echo off
set ENTRY_CONTROLLER=bandit
set TT_MAX_CYCLES=100
set TT_PRINT_EVERY=20
set PAPER_TRADING=True
set TT_DISABLE_TRADABILITY_GATE=1
set MODEL_RUN_DIR=models/tp_fix_test
cd /d E:\gaussian\output3
python scripts/train_time_travel.py
