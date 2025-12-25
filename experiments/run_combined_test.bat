@echo off
cd /d E:\gaussian\output3

set ENTRY_CONTROLLER=consensus
set TT_MAX_CYCLES=1000
set TT_PRINT_EVERY=100
set PAPER_TRADING=True
set MODEL_RUN_DIR=models\combined_v3
set TT_MAX_HOLD_MINUTES=45

python scripts/train_time_travel.py
