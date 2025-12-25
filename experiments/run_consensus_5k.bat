@echo off
cd /d E:\gaussian\output3
set TT_MAX_CYCLES=5000
set TT_PRINT_EVERY=100
set PAPER_TRADING=True
set MODEL_RUN_DIR=models/consensus_5k_test
python scripts/train_time_travel.py
