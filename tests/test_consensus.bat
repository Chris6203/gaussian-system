@echo off
set ENTRY_CONTROLLER=consensus
set MODEL_RUN_DIR=models/consensus_test_v2
set TT_MAX_CYCLES=100
set TT_PRINT_EVERY=10
set PAPER_TRADING=True
python scripts/train_time_travel.py 2>&1
