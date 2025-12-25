@echo off
set MODEL_RUN_DIR=models/enhanced_features_test
set PAPER_TRADING=True
set TT_MAX_CYCLES=5000
set TT_PRINT_EVERY=100
python scripts/train_time_travel.py
