@echo off
set TT_MAX_CYCLES=20000
set TT_PRINT_EVERY=2000
set PAPER_TRADING=True
set MODEL_RUN_DIR=models/validation_20k
set REQUIRE_NEURAL_CONFIRM=1
python scripts/train_time_travel.py
