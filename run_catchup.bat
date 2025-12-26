@echo off
cd /d E:\gaussian-system
set TRAINING_START_DATE=2025-12-26
set TRAINING_END_DATE=2025-12-26
set MODEL_RUN_DIR=models\catchup
set TT_MAX_CYCLES=60
set TT_PRINT_EVERY=10
set PAPER_TRADING=True
python scripts/train_time_travel.py
