@echo off
cd /d E:\gaussian-system
set TEMPORAL_ENCODER=transformer
set MODEL_RUN_DIR=models/transformer_jan2_test
set TT_MAX_CYCLES=2000
set TT_PRINT_EVERY=200
set PAPER_TRADING=True
set TRAINING_START_DATE=2025-12-15
set TRAINING_END_DATE=2025-12-31
echo Starting transformer test...
python scripts/train_time_travel.py
