@echo off
cd /d E:\gaussian-system
set TEMPORAL_ENCODER=transformer
set MODEL_RUN_DIR=models/transformer_jan2026_test
set TT_MAX_CYCLES=5000
set TT_PRINT_EVERY=500
set PAPER_TRADING=True
set TRAINING_START_DATE=2025-12-01
set TRAINING_END_DATE=2025-12-31
echo Starting transformer encoder test...
echo TEMPORAL_ENCODER=%TEMPORAL_ENCODER%
python scripts/train_time_travel.py
