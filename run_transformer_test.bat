@echo off
set TEMPORAL_ENCODER=transformer
set NORM_TYPE=rmsnorm
set ACTIVATION_TYPE=geglu
set MODEL_RUN_DIR=models/transformer_test
set TT_MAX_CYCLES=2000
set TT_PRINT_EVERY=100
set PAPER_TRADING=True

echo Testing Transformer Architecture
echo TEMPORAL_ENCODER=%TEMPORAL_ENCODER%
echo NORM_TYPE=%NORM_TYPE%
echo ACTIVATION_TYPE=%ACTIVATION_TYPE%

python scripts/train_time_travel.py
