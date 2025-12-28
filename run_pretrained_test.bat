@echo off
cd /d E:\gaussian-system
echo Setting environment variables...
set LOAD_PRETRAINED=1
set MODEL_RUN_DIR=models/jerry_baseline_pretrained
set TT_MAX_CYCLES=5000
set TT_PRINT_EVERY=500
set PAPER_TRADING=True
echo Starting training...
python scripts/train_time_travel.py
echo Training complete.
