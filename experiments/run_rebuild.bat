@echo off
cd /d E:\gaussian\output3
set TT_Q_LABELS=all
set Q_HORIZON_MINUTES=15
set TT_MAX_CYCLES=5000
set TT_PRINT_EVERY=500
set PAPER_TRADING=True
set MODEL_RUN_DIR=models/rebuilt_dataset
echo Starting dataset generation...
echo TT_Q_LABELS=%TT_Q_LABELS%
echo TT_MAX_CYCLES=%TT_MAX_CYCLES%
python scripts/train_time_travel.py
