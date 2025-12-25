@echo off
echo ============================================================
echo Phase 13: PnL Calibration Gate Test (5K cycles)
echo ============================================================

set PNL_CAL_GATE=1
set PNL_CAL_MIN_PROB=0.40
set PNL_CAL_MIN_SAMPLES=30
set MODEL_RUN_DIR=models/pnl_cal_test_5k
set TT_MAX_CYCLES=5000
set TT_PRINT_EVERY=500
set PAPER_TRADING=True

echo PNL_CAL_GATE=%PNL_CAL_GATE%
echo PNL_CAL_MIN_PROB=%PNL_CAL_MIN_PROB%
echo PNL_CAL_MIN_SAMPLES=%PNL_CAL_MIN_SAMPLES%
echo MODEL_RUN_DIR=%MODEL_RUN_DIR%
echo TT_MAX_CYCLES=%TT_MAX_CYCLES%
echo ============================================================

cd /d E:\gaussian\output3
python scripts/train_time_travel.py
