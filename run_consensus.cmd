@echo off
cd /d E:\gaussian\output3
set ENTRY_CONTROLLER=consensus
set TT_MAX_CYCLES=500
set TT_PRINT_EVERY=50
set PAPER_TRADING=True
set MODEL_RUN_DIR=models/consensus_test

echo ============================================================
echo CONSENSUS CONTROLLER TEST
echo ============================================================
echo ENTRY_CONTROLLER = %ENTRY_CONTROLLER%
echo TT_MAX_CYCLES = %TT_MAX_CYCLES%
echo ============================================================

python scripts/train_time_travel.py
