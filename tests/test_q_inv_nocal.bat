@echo off
REM Test Q-scorer with: Inversion ON, Calibration OFF
echo ============================================
echo Q-Scorer Test: Inversion ON, Calibration OFF
echo ============================================

set ENTRY_CONTROLLER=q_scorer
set Q_SCORER_MODEL_PATH=models/q_scorer_bs_full.pt
set Q_SCORER_METADATA_PATH=models/q_scorer_bs_full_metadata.json
set Q_INVERT_FIX=1
set Q_DISABLE_CALIBRATION=1
set ENTRY_Q_THRESHOLD=0
set MODEL_RUN_DIR=models/q_test_inv_nocal
set TT_MAX_CYCLES=200
set TT_PRINT_EVERY=50
set PAPER_TRADING=True
set TT_DISABLE_TRADABILITY_GATE=1

python scripts/train_time_travel.py

echo.
echo ============================================
echo Test Complete - Check models/q_test_inv_nocal/SUMMARY.txt
echo ============================================
