@echo off
REM Test all Q-scorer inversion/calibration combinations
REM Each test runs 100 cycles

set Q_SCORER_MODEL_PATH=models/q_scorer_bs_full.pt
set Q_SCORER_METADATA_PATH=models/q_scorer_bs_full_metadata.json
set ENTRY_CONTROLLER=q_scorer
set ENTRY_Q_THRESHOLD=0
set TT_MAX_CYCLES=100
set TT_PRINT_EVERY=100
set PAPER_TRADING=True
set TT_DISABLE_TRADABILITY_GATE=1

echo ============================================
echo TEST 1: No inversion, No calibration
echo ============================================
set Q_INVERT_FIX=0
set Q_DISABLE_CALIBRATION=1
set MODEL_RUN_DIR=models/q_test_noinv_nocal
python scripts/train_time_travel.py 2>&1 | findstr "Win Rate Final Balance Trades made"

echo.
echo ============================================
echo TEST 2: No inversion, With calibration
echo ============================================
set Q_INVERT_FIX=0
set Q_DISABLE_CALIBRATION=0
set MODEL_RUN_DIR=models/q_test_noinv_cal
python scripts/train_time_travel.py 2>&1 | findstr "Win Rate Final Balance Trades made"

echo.
echo ============================================
echo TEST 3: Inversion, No calibration
echo ============================================
set Q_INVERT_FIX=1
set Q_DISABLE_CALIBRATION=1
set MODEL_RUN_DIR=models/q_test_inv_nocal
python scripts/train_time_travel.py 2>&1 | findstr "Win Rate Final Balance Trades made"

echo.
echo ============================================
echo TEST 4: Inversion, With calibration
echo ============================================
set Q_INVERT_FIX=1
set Q_DISABLE_CALIBRATION=0
set MODEL_RUN_DIR=models/q_test_inv_cal
python scripts/train_time_travel.py 2>&1 | findstr "Win Rate Final Balance Trades made"

echo.
echo ============================================
echo ALL TESTS COMPLETE
echo ============================================
pause
