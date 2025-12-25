@echo off
echo ============================================
echo Testing Rebuilt Q-Scorer Model
echo ============================================

set ENTRY_CONTROLLER=q_scorer
set Q_SCORER_MODEL_PATH=models/q_scorer_rebuilt.pt
set Q_SCORER_METADATA_PATH=models/q_scorer_rebuilt_metadata.json
set Q_INVERT_FIX=1
set Q_DISABLE_CALIBRATION=0
set ENTRY_Q_THRESHOLD=0
set MODEL_RUN_DIR=models/qscorer_test
set TT_MAX_CYCLES=500
set TT_PRINT_EVERY=100
set PAPER_TRADING=True
set TT_DISABLE_TRADABILITY_GATE=1

echo.
echo Using Q-scorer model: %Q_SCORER_MODEL_PATH%
echo Entry threshold: %ENTRY_Q_THRESHOLD%
echo Max cycles: %TT_MAX_CYCLES%
echo.

python scripts/train_time_travel.py

echo.
echo ============================================
echo Test Complete
echo ============================================
if exist "models\qscorer_test\SUMMARY.txt" (
    type models\qscorer_test\SUMMARY.txt
) else (
    echo ERROR: SUMMARY.txt not found
)
