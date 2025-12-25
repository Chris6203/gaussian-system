@echo off
cd /d E:\gaussian\output3
echo ============================================
echo Testing Exit Logic Fix (Baseline)
echo ============================================

set MODEL_RUN_DIR=models/exit_fix_test
set TT_MAX_CYCLES=500
set TT_PRINT_EVERY=100
set PAPER_TRADING=True

python scripts/train_time_travel.py

echo.
echo ============================================
echo Test Complete
echo ============================================
if exist "models\exit_fix_test\SUMMARY.txt" (
    type models\exit_fix_test\SUMMARY.txt
) else (
    echo ERROR: SUMMARY.txt not found
)
