@echo off
echo ============================================
echo Verification Test - Exit Logic Fix
echo ============================================

set ENTRY_CONTROLLER=bandit
set TT_MAX_CYCLES=150
set TT_PRINT_EVERY=30
set PAPER_TRADING=True
set MODEL_RUN_DIR=models/verify_fix

python scripts/train_time_travel.py

echo.
echo ============================================
echo Test Complete
echo ============================================
type models\verify_fix\SUMMARY.txt
