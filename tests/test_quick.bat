@echo off
REM Quick test of exit logic fix
echo ============================================
echo Quick Exit Fix Test - 100 cycles
echo ============================================

set ENTRY_CONTROLLER=bandit
set TT_MAX_CYCLES=100
set TT_PRINT_EVERY=25
set PAPER_TRADING=True
set MODEL_RUN_DIR=models/quick_test

python scripts/train_time_travel.py

echo.
echo ============================================
echo Test Complete - Check models/quick_test/SUMMARY.txt
echo ============================================
