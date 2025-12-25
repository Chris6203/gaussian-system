@echo off
echo ============================================
echo Model Rebuild - Dataset Generation
echo ============================================
echo.
echo Setting environment variables...

set TT_Q_LABELS=all
set Q_HORIZON_MINUTES=15
set TT_MAX_CYCLES=5000
set TT_PRINT_EVERY=500
set PAPER_TRADING=True
set MODEL_RUN_DIR=models/rebuilt_dataset

echo TT_Q_LABELS=%TT_Q_LABELS%
echo Q_HORIZON_MINUTES=%Q_HORIZON_MINUTES%
echo TT_MAX_CYCLES=%TT_MAX_CYCLES%
echo MODEL_RUN_DIR=%MODEL_RUN_DIR%
echo.

python scripts/train_time_travel.py

echo.
echo ============================================
echo Dataset Generation Complete
echo ============================================
echo.
echo Checking generated data...
if exist "data\missed_opportunities_full.jsonl" (
    for /f %%a in ('find /c /v "" ^< "data\missed_opportunities_full.jsonl"') do echo Missed opportunities records: %%a
) else (
    echo ERROR: missed_opportunities_full.jsonl not found!
)
