@echo off
set ENTRY_CONTROLLER=q_scorer
set Q_INVERT_FIX=1
set Q_SCORER_MODEL_PATH=models/q_scorer_bs_full.pt
set Q_SCORER_METADATA_PATH=models/q_scorer_bs_full_metadata.json
set ENTRY_Q_THRESHOLD=0
set MODEL_RUN_DIR=models/q_final_verification
set TT_MAX_CYCLES=200
set TT_PRINT_EVERY=20
set PAPER_TRADING=True
set TT_DISABLE_TRADABILITY_GATE=1

echo [DEBUG] ENTRY_CONTROLLER is set to: %ENTRY_CONTROLLER%
echo [DEBUG] Q_INVERT_FIX is set to: %Q_INVERT_FIX%
echo [DEBUG] TT_DISABLE_TRADABILITY_GATE is set to: %TT_DISABLE_TRADABILITY_GATE%
echo [DEBUG] About to run Python script...

python scripts/train_time_travel.py
