@echo off
cd /d E:\gaussian-system

echo ========================================
echo TCN + SKEW EXITS + CONDOR REGIME FEATURES
echo ========================================
echo.
echo This test uses Iron Condor logic as weights
echo in the neural network (21 features vs 18):
echo - condor_suitability: 0=trending, 1=neutral
echo - mtf_consensus: multi-timeframe consensus
echo - trending_signal_count: count of trend signals
echo.
echo The NN learns optimal weighting for these!
echo ========================================
echo.

REM ENABLE CONDOR FEATURES - This is the key flag!
set CONDOR_FEATURES_ENABLED=1

REM Base config - TCN + Skew Exits (user's observed +431% config)
set TEMPORAL_ENCODER=tcn
set SKEW_EXIT_ENABLED=1
set SKEW_EXIT_MODE=partial

REM Standard test params - FRESH model directory (no old 18-feature checkpoints)
set MODEL_RUN_DIR=models/tcn_skew_condor_v1
set TT_MAX_CYCLES=5000
set TT_PRINT_EVERY=500
set PAPER_TRADING=True

python scripts/train_time_travel.py

echo.
echo ========================================
echo TEST COMPLETE - Check models/tcn_skew_condor_v1
echo ========================================
pause
