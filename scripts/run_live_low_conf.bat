@echo off
REM ============================================================================
REM LIVE BOT - Low Confidence Strategy (51.7% WR, +74.30% P&L)
REM ============================================================================
REM Data analysis shows: Low confidence < 25% has HIGHER win rate than high conf
REM This is the inverted confidence filter strategy
REM
REM To switch to LIVE trading: create 'go_live.flag' file in this directory
REM To stay in PAPER mode: ensure 'go_live.flag' does not exist
REM ============================================================================

echo.
echo ============================================
echo   LOW CONFIDENCE STRATEGY - LIVE BOT
echo ============================================
echo   MAX_CONFIDENCE = 0.25 (only low conf trades)
echo   Expected: ~51.7%% WR, +74%% P/L
echo ============================================
echo.

REM Set the inverted confidence filter
set MAX_CONFIDENCE=0.25

REM Optional: Day of week filter (skip Mon/Fri for even better results)
REM Uncomment these for the full combo_dow strategy:
REM set DAY_OF_WEEK_FILTER=1
REM set SKIP_MONDAY=1
REM set SKIP_FRIDAY=1

REM Run the live bot
REM Usage: run_live_low_conf.bat [model_directory]
REM Example: run_live_low_conf.bat models/run_20260104_120000

if "%1"=="" (
    echo ERROR: Please specify a model directory
    echo Usage: run_live_low_conf.bat models/run_YYYYMMDD_HHMMSS
    echo.
    echo Available models:
    dir /b models\run_* 2>nul
    pause
    exit /b 1
)

python go_live_only.py %1

pause
