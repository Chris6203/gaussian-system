@echo off
E:
cd \gaussian-system
set MAX_CONFIDENCE=0.25
echo.
echo ============================================
echo   LOW CONFIDENCE STRATEGY BOT STARTING
echo   MAX_CONFIDENCE = %MAX_CONFIDENCE%
echo ============================================
echo.
python go_live_only.py models\combo_dow_validation
