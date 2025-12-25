@echo off
echo ============================================
echo Fetching 6 months of historical 1-min data
echo ============================================
echo.

cd /d e:\gaussian\output3
python fetch_6months.py

echo.
echo ============================================
echo Check fetch_6months.log for details
echo ============================================
pause



