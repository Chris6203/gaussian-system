#!/usr/bin/env python3
"""20K test for Phase 12 Regime Filter V3"""

import os
import sys

# Set environment variables BEFORE importing anything
os.environ['TT_REGIME_FILTER'] = '3'  # Use full RegimeFilter
os.environ['REGIME_MIN_QUALITY'] = '0.35'
os.environ['MODEL_RUN_DIR'] = 'models/regime_filter_20k'
os.environ['TT_MAX_CYCLES'] = '20000'
os.environ['TT_PRINT_EVERY'] = '1000'
os.environ['PAPER_TRADING'] = 'True'

print("=" * 60)
print("Phase 12: Regime Filter V3 - 20K Validation Test")
print("=" * 60)
print(f"TT_REGIME_FILTER = {os.environ.get('TT_REGIME_FILTER')}")
print(f"REGIME_MIN_QUALITY = {os.environ.get('REGIME_MIN_QUALITY')}")
print(f"MODEL_RUN_DIR = {os.environ.get('MODEL_RUN_DIR')}")
print(f"TT_MAX_CYCLES = {os.environ.get('TT_MAX_CYCLES')}")
print("=" * 60)

# Change to the right directory
os.chdir('E:/gaussian/output3')

# Use runpy to run the script properly
import runpy
sys.argv = ['train_time_travel.py']
runpy.run_path('scripts/train_time_travel.py', run_name='__main__')
