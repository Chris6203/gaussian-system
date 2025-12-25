#!/usr/bin/env python3
"""Baseline 5K test WITHOUT Regime Filter for comparison"""

import os
import sys

# NO regime filter - baseline comparison
os.environ['TT_REGIME_FILTER'] = '0'  # Disabled
os.environ['MODEL_RUN_DIR'] = 'models/baseline_no_filter_5k'
os.environ['TT_MAX_CYCLES'] = '5000'
os.environ['TT_PRINT_EVERY'] = '500'
os.environ['PAPER_TRADING'] = 'True'

print("=" * 60)
print("Baseline 5K Test - NO Regime Filter")
print("=" * 60)
print(f"TT_REGIME_FILTER = {os.environ.get('TT_REGIME_FILTER')} (disabled)")
print(f"MODEL_RUN_DIR = {os.environ.get('MODEL_RUN_DIR')}")
print(f"TT_MAX_CYCLES = {os.environ.get('TT_MAX_CYCLES')}")
print("=" * 60)

os.chdir('E:/gaussian/output3')
import runpy
sys.argv = ['train_time_travel.py']
runpy.run_path('scripts/train_time_travel.py', run_name='__main__')
