"""
Test V3 Direction Predictor with HMM Alignment Filter.

This version only trades when V3 prediction matches HMM trend:
- BUY_CALLS only when HMM is bullish (>0.6)
- BUY_PUTS only when HMM is bearish (<0.4)
"""

import os
import sys
import subprocess

os.chdir('E:/gaussian/output3')
env = os.environ.copy()

# Use V3 direction controller with HMM filter
env['ENTRY_CONTROLLER'] = 'v3_direction'
env['V3_MODEL_PATH'] = 'models/direction_v3.pt'
env['V3_MIN_CONFIDENCE'] = '0.52'

# Standard time-travel settings
env['MODEL_RUN_DIR'] = 'models/v3_hmm_test'
env['TT_MAX_CYCLES'] = '1000'
env['TT_PRINT_EVERY'] = '100'
env['PAPER_TRADING'] = 'True'

print('=' * 60)
print('V3 + HMM ALIGNMENT FILTER TEST')
print('=' * 60)
print('Model: models/direction_v3.pt (56% val acc)')
print('Filter: Only trade when V3 agrees with HMM trend')
print('  - CALLS only if HMM > 0.6 (bullish)')
print('  - PUTS only if HMM < 0.4 (bearish)')
print('=' * 60)

result = subprocess.run(
    [sys.executable, 'scripts/train_time_travel.py'],
    env=env,
    cwd='E:/gaussian/output3'
)
sys.exit(result.returncode)
