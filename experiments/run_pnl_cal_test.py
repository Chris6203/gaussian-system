#!/usr/bin/env python3
"""Phase 13: PnL Calibration Gate Test

Tests the CalibrationTracker PnL-based entry gating.
The gate learns P(profit|confidence) and blocks low-probability entries.
"""

import os
import sys
import subprocess

# Set environment variables for the subprocess
env = os.environ.copy()
env['PNL_CAL_GATE'] = '1'
env['PNL_CAL_MIN_PROB'] = '0.40'
env['PNL_CAL_MIN_SAMPLES'] = '30'
env['MODEL_RUN_DIR'] = 'models/pnl_cal_test_5k'
env['TT_MAX_CYCLES'] = '5000'
env['TT_PRINT_EVERY'] = '500'
env['PAPER_TRADING'] = 'True'

print("=" * 60)
print("Phase 13: PnL Calibration Gate Test (5K cycles)")
print("=" * 60)
print(f"PNL_CAL_GATE = 1 (enabled)")
print(f"PNL_CAL_MIN_PROB = 0.40 (40% min P(profit))")
print(f"PNL_CAL_MIN_SAMPLES = 30 (learn first 30 trades)")
print(f"MODEL_RUN_DIR = models/pnl_cal_test_5k")
print("=" * 60)
sys.stdout.flush()

# Run the training script as a subprocess with proper environment
result = subprocess.run(
    [sys.executable, 'scripts/train_time_travel.py'],
    cwd='E:/gaussian/output3',
    env=env
)
sys.exit(result.returncode)
