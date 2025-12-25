#!/usr/bin/env python3
"""Combined Test: PnL Calibration Gate + Feature Attribution

Tests both Phase 13 (PnL Cal Gate) and Phase 14 (Feature Attribution) together.
This should be the best configuration.
"""

import os
import sys
import subprocess

# Set environment variables for the subprocess
env = os.environ.copy()

# Enable PnL Calibration Gate (Phase 13)
env['PNL_CAL_GATE'] = '1'
env['PNL_CAL_MIN_PROB'] = '0.40'
env['PNL_CAL_MIN_SAMPLES'] = '30'

# Feature Attribution is automatic when RLThresholdLearner is enabled (Phase 14)

env['MODEL_RUN_DIR'] = 'models/combined_test_5k'
env['TT_MAX_CYCLES'] = '5000'
env['TT_PRINT_EVERY'] = '500'
env['PAPER_TRADING'] = 'True'

print("=" * 60)
print("Combined Test: PnL Cal Gate + Feature Attribution (5K cycles)")
print("=" * 60)
print(f"PNL_CAL_GATE = 1 (enabled)")
print(f"PNL_CAL_MIN_PROB = 0.40 (40% min P(profit))")
print(f"PNL_CAL_MIN_SAMPLES = 30 (learn first 30 trades)")
print(f"Feature Attribution = via RLThresholdLearner")
print(f"MODEL_RUN_DIR = models/combined_test_5k")
print("=" * 60)
sys.stdout.flush()

# Run the training script as a subprocess with proper environment
result = subprocess.run(
    [sys.executable, 'scripts/train_time_travel.py'],
    cwd='E:/gaussian/output3',
    env=env
)
sys.exit(result.returncode)
