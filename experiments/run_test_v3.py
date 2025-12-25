"""
Test DirectionPredictorV3 in trading simulation.

This script runs time-travel simulation using V3 for entry decisions.
"""

import os
import sys
import subprocess

os.chdir('E:/gaussian/output3')
env = os.environ.copy()

# Use V3 direction controller
env['ENTRY_CONTROLLER'] = 'v3_direction'
env['V3_MODEL_PATH'] = 'models/direction_v3.pt'
env['V3_MIN_CONFIDENCE'] = '0.52'  # Lower threshold for testing

# Standard time-travel settings
env['MODEL_RUN_DIR'] = 'models/v3_test'
env['TT_MAX_CYCLES'] = '1000'
env['TT_PRINT_EVERY'] = '100'
env['PAPER_TRADING'] = 'True'

print('=' * 60)
print('V3 DIRECTION PREDICTOR TEST')
print('=' * 60)
print('Model: models/direction_v3.pt')
print('Validation Accuracy: 56.03%')
print('Min Confidence: 0.52')
print('=' * 60)

result = subprocess.run(
    [sys.executable, 'scripts/train_time_travel.py'],
    env=env,
    cwd='E:/gaussian/output3'
)
sys.exit(result.returncode)
