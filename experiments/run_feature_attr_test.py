#!/usr/bin/env python3
"""Phase 14: Feature Attribution Test

Tests the RLThresholdLearner feature attribution system.
Tracks which of the 16 features are most predictive of winning trades.
"""

import os
import sys
import subprocess

# Set environment variables for the subprocess
env = os.environ.copy()

# Enable RLThresholdLearner (required for feature attribution)
# Set via config.json adaptive_learning.rl_threshold_learner.enabled = true
# Or ensure it's enabled by default

env['MODEL_RUN_DIR'] = 'models/feature_attr_test_5k'
env['TT_MAX_CYCLES'] = '5000'
env['TT_PRINT_EVERY'] = '500'
env['PAPER_TRADING'] = 'True'

print("=" * 60)
print("Phase 14: Feature Attribution Test (5K cycles)")
print("=" * 60)
print(f"MODEL_RUN_DIR = models/feature_attr_test_5k")
print(f"Feature attribution is computed via gradient saliency")
print(f"Tracks which features predict winners vs losers")
print("=" * 60)
sys.stdout.flush()

# Run the training script as a subprocess with proper environment
result = subprocess.run(
    [sys.executable, 'scripts/train_time_travel.py'],
    cwd='E:/gaussian/output3',
    env=env
)
sys.exit(result.returncode)
