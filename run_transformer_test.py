#!/usr/bin/env python
"""Run transformer test with proper environment variables."""
import os
import subprocess
import sys

# Set environment variables
env = os.environ.copy()
env['TEMPORAL_ENCODER'] = 'transformer'
env['NORM_TYPE'] = 'rmsnorm'
env['ACTIVATION_TYPE'] = 'geglu'
env['MODEL_RUN_DIR'] = 'models/transformer_test'
env['TT_MAX_CYCLES'] = '2000'
env['TT_PRINT_EVERY'] = '100'
env['PAPER_TRADING'] = 'True'

print("=" * 60)
print("TRANSFORMER TEST CONFIGURATION")
print("=" * 60)
print(f"TEMPORAL_ENCODER: {env.get('TEMPORAL_ENCODER')}")
print(f"NORM_TYPE: {env.get('NORM_TYPE')}")
print(f"ACTIVATION_TYPE: {env.get('ACTIVATION_TYPE')}")
print(f"MODEL_RUN_DIR: {env.get('MODEL_RUN_DIR')}")
print(f"TT_MAX_CYCLES: {env.get('TT_MAX_CYCLES')}")
print("=" * 60)
sys.stdout.flush()

# Run the training script as subprocess
result = subprocess.run(
    [sys.executable, 'scripts/train_time_travel.py'],
    cwd='E:/gaussian/output3',
    env=env
)
sys.exit(result.returncode)
