import os
import sys
import subprocess

os.chdir('E:/gaussian/output3')
env = os.environ.copy()
env['ENTRY_CONTROLLER'] = 'bandit'
env['MODEL_RUN_DIR'] = 'models/baseline_low_friction'
env['TT_MAX_CYCLES'] = '1000'
env['TT_PRINT_EVERY'] = '100'
env['PAPER_TRADING'] = 'True'
# NO HMM_PURE_MODE - use default entry logic

print('Running BASELINE (no HMM-pure) with reduced friction...')
result = subprocess.run(
    [sys.executable, 'scripts/train_time_travel.py'],
    env=env,
    cwd='E:/gaussian/output3'
)
sys.exit(result.returncode)
