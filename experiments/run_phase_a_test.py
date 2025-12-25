import os
import sys
import subprocess

os.chdir('E:/gaussian/output3')
env = os.environ.copy()
env['ENTRY_CONTROLLER'] = 'bandit'
env['MODEL_RUN_DIR'] = 'models/phase_a_v2_retrain'
env['TT_MAX_CYCLES'] = '1000'
env['TT_PRINT_EVERY'] = '100'
env['PAPER_TRADING'] = 'True'

print('=' * 60)
print('PHASE A TEST: V2 with improved hyperparameters')
print('=' * 60)
print('Changes:')
print('  - RBF kernels: DISABLED (was enabled)')
print('  - Learning rate: 0.0003 (was 0.0001)')
print('  - Batch size: 64 (was 32)')
print('  - Direction loss weight: 3.0 (was 2.0)')
print('=' * 60)

result = subprocess.run(
    [sys.executable, 'scripts/train_time_travel.py'],
    env=env,
    cwd='E:/gaussian/output3'
)
sys.exit(result.returncode)
