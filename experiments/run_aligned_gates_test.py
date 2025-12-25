import os
import sys
import subprocess

os.chdir('E:/gaussian/output3')
env = os.environ.copy()
env['ENTRY_CONTROLLER'] = 'bandit'
env['MODEL_RUN_DIR'] = 'models/aligned_gates_test'
env['TT_MAX_CYCLES'] = '1000'
env['TT_PRINT_EVERY'] = '100'
env['PAPER_TRADING'] = 'True'

print('Running test with aligned gates (55% conf, 0.5% return)...')
result = subprocess.run(
    [sys.executable, 'scripts/train_time_travel.py'],
    env=env,
    cwd='E:/gaussian/output3'
)
sys.exit(result.returncode)
