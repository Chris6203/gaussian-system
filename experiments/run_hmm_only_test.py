import os
import sys
import subprocess

os.chdir('E:/gaussian/output3')
env = os.environ.copy()
env['ENTRY_CONTROLLER'] = 'bandit'
env['MODEL_RUN_DIR'] = 'models/hmm_only_test'
env['TT_MAX_CYCLES'] = '1000'
env['TT_PRINT_EVERY'] = '100'
env['PAPER_TRADING'] = 'True'

result = subprocess.run(
    [sys.executable, 'scripts/train_time_travel.py'],
    env=env,
    cwd='E:/gaussian/output3'
)
sys.exit(result.returncode)
