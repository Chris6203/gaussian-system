import os
import sys
import subprocess

os.chdir('E:/gaussian/output3')
env = os.environ.copy()
env['ENTRY_CONTROLLER'] = 'bandit'
env['MODEL_RUN_DIR'] = 'models/arch_fix_test'
env['TT_MAX_CYCLES'] = '500'
env['TT_PRINT_EVERY'] = '50'
env['PAPER_TRADING'] = 'True'

result = subprocess.run(
    [sys.executable, 'scripts/train_time_travel.py'],
    env=env,
    cwd='E:/gaussian/output3'
)
sys.exit(result.returncode)
