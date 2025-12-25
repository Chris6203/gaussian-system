import os
import sys
import subprocess

os.chdir('E:/gaussian/output3')
env = os.environ.copy()

# Test with ALL improvements enabled
env['ENTRY_CONTROLLER'] = 'consensus'
env['MODEL_RUN_DIR'] = 'models/all_improvements_test'
env['TT_MAX_CYCLES'] = '100'
env['TT_PRINT_EVERY'] = '20'
env['PAPER_TRADING'] = 'True'
env['TT_MAX_HOLD_MINUTES'] = '45'

# Enable all new approaches
env['CONSENSUS_ENABLE_MEAN_REVERSION'] = '1'
env['CONSENSUS_ENABLE_REGIME_FILTER'] = '1'
env['CONSENSUS_ENABLE_ADVANCED_FEATURES'] = '1'
env['XGBOOST_MIN_PROFIT_EXIT'] = '3.0'

print("=" * 60)
print("TESTING ALL IMPROVEMENTS")
print("=" * 60)
print(f"ENTRY_CONTROLLER: {env.get('ENTRY_CONTROLLER')}")
print(f"MEAN_REVERSION: {env.get('CONSENSUS_ENABLE_MEAN_REVERSION')}")
print(f"REGIME_FILTER: {env.get('CONSENSUS_ENABLE_REGIME_FILTER')}")
print(f"ADVANCED_FEATURES: {env.get('CONSENSUS_ENABLE_ADVANCED_FEATURES')}")
print(f"MIN_PROFIT_EXIT: {env.get('XGBOOST_MIN_PROFIT_EXIT')}")
print("=" * 60)

result = subprocess.run(
    [sys.executable, 'scripts/train_time_travel.py'],
    env=env,
    cwd='E:/gaussian/output3'
)
sys.exit(result.returncode)
