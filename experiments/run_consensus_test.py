"""
Quick test script for consensus controller
"""
import os
import sys

# Set environment variables BEFORE importing
os.environ['ENTRY_CONTROLLER'] = 'consensus'
os.environ['TT_MAX_CYCLES'] = '500'
os.environ['TT_PRINT_EVERY'] = '50'
os.environ['PAPER_TRADING'] = 'True'
os.environ['MODEL_RUN_DIR'] = 'models/consensus_test'

print("=" * 60)
print("CONSENSUS CONTROLLER TEST")
print("=" * 60)
print(f"ENTRY_CONTROLLER = {os.environ.get('ENTRY_CONTROLLER')}")
print(f"TT_MAX_CYCLES = {os.environ.get('TT_MAX_CYCLES')}")
print("=" * 60)

# Now run the training script
if __name__ == "__main__":
    # Add project root to path
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

    # Import and run
    exec(open('scripts/train_time_travel.py', encoding='utf-8').read())
