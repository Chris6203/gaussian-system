import os
import sys
import subprocess

os.chdir('E:/gaussian/output3')
result = subprocess.run(
    [sys.executable, 'scripts/train_direction_v3.py',
     '--epochs', '30',
     '--output', 'models/direction_v3_balanced.pt'],
    cwd='E:/gaussian/output3'
)
sys.exit(result.returncode)
