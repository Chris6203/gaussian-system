#!/usr/bin/env python3
"""Simple database existence check - writes results to db_check_result.txt"""
import os
from pathlib import Path

result_file = Path('db_check_result.txt')

with open(result_file, 'w') as f:
    # Check current working directory
    f.write(f"CWD: {os.getcwd()}\n")
    
    # Check various paths for the database
    paths_to_check = [
        'data/db/historical.db',
        'E:/gaussian/output3/data/db/historical.db',
        '../data/db/historical.db',
        'data/market_data_cache.db',
    ]
    
    f.write("\nChecking database paths:\n")
    for p in paths_to_check:
        exists = Path(p).exists()
        size = Path(p).stat().st_size if exists else 0
        f.write(f"  {p}: {'EXISTS' if exists else 'NOT FOUND'}")
        if exists:
            f.write(f" ({size:,} bytes)")
        f.write("\n")
    
    # List data directory contents
    f.write("\nContents of data/ directory:\n")
    data_dir = Path('data')
    if data_dir.exists():
        for item in data_dir.rglob('*'):
            f.write(f"  {item}\n")
    else:
        f.write("  data/ directory does not exist\n")

print(f"Results written to {result_file}")



