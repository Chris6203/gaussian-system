# core/benchmark_cpu.py
import time
import psutil
import os
import sys
from core.backtest_engine import run_backtest_headless
from core.config import StrategyConfig, RunConfig

def benchmark():
    print("==============================================")
    print(" HARDWARE & WORKLOAD BENCHMARK")
    print("==============================================\n")

    # 1. System Info
    cpu_count = psutil.cpu_count(logical=False)
    logical_count = psutil.cpu_count(logical=True)
    mem = psutil.virtual_memory()
    
    print(f"Processor: {cpu_count} Physical Cores / {logical_count} Logical Processors")
    print(f"Total RAM: {mem.total / (1024**3):.2f} GB")
    print(f"Available RAM: {mem.available / (1024**3):.2f} GB")
    print("-" * 40)

    # 2. Workload Test
    s_cfg = StrategyConfig()
    r_cfg = RunConfig()
    r_cfg.backtest_samples = 5000  # Standard test slice
    
    print(f"Starting sample backtest (5,000 bars)...")
    start_time = time.time()
    
    # Run once to measure baseline
    run_backtest_headless(s_cfg, r_cfg)
    
    end_time = time.time()
    duration = end_time - start_time
    
    print(f"\nBenchmark Result:")
    print(f"Time per 5,000 bars: {duration:.2f} seconds")
    
    # 3. Projection
    # Assuming 100,000 combinations in a full grid (hypothetical)
    projected_serial = (duration * 100) / 60  # For 100 runs in minutes
    print(f"Projected time for 100 backtests (Serial): {projected_serial:.2f} minutes")
    
    if physical_cores := cpu_count:
        projected_parallel = projected_serial / physical_cores
        print(f"Projected time for 100 backtests (Parallel - {physical_cores} cores): {projected_parallel:.2f} minutes")

    print("\nRecommendation:")
    if cpu_count <= 2:
        print("-> Use 2 parallel workers. This will saturate physical cores without locking the OS.")
    else:
        print(f"-> Use {cpu_count - 1} parallel workers.")

if __name__ == "__main__":
    benchmark()
