import itertools
import copy
import time
import os
import numpy as np
import pandas as pd
import datetime
from core.backtest_engine import run_backtest_headless
from core.config import StrategyConfig, RunConfig
from tabulate import tabulate

# ==========================================================
# OPTIMIZATION MATRIX (Edit START, STOP, STEP here)
# ==========================================================
# Format: "param_name": np.arange(start, stop + step, step)
# Use np.arange for floats, range for ints if needed
# ==========================================================
OPTIMIZATION_MATRIX = {
    # Phase A: Exits (Optimized to Ratio: Net Profit / Max DD)
    "profit_take_pct": np.arange(0.10, 1.0, 0.2),
    #"loss_close_multiple": np.arange(1.0, 5.0, 0.2),
    
    # Phase B: Entry & Structure
    #"dte_min": range(7, 46, 7),                        # e.g. 7, 14, 21...
    #"dte_max": range(14, 91, 14),
    #"target_short_delta_low": np.arange(0.05, 0.26, 0.05),
    #"target_short_delta_high": np.arange(0.10, 0.36, 0.05),
    #"wing_width_min": np.arange(5.0, 15.1, 5.0),
    
    # Phase C: Filters
    #"iv_rank_min": np.arange(15, 55, 5),
    #"vix_threshold": np.arange(15, 55, 5),
}

def run_optimization(base_s_cfg: StrategyConfig, run_cfg: RunConfig):
    """
    Run phased serial optimization with time estimation.
    """
    print("\n" + "="*50)
    print(" HIGH-FIDELITY SERIAL OPTIMIZER")
    print(" Target: Maximize Net Profit / Max Drawdown")
    print("="*50 + "\n")

    # 1. Automatic Benchmarking & Data Loading
    print("[1/3] Loading Data & Running Hardware Benchmark...")
    
    # Pre-load data once
    from data_factory.sync_engine import MTFSyncEngine
    
    csv_path = os.path.join("reports", base_s_cfg.underlying, f"{base_s_cfg.underlying}_5.csv")
    full_df = pd.read_csv(csv_path, parse_dates=["timestamp"])
    full_df['timestamp'] = pd.to_datetime(full_df['timestamp']).dt.tz_localize(None)
    full_df.set_index("timestamp", inplace=True)
    full_df.sort_index(inplace=True)
    
    if run_cfg.backtest_samples and run_cfg.backtest_samples > 0:
        if len(full_df) > run_cfg.backtest_samples:
            full_df = full_df.iloc[-run_cfg.backtest_samples:]
            
    options_path = os.path.join("data", "synthetic_options", f"{base_s_cfg.underlying}_5min.csv")
    options_df = pd.read_csv(options_path, parse_dates=["date", "expiration"])
    preloaded_options = {}
    for date, group in options_df.groupby('date'):
        preloaded_options[date.date()] = group.to_dict('records')

    # Pre-load MTF Sync Engine if requested
    preloaded_sync = None
    if run_cfg.use_mtf:
        preloaded_sync = MTFSyncEngine(base_s_cfg.underlying, run_cfg.mtf_timeframes)

    bench_start = time.time()
    _ = run_backtest_headless(base_s_cfg, run_cfg, 
                              preloaded_df=full_df, 
                              preloaded_options=preloaded_options,
                              preloaded_sync=preloaded_sync,
                              verbose=True)
    bench_end = time.time()
    baseline_duration = bench_end - bench_start
    print(f"  -> Baseline Backtest (Cached): {baseline_duration:.2f} seconds\n")

    # 2. Build Grid
    keys = list(OPTIMIZATION_MATRIX.keys())
    values = list(OPTIMIZATION_MATRIX.values())
    combinations = list(itertools.product(*values))
    total_combos = len(combinations)
    
    estimated_total_sec = baseline_duration * total_combos
    estimated_min = estimated_total_sec / 60
    
    print(f"[2/3] Optimization Plan:")
    print(f"  -> Parameters: {', '.join(keys)}")
    print(f"  -> Total Combinations: {total_combos}")
    print(f"  -> Estimated Time: {estimated_min:.1f} minutes")
    print("-" * 30)
    
    confirm = input("Proceed with optimization? (y/n): ")
    if confirm.lower() != 'y':
        print("Aborted.")
        return

    # 3. Execution
    print(f"\n[3/3] Executing Grid Search...")
    results = []
    
    for i, combo in enumerate(combinations):
        s_cfg = copy.deepcopy(base_s_cfg)
        params_dict = dict(zip(keys, combo))
        
        # Apply params
        for k, v in params_dict.items():
            setattr(s_cfg, k, v)
            
        print(f"  [{i+1}/{total_combos}] Running...", end="\r")
        
        # Run Backtest (Silent mode for performance)
        strat = run_backtest_headless(s_cfg, run_cfg, 
                                      preloaded_df=full_df, 
                                      preloaded_options=preloaded_options,
                                      preloaded_sync=preloaded_sync,
                                      verbose=False)
        
        if strat is not None:
            net_profit = strat.pnl
            final_balance = run_cfg.backtest_cash + net_profit
            max_dd = max(strat.drawdowns) if strat.drawdowns else 0.0
            
            # Professional Metrics
            wins = [t for t in strat.trade_log if t["result"] == "win"]
            losses = [t for t in strat.trade_log if t["result"] == "loss"]
            
            gross_profit = sum(t["amount"] for t in wins)
            gross_loss = abs(sum(t["amount"] for t in losses))
            
            # Profit Factor
            profit_factor = gross_profit / gross_loss if gross_loss > 0 else (gross_profit if gross_profit > 0 else 0.0)
            
            win_rate = (len(wins) / strat.trades * 100) if strat.trades > 0 else 0
            loss_rate = 100 - win_rate
            
            avg_win = sum(t["amount"] for t in wins) / len(wins) if wins else 0
            avg_loss = abs(sum(t["amount"] for t in losses)) / len(losses) if losses else 0
            expectancy = ((avg_win * win_rate/100) - (avg_loss * loss_rate/100))
            
            # Primary Sorting Ratio: Net Profit / Max Drawdown
            np_dd_ratio = net_profit / max_dd if max_dd > 0 else 0.0
            
            # Annualized Sharpe (Assuming 5-min bars: 78/day * 252 days)
            sharpe = 0.0
            if len(strat.equity_series) > 1:
                equity_curve = np.array(strat.equity_series)
                returns = np.diff(equity_curve) / equity_curve[:-1]
                if np.std(returns) > 0:
                    sharpe = (np.mean(returns) / np.std(returns)) * np.sqrt(252 * 78)

            results.append({
                "params": params_dict,
                "net_profit": net_profit,
                "final_balance": final_balance,
                "max_dd": max_dd,
                "np_dd_ratio": np_dd_ratio,
                "sharpe": sharpe,
                "profit_factor": profit_factor,
                "expectancy": expectancy,
                "trades": strat.trades,
                "wins": len(wins),
                "losses": len(losses),
                "win_rate": win_rate
            })

    print(f"\n\nDone. Processing {len(results)} valid results...")

    # 4. Reporting (Top 100)
    sorted_results = sorted(results, key=lambda x: x['np_dd_ratio'], reverse=True)
    top_100 = sorted_results[:100]
    
    # 4.1 Persistence (New: Save to CSV)
    if top_100:
        now_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = os.path.join("reports", f"top100_{now_str}.csv")
        os.makedirs("reports", exist_ok=True)
        
        # Flatten and save
        csv_data = []
        for rank, res in enumerate(top_100, 1):
            row = {"Rank": rank}
            row.update(res['params'])
            row.update({
                "Balance": res['final_balance'],
                "NetProfit": res['net_profit'],
                "MaxDD": res['max_dd'],
                "NP_DD_Ratio": res['np_dd_ratio'],
                "ProfitFactor": res['profit_factor'],
                "Expectancy": res['expectancy'],
                "Sharpe": res['sharpe'],
                "Trades": res['trades'],
                "Wins": res['wins'],
                "Losses": res['losses'],
                "WinRate": res['win_rate']
            })
            csv_data.append(row)
        
        pd.DataFrame(csv_data).to_csv(report_path, index=False)
        print(f"\n[Report Saved] {report_path}")
    
    table_headers = ["Rank", "Params", "Profit", "DD", "NP/DD", "PF", "Exp.", "Sharpe", "T", "W/L", "Win%"]
    table_rows = []
    
    for rank, res in enumerate(top_100, 1):
        p = res['params']
        param_summary = ", ".join([f"{k}={v}" for k, v in p.items()])
        table_rows.append([
            rank,
            param_summary,
            f"${res['net_profit']:,.0f}",
            f"${res['max_dd']:,.0f}",
            f"{res['np_dd_ratio']:.2f}",
            f"{res['profit_factor']:.2f}",
            f"${res['expectancy']:.2f}",
            f"{res['sharpe']:.2f}",
            res['trades'],
            f"{res['wins']}/{res['losses']}",
            f"{res['win_rate']:.1f}%"
        ])

    print("\n" + "="*120)
    print(f" TOP {len(top_100)} OPTIMIZATION RESULTS (Ranked by Profit/DD)")
    print("="*120)
    print(tabulate(table_rows, headers=table_headers, tablefmt="simple"))
    print("="*120 + "\n")

    # 5. Apply Choice
    if sorted_results:
        print("Selection Interface:")
        print("  - Enter '1' to '100' to apply a specific configuration.")
        print("  - Enter 'n' to quit without applying.")
        
        choice = input("\nSelect Rank to apply: ")
        if choice.isdigit():
            idx = int(choice) - 1
            if 0 <= idx < len(sorted_results):
                chosen = sorted_results[idx]
                print(f"APPLYING CONFIG #{choice}: {chosen['params']}")
                apply_best_params(chosen['params'])
                print("Configuration updated in core/config.template.py.")
            else:
                print("Invalid rank.")
        else:
            print("Selection ignored.")

def apply_params_to_file(params, config_path):
    """Update a specific file with parameters."""
    if not os.path.exists(config_path):
        return False
        
    with open(config_path, 'r') as f:
        lines = f.readlines()
        
    new_lines = []
    for line in lines:
        updated = False
        # We only want to replace lines that look like: key = val or key: type = val
        stripped = line.strip()
        if "=" in stripped:
            first_part = stripped.split("=")[0].strip()
            # Handle type hints: "dte_min: int" -> "dte_min"
            key_candidate = first_part.split(":")[0].strip()
            
            if key_candidate in params:
                val = params[key_candidate]
                indent = line[:line.find(line.strip())]
                parts = line.split("#")
                comment = "#" + parts[1] if len(parts) > 1 else ""
                
                # Reconstruct with original formatting / type hints
                if ":" in first_part:
                    type_part = first_part.split(":")[1]
                    new_line = f"{indent}{key_candidate}: {type_part} = {val}  {comment}\n"
                else:
                    new_line = f"{indent}{key_candidate} = {val}  {comment}\n"
                
                new_lines.append(new_line)
                updated = True
        
        if not updated:
            new_lines.append(line)
            
    with open(config_path, 'w') as f:
        f.writelines(new_lines)
    return True

def apply_best_params(params):
    """
    Update both core/config.template.py and core/config.py with the chosen parameters.
    """
    targets = ["core/config.template.py", "core/config.py"]
    for t in targets:
        success = apply_params_to_file(params, t)
        if success:
            print(f"  -> Updated: {t}")
