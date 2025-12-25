import csv
import re
from pathlib import Path

runs_dir = Path('models/q_optimization_xgb_fixed_with_model/runs')
results = []

for run_dir in sorted(runs_dir.iterdir()):
    if not run_dir.is_dir():
        continue
    summary = run_dir / 'SUMMARY.txt'
    if not summary.exists():
        print(f"Skip {run_dir.name} - no SUMMARY.txt")
        continue

    try:
        text = summary.read_text()

        # Parse metrics
        pnl_match = re.search(r'P&L:\s+\$([-.0-9,]+)', text)
        pnl = float(pnl_match.group(1).replace(',', '')) if pnl_match else 0.0

        fb_match = re.search(r'Final Balance:\s+\$([-.0-9,]+)', text)
        final_bal = float(fb_match.group(1).replace(',', '')) if fb_match else 0.0

        trades_match = re.search(r'Total Trades:\s+([0-9]+)', text)
        trades = int(trades_match.group(1)) if trades_match else 0

        cycles_match = re.search(r'Total Cycles:\s+([0-9]+)', text)
        cycles = int(cycles_match.group(1)) if cycles_match else 0

        wr_match = re.search(r'Win Rate:\s+([0-9.]+)%', text)
        wr = float(wr_match.group(1)) if wr_match else 0.0

        # Parse config from dirname: thr0_conf0p15_ret0
        name = run_dir.name
        parts = name.split('_')

        thr_str = parts[0].replace('thr', '').replace('m', '-').replace('p', '.')
        thr = float(thr_str) if thr_str else 0.0

        conf_str = parts[1].replace('conf', '').replace('p', '.')
        conf = float(conf_str) if conf_str else 0.0

        ret_str = parts[2].replace('ret', '').replace('p', '.')
        ret = float(ret_str) if ret_str else 0.0

        results.append({
            'thr': thr, 'min_conf': conf, 'min_abs_ret': ret,
            'pnl': pnl, 'final_balance': final_bal, 'total_trades': trades,
            'win_rate': wr,
            'pnl_per_trade': pnl/max(1,trades), 'eligible': trades >= 10
        })
        print(f"Parsed {run_dir.name}: {trades} trades, {pnl:.2f} P&L, {wr:.1f}% WR")
    except Exception as e:
        print(f"Error parsing {run_dir.name}: {e}")
        continue

# Sort by eligible first, then pnl
results.sort(key=lambda r: (r['eligible'], r['pnl'], -r['total_trades']), reverse=True)

# Write leaderboard
outfile = Path('models/q_optimization_xgb_fixed_with_model/leaderboard.csv')
with outfile.open('w', newline='') as f:
    fieldnames = ['pnl', 'final_balance', 'total_trades', 'win_rate', 'pnl_per_trade', 'eligible', 'thr', 'min_conf', 'min_abs_ret']
    w = csv.DictWriter(f, fieldnames=fieldnames)
    w.writeheader()
    w.writerows(results)

print(f'\nWrote {len(results)} results to {outfile}')
print(f'Best eligible: {results[0]["pnl"]:.2f} P&L, {results[0]["total_trades"]} trades, {results[0]["win_rate"]:.1f}% WR')
print(f'Config: thr={results[0]["thr"]}, conf={results[0]["min_conf"]}, ret={results[0]["min_abs_ret"]}')
