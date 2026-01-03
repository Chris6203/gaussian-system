#!/usr/bin/env python3
"""Collect and summarize Phase 42 experiment results"""
import json
import os
from pathlib import Path

results = []
model_dirs = list(Path("models").glob("phase42_*"))

for model_dir in sorted(model_dirs):
    run_info = model_dir / "run_info.json"
    if run_info.exists():
        with open(run_info) as f:
            data = json.load(f)
            results.append({
                "name": model_dir.name,
                "pnl_pct": data.get("pnl_pct", 0),
                "trades": data.get("trades", 0),
                "cycles": data.get("cycles", 0),
                "final_balance": data.get("final_balance", 5000),
            })

if not results:
    print("No Phase 42 results yet...")
else:
    # Sort by P&L
    results.sort(key=lambda x: x["pnl_pct"], reverse=True)
    
    print("=" * 70)
    print("PHASE 42 EXPERIMENT RESULTS")
    print("=" * 70)
    print(f"{'Experiment':<30} {'P&L%':>10} {'Trades':>8} {'$/Trade':>12}")
    print("-" * 70)
    
    for r in results:
        per_trade = (r["pnl_pct"] * 50) / r["trades"] if r["trades"] > 0 else 0
        status = "✓" if r["pnl_pct"] > 0 else "✗"
        print(f"{r['name']:<30} {r['pnl_pct']:>+9.1f}% {r['trades']:>8} {per_trade:>+11.2f}")
    
    print("-" * 70)
    print(f"Total experiments: {len(results)}")
    profitable = sum(1 for r in results if r["pnl_pct"] > 0)
    print(f"Profitable: {profitable}/{len(results)}")
    
    if results[0]["pnl_pct"] > 0:
        print(f"\nBEST: {results[0]['name']} with {results[0]['pnl_pct']:+.1f}% P&L")
