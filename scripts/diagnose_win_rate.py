#!/usr/bin/env python3
"""
Win Rate Diagnostic Script
Checks for common issues that cause low win rates.
"""

import sys
import os

# Fix Windows encoding issues
try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
except:
    pass  # Older Python versions

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import sqlite3
import pandas as pd
from datetime import datetime
from pathlib import Path

# Import centralized P&L calculation (prevents 100x multiplier bug)
try:
    from utils.pnl import compute_pnl_pct
    HAS_PNL_HELPER = True
except ImportError:
    HAS_PNL_HELPER = False

print("="*70)
print("WIN RATE DIAGNOSTIC TOOL")
print("="*70)

# 1. Load recent trades from database
# Try multiple possible paths
possible_paths = [
    Path("data/paper_trading.db"),
    Path("data/db/paper_trading.db"),
    Path("data/trading.db"),
    Path("data/db/trading.db"),
]

db_path = None
for p in possible_paths:
    if p.exists():
        db_path = p
        break

if db_path is None:
    print(f"[ERROR] No database found. Tried: {[str(p) for p in possible_paths]}")
    sys.exit(1)

print(f"[INFO] Using database: {db_path}")

conn = sqlite3.connect(str(db_path))

# Get all closed trades
trades_df = pd.read_sql_query("""
    SELECT id, timestamp, symbol, option_type, strike_price, 
           premium_paid, quantity, entry_price, exit_price, 
           exit_timestamp, profit_loss, status
    FROM trades 
    WHERE status IN ('PROFIT_TAKEN', 'STOPPED_OUT', 'CLOSED')
    ORDER BY timestamp DESC
    LIMIT 500
""", conn)

if trades_df.empty:
    print("[ERROR] No closed trades found in database")
    sys.exit(1)

print(f"\n[INFO] Analyzing {len(trades_df)} closed trades...")

# ============================================================================
# CHECK 1: Naive win rate (raw P&L > 0)
# ============================================================================
print("\n" + "="*70)
print("CHECK 1: WIN RATE CALCULATION")
print("="*70)

winners = (trades_df['profit_loss'] > 0).sum()
losers = (trades_df['profit_loss'] <= 0).sum()
total = len(trades_df)

print(f"  Winners: {winners} ({winners/total*100:.1f}%)")
print(f"  Losers:  {losers} ({losers/total*100:.1f}%)")
print(f"  Win Rate (P&L > 0): {winners/total*100:.1f}%")

# ============================================================================
# CHECK 2: P&L PERCENTAGES - The bug we found
# ============================================================================
print("\n" + "="*70)
print("CHECK 2: P&L PERCENTAGE CALCULATION")
print("="*70)

# WRONG way (the bug)
trades_df['pnl_pct_WRONG'] = trades_df['profit_loss'] / trades_df['premium_paid'].clip(lower=1.0)

# CORRECT way 
trades_df['cost_basis'] = trades_df['premium_paid'] * trades_df['quantity'] * 100
trades_df['pnl_pct_CORRECT'] = trades_df['profit_loss'] / trades_df['cost_basis'].clip(lower=1.0)

print("\n  WRONG calculation (bug):")
print(f"    Mean P&L%: {trades_df['pnl_pct_WRONG'].mean()*100:.1f}%")
print(f"    Min P&L%:  {trades_df['pnl_pct_WRONG'].min()*100:.1f}%")
print(f"    Max P&L%:  {trades_df['pnl_pct_WRONG'].max()*100:.1f}%")

print("\n  CORRECT calculation:")
print(f"    Mean P&L%: {trades_df['pnl_pct_CORRECT'].mean()*100:.2f}%")
print(f"    Min P&L%:  {trades_df['pnl_pct_CORRECT'].min()*100:.2f}%")
print(f"    Max P&L%:  {trades_df['pnl_pct_CORRECT'].max()*100:.2f}%")

# Check if bug exists
if trades_df['pnl_pct_WRONG'].abs().max() > 1.0:
    print("\n  [WARNING] BUG DETECTED: P&L percentages exceed 100%!")
    print("     This indicates the old buggy calculation was used.")
    print("     FIXES HAVE BEEN APPLIED - run training again to see improvement.")
else:
    print("\n  [OK] P&L percentages look reasonable")

# ============================================================================
# CHECK 3: CALL vs PUT performance
# ============================================================================
print("\n" + "="*70)
print("CHECK 3: CALL vs PUT PERFORMANCE")
print("="*70)

for opt_type in ['OrderType.CALL', 'CALL', 'OrderType.PUT', 'PUT']:
    mask = trades_df['option_type'].astype(str).str.contains(opt_type.replace('OrderType.', ''), case=False)
    if mask.any():
        subset = trades_df[mask]
        if len(subset) > 0:
            win_rate = (subset['profit_loss'] > 0).mean() * 100
            avg_pnl = subset['profit_loss'].mean()
            print(f"\n  {opt_type.replace('OrderType.', '')}:")
            print(f"    Count: {len(subset)}")
            print(f"    Win rate: {win_rate:.1f}%")
            print(f"    Avg P&L: ${avg_pnl:.2f}")
            
            # Check for sign issues
            if win_rate < 30:
                print(f"    [WARNING] Low win rate might indicate P&L sign issue!")

# ============================================================================
# CHECK 4: Exit reason distribution
# ============================================================================
print("\n" + "="*70)
print("CHECK 4: EXIT REASONS")
print("="*70)

print("\n  Status distribution:")
print(trades_df['status'].value_counts().to_string())

# ============================================================================
# CHECK 5: Sample trades for manual verification
# ============================================================================
print("\n" + "="*70)
print("CHECK 5: SAMPLE TRADES FOR MANUAL VERIFICATION")
print("="*70)

print("\nRecent trades (newest first):")
for _, trade in trades_df.head(10).iterrows():
    entry_premium = trade['premium_paid']
    exit_price = trade['exit_price'] if pd.notna(trade['exit_price']) else 0
    pnl = trade['profit_loss']
    qty = trade['quantity']
    opt_type = str(trade['option_type']).replace('OrderType.', '')
    
    # Expected P&L calculation
    expected_pnl = (exit_price - entry_premium) * qty * 100
    pnl_match = abs(pnl - expected_pnl) < 5  # Allow $5 for fees
    
    correct_pnl_pct = trade['pnl_pct_CORRECT'] * 100
    
    print(f"\n  Trade {trade['id'][:8]}...")
    print(f"    Type: {opt_type} | Qty: {qty}")
    print(f"    Entry: ${entry_premium:.2f} | Exit: ${exit_price:.2f}")
    print(f"    P&L: ${pnl:.2f} (correct %: {correct_pnl_pct:+.1f}%)")
    print(f"    Expected P&L: ${expected_pnl:.2f} | Match: {'[OK]' if pnl_match else '[MISMATCH]'}")

# ============================================================================
# CHECK 6: Entry signal quality
# ============================================================================
print("\n" + "="*70)
print("CHECK 6: SUMMARY & RECOMMENDATIONS")
print("="*70)

actual_win_rate = winners/total*100

print(f"\n  Current Win Rate: {actual_win_rate:.1f}%")

if actual_win_rate < 35:
    print("\n  [WARNING] WIN RATE IS LOW. Likely causes:")
    print("     1. [FIXED] P&L% calculation bug in experience_replay.py")
    print("     2. [FIXED] Status not updated after fee/friction calculation")
    print("     3. Check if HMM alignment is blocking good trades")
    print("     4. Check if entry confidence is too high/low")
    print("     5. Run this script AFTER a fresh training run")
elif actual_win_rate >= 35 and actual_win_rate < 45:
    print("\n  [INFO] Win rate is mediocre - consider:")
    print("     - Tightening confidence thresholds")
    print("     - Improving HMM regime detection")
    print("     - Better exit timing")
else:
    print("\n  [OK] Win rate looks acceptable!")

conn.close()
print("\n" + "="*70)
print("DIAGNOSTIC COMPLETE")
print("="*70)
