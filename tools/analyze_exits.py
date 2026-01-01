#!/usr/bin/env python3
"""
Exit Reason Analysis Tool

Analyzes trade exit patterns to understand why trades lose money despite decent win rates.
Helps identify whether the issue is:
- Stop loss hits (SL)
- Take profit hits (TP)
- Time exits / Force close
- Hold time issues

Usage:
    python tools/analyze_exits.py                    # Analyze main paper_trading.db
    python tools/analyze_exits.py --db path/to/db   # Analyze specific database
"""

import sqlite3
import argparse
from pathlib import Path
from collections import defaultdict
import sys

def analyze_database(db_path: str):
    """Analyze trades from a SQLite database."""

    if not Path(db_path).exists():
        print(f"Error: Database not found at {db_path}")
        return

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    print("=" * 70)
    print(f"EXIT REASON ANALYSIS: {db_path}")
    print("=" * 70)

    # 1. Overall statistics
    cursor.execute("""
        SELECT
            COUNT(*) as total_trades,
            SUM(CASE WHEN profit_loss > 0 THEN 1 ELSE 0 END) as wins,
            SUM(CASE WHEN profit_loss <= 0 THEN 1 ELSE 0 END) as losses,
            SUM(profit_loss) as total_pnl,
            AVG(profit_loss) as avg_pnl
        FROM trades
        WHERE exit_timestamp IS NOT NULL
    """)
    row = cursor.fetchone()
    total, wins, losses, total_pnl, avg_pnl = row

    print(f"\n1. OVERALL STATISTICS")
    print("-" * 40)
    print(f"Total Closed Trades: {total}")
    print(f"Wins: {wins} ({100*wins/total:.1f}%)")
    print(f"Losses: {losses} ({100*losses/total:.1f}%)")
    print(f"Total P&L: ${total_pnl:,.2f}")
    print(f"Avg P&L/Trade: ${avg_pnl:.2f}")

    # 2. Status distribution
    print(f"\n2. EXIT STATUS DISTRIBUTION")
    print("-" * 40)
    cursor.execute("""
        SELECT
            status,
            COUNT(*) as count,
            AVG(profit_loss) as avg_pnl,
            SUM(profit_loss) as total_pnl,
            AVG((julianday(exit_timestamp) - julianday(timestamp)) * 24 * 60) as avg_hold_min
        FROM trades
        WHERE exit_timestamp IS NOT NULL
        GROUP BY status
        ORDER BY count DESC
    """)
    for row in cursor.fetchall():
        status, count, avg_pnl, total_pnl, avg_hold = row
        print(f"  {status:15} | {count:5} trades | Avg: ${avg_pnl:+7.2f} | Total: ${total_pnl:+10.2f} | Hold: {avg_hold:.0f}min")

    # 3. P&L Bucket Distribution
    print(f"\n3. P&L DISTRIBUTION (Realized)")
    print("-" * 40)
    cursor.execute("""
        SELECT
            CASE
                WHEN profit_loss >= 15 THEN '+15% and up'
                WHEN profit_loss >= 10 THEN '+10% to +15%'
                WHEN profit_loss >= 5 THEN '+5% to +10%'
                WHEN profit_loss >= 2 THEN '+2% to +5%'
                WHEN profit_loss >= 0 THEN '0% to +2%'
                WHEN profit_loss >= -2 THEN '-2% to 0%'
                WHEN profit_loss >= -5 THEN '-5% to -2%'
                WHEN profit_loss >= -10 THEN '-10% to -5%'
                ELSE '-10% and below'
            END as bucket,
            COUNT(*) as count,
            AVG(profit_loss) as avg_pnl,
            SUM(profit_loss) as total_pnl
        FROM trades
        WHERE exit_timestamp IS NOT NULL
        GROUP BY bucket
        ORDER BY MIN(profit_loss) DESC
    """)

    buckets = list(cursor.fetchall())
    for bucket, count, avg_pnl, total_pnl in buckets:
        bar = "█" * min(int(count / 10), 30)
        print(f"  {bucket:18} | {count:5} | {bar}")

    # 4. Hold Time Analysis
    print(f"\n4. HOLD TIME ANALYSIS")
    print("-" * 40)
    cursor.execute("""
        SELECT
            CASE
                WHEN (julianday(exit_timestamp) - julianday(timestamp)) * 24 * 60 < 15 THEN '< 15 min'
                WHEN (julianday(exit_timestamp) - julianday(timestamp)) * 24 * 60 < 30 THEN '15-30 min'
                WHEN (julianday(exit_timestamp) - julianday(timestamp)) * 24 * 60 < 45 THEN '30-45 min'
                WHEN (julianday(exit_timestamp) - julianday(timestamp)) * 24 * 60 < 60 THEN '45-60 min'
                WHEN (julianday(exit_timestamp) - julianday(timestamp)) * 24 * 60 < 120 THEN '1-2 hours'
                WHEN (julianday(exit_timestamp) - julianday(timestamp)) * 24 * 60 < 240 THEN '2-4 hours'
                ELSE '4+ hours'
            END as hold_bucket,
            COUNT(*) as count,
            AVG(profit_loss) as avg_pnl,
            SUM(CASE WHEN profit_loss > 0 THEN 1 ELSE 0 END) as wins,
            SUM(CASE WHEN profit_loss <= 0 THEN 1 ELSE 0 END) as losses
        FROM trades
        WHERE exit_timestamp IS NOT NULL
        GROUP BY hold_bucket
        ORDER BY MIN((julianday(exit_timestamp) - julianday(timestamp)) * 24 * 60)
    """)

    print(f"  {'Hold Time':12} | {'Trades':6} | {'Win Rate':8} | {'Avg P&L':10}")
    print("  " + "-" * 50)
    for hold, count, avg_pnl, wins, losses in cursor.fetchall():
        win_rate = 100 * wins / count if count > 0 else 0
        print(f"  {hold:12} | {count:6} | {win_rate:6.1f}% | ${avg_pnl:+8.2f}")

    # 5. Win/Loss Size Asymmetry
    print(f"\n5. WIN/LOSS SIZE ASYMMETRY")
    print("-" * 40)
    cursor.execute("""
        SELECT
            AVG(CASE WHEN profit_loss > 0 THEN profit_loss END) as avg_win,
            AVG(CASE WHEN profit_loss <= 0 THEN profit_loss END) as avg_loss,
            MAX(profit_loss) as max_win,
            MIN(profit_loss) as max_loss
        FROM trades
        WHERE exit_timestamp IS NOT NULL
    """)
    avg_win, avg_loss, max_win, max_loss = cursor.fetchone()

    print(f"  Average Win:  ${avg_win:+.2f}")
    print(f"  Average Loss: ${avg_loss:.2f}")
    print(f"  Win/Loss Ratio: {abs(avg_win/avg_loss):.2f}:1")
    print(f"  Max Win:  ${max_win:+.2f}")
    print(f"  Max Loss: ${max_loss:.2f}")

    # Calculate required win rate for break-even
    if avg_win and avg_loss:
        required_wr = abs(avg_loss) / (avg_win + abs(avg_loss)) * 100
        print(f"\n  ⚠️ Required Win Rate for Break-Even: {required_wr:.1f}%")
        print(f"     (Actual Win Rate: {100*wins/total:.1f}%)")

    # 6. Exit Timing Analysis (if exit reaches SL/TP vs Time)
    print(f"\n6. EXIT TIMING INFERENCE")
    print("-" * 40)

    # Try to infer exit reason from P&L relative to SL/TP
    cursor.execute("""
        SELECT
            stop_loss, take_profit, profit_loss, premium_paid, exit_price, status,
            (julianday(exit_timestamp) - julianday(timestamp)) * 24 * 60 as hold_min
        FROM trades
        WHERE exit_timestamp IS NOT NULL
    """)

    exit_reasons = defaultdict(lambda: {'count': 0, 'pnl': 0})

    for sl, tp, pnl, entry, exit, status, hold_min in cursor.fetchall():
        if entry and sl and tp:
            pnl_pct = (exit / entry - 1) * 100 if entry > 0 else 0
            sl_pct = (sl / entry - 1) * 100 if entry > 0 else -100
            tp_pct = (tp / entry - 1) * 100 if entry > 0 else 100

            # Infer exit reason
            if pnl_pct <= sl_pct * 0.9:  # Hit stop loss (within 10%)
                reason = "STOP_LOSS"
            elif pnl_pct >= tp_pct * 0.9:  # Hit take profit (within 10%)
                reason = "TAKE_PROFIT"
            elif hold_min and hold_min > 40:  # Likely time exit
                reason = "TIME_EXIT"
            else:
                reason = "OTHER"
        else:
            reason = "UNKNOWN"

        exit_reasons[reason]['count'] += 1
        exit_reasons[reason]['pnl'] += pnl if pnl else 0

    print("  Inferred Exit Reasons (from SL/TP/hold time):")
    for reason in ['STOP_LOSS', 'TAKE_PROFIT', 'TIME_EXIT', 'OTHER', 'UNKNOWN']:
        if reason in exit_reasons:
            data = exit_reasons[reason]
            avg_pnl = data['pnl'] / data['count'] if data['count'] > 0 else 0
            print(f"    {reason:15} | {data['count']:5} trades | Avg: ${avg_pnl:+.2f}")

    # 7. Recommendations
    print(f"\n7. RECOMMENDATIONS")
    print("-" * 40)

    # Check if losses are held too long
    cursor.execute("""
        SELECT
            AVG(CASE WHEN profit_loss > 0 THEN (julianday(exit_timestamp) - julianday(timestamp)) * 24 * 60 END) as win_hold,
            AVG(CASE WHEN profit_loss <= 0 THEN (julianday(exit_timestamp) - julianday(timestamp)) * 24 * 60 END) as loss_hold
        FROM trades
        WHERE exit_timestamp IS NOT NULL
    """)
    win_hold, loss_hold = cursor.fetchone()

    if loss_hold and win_hold:
        if loss_hold > win_hold * 1.5:
            print(f"  ⚠️ ISSUE: Losses held {loss_hold:.0f}min vs Winners {win_hold:.0f}min")
            print(f"     → Cut losses faster! Reduce max_hold or tighten SL")

    if avg_loss and avg_win:
        if abs(avg_loss) > avg_win:
            print(f"  ⚠️ ISSUE: Avg loss (${abs(avg_loss):.2f}) > Avg win (${avg_win:.2f})")
            print(f"     → Tighten stop loss OR widen take profit")

    time_exits = exit_reasons.get('TIME_EXIT', {'count': 0})['count']
    if time_exits > total * 0.3:
        print(f"  ⚠️ ISSUE: {100*time_exits/total:.0f}% of trades are time exits")
        print(f"     → Reduce TP target to be reachable within hold time")
        print(f"     → Or increase max_hold_minutes")

    conn.close()
    print("\n" + "=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Analyze trade exit patterns")
    parser.add_argument("--db", default="data/paper_trading.db",
                       help="Path to SQLite database")
    args = parser.parse_args()

    # Find database
    base_dir = Path(__file__).parent.parent
    db_path = base_dir / args.db

    if not db_path.exists():
        # Try absolute path
        db_path = Path(args.db)

    analyze_database(str(db_path))


if __name__ == "__main__":
    main()
