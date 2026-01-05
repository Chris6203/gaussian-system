#!/usr/bin/env python3
"""
Update Results and Push to GitHub

Queries the experiments database for the latest best results,
updates BEST_CONFIGS.md, and commits/pushes to GitHub.

Usage:
    python scripts/update_results_github.py           # Update and push
    python scripts/update_results_github.py --dry-run # Show changes without committing
"""

import os
import sys
import sqlite3
import subprocess
from datetime import datetime
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def get_top_results(db_path: str, limit: int = 15) -> tuple:
    """Get top experiment results from database (by win rate and P&L)."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Get best by win rate (min 20 trades, positive P&L)
    cursor.execute('''
        SELECT run_name, pnl_pct, win_rate, trades, per_trade_pnl
        FROM experiments
        WHERE trades >= 20 AND pnl_pct > 0
        ORDER BY win_rate DESC
        LIMIT ?
    ''', (limit,))

    by_winrate = []
    for row in cursor.fetchall():
        by_winrate.append({
            'name': row[0],
            'pnl_pct': row[1],
            'win_rate': row[2] * 100 if row[2] else 0,
            'trades': row[3],
            'per_trade': row[4]
        })

    # Get best by P&L (filter out pre-bugfix phantom gains)
    cursor.execute('''
        SELECT run_name, pnl_pct, win_rate, trades, per_trade_pnl
        FROM experiments
        WHERE trades > 10 AND pnl_pct < 600 AND pnl_pct > 0
        ORDER BY pnl_pct DESC
        LIMIT ?
    ''', (limit,))

    by_pnl = []
    for row in cursor.fetchall():
        by_pnl.append({
            'name': row[0],
            'pnl_pct': row[1],
            'win_rate': row[2] * 100 if row[2] else 0,
            'trades': row[3],
            'per_trade': row[4]
        })

    conn.close()
    return by_winrate, by_pnl


def get_experiment_count(db_path: str) -> int:
    """Get total experiment count."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute('SELECT COUNT(*) FROM experiments')
    count = cursor.fetchone()[0]
    conn.close()
    return count


def update_best_configs(by_winrate: list, by_pnl: list, total_experiments: int):
    """Update BEST_CONFIGS.md with latest results."""
    best_configs_path = PROJECT_ROOT / 'docs' / 'BEST_CONFIGS.md'

    if not best_configs_path.exists():
        print(f"Warning: {best_configs_path} not found")
        return False

    content = best_configs_path.read_text()

    # Build the new results tables
    today = datetime.now().strftime('%Y-%m-%d')

    # Best result (highest win rate with good P&L)
    best = by_winrate[0] if by_winrate else None

    table_lines = [
        f"**Last Updated**: {today}",
        "",
        "This document tracks the best validated configurations for the Gaussian Options Trading Bot.",
        "",
        f"**Total Experiments**: {total_experiments}",
        "",
        "---",
        "",
        "## 🏆 Current Best Result",
        "",
        "| Run | P&L% | Win Rate | $/Trade | Trades | Description |",
        "|-----|------|----------|---------|--------|-------------|",
    ]

    if best:
        table_lines.append(
            f"| **{best['name']}** | **+{best['pnl_pct']:.2f}%** | **{best['win_rate']:.1f}%** | **${best['per_trade']:.2f}** | {best['trades']} | Best Win Rate |"
        )

    table_lines.extend([
        "",
        f"This is the highest win rate achieved with significant P&L.",
        "",
        "---",
        "",
        "## Top Results by Win Rate (min 20 trades)",
        "",
        "| Rank | Run Name | P&L% | WR% | $/Trade | Trades |",
        "|------|----------|------|-----|---------|--------|",
    ])

    medals = ['🥇', '🥈', '🥉']
    for i, r in enumerate(by_winrate[:6]):
        rank = medals[i] if i < 3 else str(i + 1)
        bold = "**" if i == 0 else ""
        table_lines.append(
            f"| {rank} | {bold}{r['name']}{bold} | +{r['pnl_pct']:.2f}% | {bold}{r['win_rate']:.1f}%{bold} | ${r['per_trade']:.2f} | {r['trades']} |"
        )

    table_lines.extend([
        "",
        "## Top Results by P&L% (filtered)",
        "",
        "| Rank | Run Name | P&L% | WR% | $/Trade | Trades |",
        "|------|----------|------|-----|---------|--------|",
    ])

    for i, r in enumerate(by_pnl[:5], 1):
        table_lines.append(
            f"| {i} | {r['name']} | +{r['pnl_pct']:.2f}% | {r['win_rate']:.1f}% | ${r['per_trade']:.2f} | {r['trades']} |"
        )

    table_lines.extend([
        "",
        "**Note:** Results filtered to P&L < 600% to exclude pre-bugfix phantom gains.",
        "",
        "---",
    ])

    # Find where to insert (after the title)
    lines = content.split('\n')
    new_lines = ['# Best Trading Configurations', '']
    new_lines.extend(table_lines)

    # Find the Bug Fix Notice section and keep everything after it
    found_bug_notice = False
    for i, line in enumerate(lines):
        if '## IMPORTANT: Bug Fix Notice' in line:
            found_bug_notice = True
            new_lines.extend(lines[i:])
            break

    if not found_bug_notice:
        # Fallback: keep original content after Validated Best Configuration
        for i, line in enumerate(lines):
            if '## Validated Best Configuration' in line:
                new_lines.extend(lines[i:])
                break

    best_configs_path.write_text('\n'.join(new_lines))
    return True


def git_commit_and_push(dry_run: bool = False):
    """Commit and push changes to GitHub."""
    os.chdir(PROJECT_ROOT)

    # Check for changes
    result = subprocess.run(['git', 'status', '--porcelain', 'docs/BEST_CONFIGS.md'],
                          capture_output=True, text=True)

    if not result.stdout.strip():
        print("No changes to commit")
        return False

    if dry_run:
        print("DRY RUN - Would commit these changes:")
        print(result.stdout)
        return True

    # Add and commit
    subprocess.run(['git', 'add', 'docs/BEST_CONFIGS.md'], check=True)

    commit_msg = f"""Update best results scoreboard ({datetime.now().strftime('%Y-%m-%d %H:%M')})

Automated update from experiments database.

🤖 Generated with [Claude Code](https://claude.com/claude-code)
"""

    subprocess.run(['git', 'commit', '-m', commit_msg], check=True)

    # Push
    result = subprocess.run(['git', 'push'], capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Push failed: {result.stderr}")
        return False

    print("Successfully pushed to GitHub")
    return True


def main():
    dry_run = '--dry-run' in sys.argv

    db_path = PROJECT_ROOT / 'data' / 'experiments.db'

    if not db_path.exists():
        print(f"Error: Database not found at {db_path}")
        return 1

    print("Fetching top experiment results...")
    by_winrate, by_pnl = get_top_results(str(db_path))
    total = get_experiment_count(str(db_path))

    if not by_winrate and not by_pnl:
        print("No results found in database")
        return 1

    print(f"Found results from {total} total experiments")

    if by_winrate:
        print("\n🏆 Best by Win Rate:")
        best = by_winrate[0]
        print(f"  {best['name']}: {best['win_rate']:.1f}% WR, +{best['pnl_pct']:.2f}% P&L, ${best['per_trade']:.2f}/trade")

    print("\nTop 5 by P&L:")
    for i, r in enumerate(by_pnl[:5], 1):
        print(f"  {i}. {r['name']}: +{r['pnl_pct']:.2f}% P&L, {r['win_rate']:.1f}% WR")

    print("\nUpdating BEST_CONFIGS.md...")
    if update_best_configs(by_winrate, by_pnl, total):
        print("Updated successfully")

    print("\nCommitting and pushing to GitHub...")
    if git_commit_and_push(dry_run):
        if dry_run:
            print("DRY RUN complete - no changes made")
        else:
            print("GitHub updated successfully")

    return 0


if __name__ == '__main__':
    sys.exit(main())
