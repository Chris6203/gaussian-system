#!/usr/bin/env python3
"""
Post-Experiment Analysis Tool

Generates detailed analysis of an experiment run including:
- Trade pattern analysis (winners vs losers)
- Signal type breakdown
- Confidence distribution
- Exit reason analysis
- Actionable improvement suggestions

Output: ANALYSIS.md in the experiment directory

Usage:
    python tools/post_experiment_analysis.py models/EXP-0169_IDEA-268
"""

import os
import sys
import json
import sqlite3
from datetime import datetime
from pathlib import Path
from collections import defaultdict
import statistics


def load_trades_from_db(run_dir: str) -> list:
    """Load trades from paper_trading.db for this run."""
    db_path = "data/paper_trading.db"
    if not os.path.exists(db_path):
        return []

    # Get run_id from the directory name (e.g., EXP-0169_IDEA-268)
    run_name = os.path.basename(run_dir)

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    # Try to load by run_id first
    cursor.execute("""
        SELECT * FROM trades
        WHERE run_id = ?
        ORDER BY timestamp
    """, (run_name,))

    trades = [dict(row) for row in cursor.fetchall()]

    # If no trades found by run_id, try by time range from run_info.json
    if not trades:
        run_info_path = os.path.join(run_dir, "run_info.json")
        if os.path.exists(run_info_path):
            with open(run_info_path) as f:
                run_info = json.load(f)

            start_time = run_info.get("start_time", "")
            end_time = run_info.get("end_time", datetime.now().isoformat())

            cursor.execute("""
                SELECT * FROM trades
                WHERE timestamp >= ? AND timestamp <= ?
                ORDER BY timestamp
            """, (start_time, end_time))

            trades = [dict(row) for row in cursor.fetchall()]

    conn.close()
    return trades


def load_decision_records(run_dir: str) -> list:
    """Load decision records from JSONL file."""
    records_path = os.path.join(run_dir, "state", "decision_records.jsonl")
    if not os.path.exists(records_path):
        return []

    records = []
    with open(records_path) as f:
        for line in f:
            if line.strip():
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return records


def analyze_trades(trades: list) -> dict:
    """Analyze trade patterns."""
    if not trades:
        return {"error": "No trades found"}

    analysis = {
        "total": len(trades),
        "winners": [],
        "losers": [],
        "by_exit_reason": defaultdict(list),
        "by_option_type": defaultdict(list),
        "by_signal_strategy": defaultdict(list),
        "by_hour": defaultdict(list),
        "confidence_distribution": {"winners": [], "losers": []},
    }

    for trade in trades:
        # Use correct column names from schema
        pnl = trade.get("profit_loss", 0) or 0
        entry_price = trade.get("entry_price", 1) or 1
        exit_price = trade.get("exit_price", entry_price) or entry_price
        pnl_pct = ((exit_price - entry_price) / entry_price * 100) if entry_price > 0 else 0

        # Get confidence from ml_confidence or calibrated_confidence
        confidence = trade.get("calibrated_confidence") or trade.get("ml_confidence", 0) or 0

        trade_summary = {
            "pnl": pnl,
            "pnl_pct": pnl_pct,
            "option_type": trade.get("option_type", "unknown"),
            "exit_reason": trade.get("exit_reason", "unknown"),
            "confidence": confidence,
            "signal_strategy": trade.get("signal_strategy") or "unknown",
            "entry_time": trade.get("timestamp", ""),
            "hold_minutes": trade.get("hold_minutes", 0) or 0,
        }

        # Winners vs losers
        if pnl > 0:
            analysis["winners"].append(trade_summary)
            analysis["confidence_distribution"]["winners"].append(confidence)
        else:
            analysis["losers"].append(trade_summary)
            analysis["confidence_distribution"]["losers"].append(confidence)

        # By exit reason
        exit_reason = trade.get("exit_reason") or "unknown"
        analysis["by_exit_reason"][exit_reason].append(pnl)

        # By option type
        option_type = trade.get("option_type", "unknown")
        analysis["by_option_type"][option_type].append(pnl)

        # By signal strategy
        signal_strategy = trade.get("signal_strategy") or "unknown"
        analysis["by_signal_strategy"][signal_strategy].append(pnl)

        # By hour
        entry_time = trade.get("timestamp", "")
        if entry_time:
            try:
                hour = datetime.fromisoformat(entry_time.replace("Z", "+00:00")).hour
                analysis["by_hour"][hour].append(pnl)
            except:
                pass

    return analysis


def generate_insights(analysis: dict) -> list:
    """Generate actionable insights from analysis."""
    insights = []

    if "error" in analysis:
        return [analysis["error"]]

    # Win rate
    total = analysis["total"]
    winners = len(analysis["winners"])
    win_rate = winners / total * 100 if total > 0 else 0
    insights.append(f"Win Rate: {win_rate:.1f}% ({winners}/{total})")

    # Avg P&L
    all_pnls = [t["pnl"] for t in analysis["winners"] + analysis["losers"]]
    avg_pnl = statistics.mean(all_pnls) if all_pnls else 0
    insights.append(f"Avg P&L per trade: ${avg_pnl:.2f}")

    # Confidence analysis
    if analysis["confidence_distribution"]["winners"] and analysis["confidence_distribution"]["losers"]:
        avg_win_conf = statistics.mean(analysis["confidence_distribution"]["winners"])
        avg_lose_conf = statistics.mean(analysis["confidence_distribution"]["losers"])

        if avg_win_conf > avg_lose_conf:
            insights.append(f"✅ Confidence is predictive: Winners avg {avg_win_conf:.1%} vs Losers avg {avg_lose_conf:.1%}")
        else:
            insights.append(f"⚠️ Confidence is INVERTED: Winners avg {avg_win_conf:.1%} vs Losers avg {avg_lose_conf:.1%}")

    # Exit reason analysis
    if analysis["by_exit_reason"]:
        insights.append("\nExit Reason Performance:")
        for reason, pnls in sorted(analysis["by_exit_reason"].items(), key=lambda x: sum(x[1]), reverse=True):
            total_pnl = sum(pnls)
            count = len(pnls)
            win_count = sum(1 for p in pnls if p > 0)
            insights.append(f"  {reason}: {count} trades, ${total_pnl:.2f} total, {win_count/count*100:.0f}% win rate")

    # Signal strategy analysis
    if analysis["by_signal_strategy"]:
        insights.append("\nSignal Strategy Performance:")
        for strategy, pnls in sorted(analysis["by_signal_strategy"].items(), key=lambda x: sum(x[1]), reverse=True):
            total_pnl = sum(pnls)
            count = len(pnls)
            win_count = sum(1 for p in pnls if p > 0)
            avg = total_pnl / count if count > 0 else 0
            status = "✅" if total_pnl > 0 else "❌"
            insights.append(f"  {status} {strategy}: {count} trades, ${total_pnl:.2f} total, ${avg:.2f}/trade")

    # Hour analysis
    if analysis["by_hour"]:
        best_hour = max(analysis["by_hour"].items(), key=lambda x: sum(x[1]))
        worst_hour = min(analysis["by_hour"].items(), key=lambda x: sum(x[1]))
        insights.append(f"\nBest trading hour: {best_hour[0]}:00 (${sum(best_hour[1]):.2f})")
        insights.append(f"Worst trading hour: {worst_hour[0]}:00 (${sum(worst_hour[1]):.2f})")

    return insights


def generate_recommendations(analysis: dict) -> list:
    """Generate specific recommendations based on analysis."""
    recs = []

    if "error" in analysis:
        return []

    # Check confidence inversion
    if analysis["confidence_distribution"]["winners"] and analysis["confidence_distribution"]["losers"]:
        avg_win_conf = statistics.mean(analysis["confidence_distribution"]["winners"])
        avg_lose_conf = statistics.mean(analysis["confidence_distribution"]["losers"])

        if avg_lose_conf > avg_win_conf:
            recs.append("🔧 CONFIDENCE IS INVERTED: Consider enabling USE_ENTROPY_CONFIDENCE_V2=1")

    # Check losing signal strategies
    losing_strategies = []
    for strategy, pnls in analysis["by_signal_strategy"].items():
        if sum(pnls) < 0 and len(pnls) >= 3:  # At least 3 trades to be significant
            losing_strategies.append(strategy)

    if losing_strategies:
        strategies_str = ",".join(losing_strategies[:3])  # Top 3
        recs.append(f"🔧 BLOCK LOSING STRATEGIES: Set BLOCK_SIGNAL_STRATEGIES={strategies_str}")

    # Check exit reasons
    force_close_pnl = sum(analysis["by_exit_reason"].get("FORCE_CLOSE", []))
    if force_close_pnl < 0 and len(analysis["by_exit_reason"].get("FORCE_CLOSE", [])) > 5:
        recs.append("🔧 FORCE_CLOSE losing money: Consider increasing HARD_TAKE_PROFIT_PCT")

    stop_loss_count = len(analysis["by_exit_reason"].get("STOP_LOSS", []))
    take_profit_count = len(analysis["by_exit_reason"].get("TAKE_PROFIT", []))
    if stop_loss_count > take_profit_count * 2:
        recs.append("🔧 Too many stop-outs: Consider widening HARD_STOP_LOSS_PCT")

    return recs


def write_analysis(run_dir: str, analysis: dict, insights: list, recommendations: list, config: dict):
    """Write ANALYSIS.md to the run directory."""
    output_path = os.path.join(run_dir, "ANALYSIS.md")

    with open(output_path, "w") as f:
        f.write(f"# Experiment Analysis\n\n")
        f.write(f"**Run Directory:** `{run_dir}`\n")
        f.write(f"**Generated:** {datetime.now().isoformat()}\n\n")

        f.write("## Configuration\n\n")
        f.write("```json\n")
        f.write(json.dumps(config, indent=2))
        f.write("\n```\n\n")

        f.write("## Key Insights\n\n")
        for insight in insights:
            f.write(f"{insight}\n")
        f.write("\n")

        f.write("## Recommendations\n\n")
        if recommendations:
            for rec in recommendations:
                f.write(f"- {rec}\n")
        else:
            f.write("No specific recommendations - results look good!\n")
        f.write("\n")

        # Top winners
        f.write("## Top 5 Winners\n\n")
        winners = sorted(analysis.get("winners", []), key=lambda x: x["pnl"], reverse=True)[:5]
        if winners:
            f.write("| P&L | P&L% | Type | Exit | Confidence | Strategy |\n")
            f.write("|-----|------|------|------|------------|----------|\n")
            for w in winners:
                f.write(f"| ${w['pnl']:.2f} | {w['pnl_pct']:.1f}% | {w['option_type']} | {w['exit_reason']} | {w['confidence']:.1%} | {w['signal_strategy'][:20]} |\n")
        else:
            f.write("No winners recorded.\n")
        f.write("\n")

        # Worst losers
        f.write("## Worst 5 Losers\n\n")
        losers = sorted(analysis.get("losers", []), key=lambda x: x["pnl"])[:5]
        if losers:
            f.write("| P&L | P&L% | Type | Exit | Confidence | Strategy |\n")
            f.write("|-----|------|------|------|------------|----------|\n")
            for l in losers:
                f.write(f"| ${l['pnl']:.2f} | {l['pnl_pct']:.1f}% | {l['option_type']} | {l['exit_reason']} | {l['confidence']:.1%} | {l['signal_strategy'][:20]} |\n")
        else:
            f.write("No losers recorded.\n")
        f.write("\n")

        f.write("---\n")
        f.write("*Generated by post_experiment_analysis.py*\n")

    print(f"✅ Analysis written to: {output_path}")
    return output_path


def main():
    if len(sys.argv) < 2:
        print("Usage: python tools/post_experiment_analysis.py <run_directory>")
        sys.exit(1)

    run_dir = sys.argv[1]

    if not os.path.exists(run_dir):
        print(f"Error: Directory not found: {run_dir}")
        sys.exit(1)

    print(f"Analyzing experiment: {run_dir}")

    # Load configuration
    env_vars_path = os.path.join(run_dir, "env_vars.json")
    config = {}
    if os.path.exists(env_vars_path):
        with open(env_vars_path) as f:
            config = json.load(f)

    # Load and analyze trades
    trades = load_trades_from_db(run_dir)
    print(f"Found {len(trades)} trades")

    analysis = analyze_trades(trades)
    insights = generate_insights(analysis)
    recommendations = generate_recommendations(analysis)

    # Write analysis
    write_analysis(run_dir, analysis, insights, recommendations, config)

    # Print summary
    print("\n" + "="*60)
    print("ANALYSIS SUMMARY")
    print("="*60)
    for insight in insights[:10]:  # First 10 insights
        print(insight)

    if recommendations:
        print("\nRECOMMENDATIONS:")
        for rec in recommendations:
            print(f"  {rec}")


if __name__ == "__main__":
    main()
