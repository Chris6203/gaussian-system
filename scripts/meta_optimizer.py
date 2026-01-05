#!/usr/bin/env python3
"""
Meta Optimizer (Layer 2)

Reads all experiment analyses, identifies patterns, and uses AI to suggest
new configurations to test. Feeds suggestions to the Layer 1 optimizer queue.

Architecture:
    Layer 1 (continuous_optimizer.py): Tests specific configurations
    Layer 2 (meta_optimizer.py): Analyzes results, suggests new configs

Usage:
    python scripts/meta_optimizer.py              # Run once
    python scripts/meta_optimizer.py --loop       # Run continuously
    python scripts/meta_optimizer.py --dry-run    # Show suggestions without adding to queue
"""

import os
import sys
import json
import re
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
import logging
import argparse
import subprocess

# Add project root to path
BASE_DIR = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(BASE_DIR))

# Configuration
COLLAB_DIR = BASE_DIR / ".claude" / "collab"
MODELS_DIR = BASE_DIR / "models"
MIN_EXPERIMENTS_FOR_ANALYSIS = 10
MAX_IDEAS_PER_RUN = 5

logging.basicConfig(level=logging.INFO, format="%(asctime)s [META] %(message)s")
logger = logging.getLogger(__name__)


def load_all_analyses() -> List[Dict]:
    """Load all ANALYSIS.md files and parse key insights."""
    analyses = []

    for model_dir in MODELS_DIR.iterdir():
        if not model_dir.is_dir():
            continue

        analysis_file = model_dir / "ANALYSIS.md"
        summary_file = model_dir / "SUMMARY.txt"
        env_file = model_dir / "env_vars.json"

        if not analysis_file.exists():
            continue

        try:
            analysis_text = analysis_file.read_text(encoding='utf-8')
            summary_text = summary_file.read_text(encoding='utf-8') if summary_file.exists() else ""
            env_vars = json.loads(env_file.read_text()) if env_file.exists() else {}

            # Extract key metrics
            analysis = {
                "run_name": model_dir.name,
                "analysis_text": analysis_text,
                "env_vars": env_vars,
                "metrics": parse_metrics(summary_text),
                "recommendations": extract_recommendations(analysis_text),
                "confidence_inverted": "INVERTED" in analysis_text.upper(),
                "losing_strategies": extract_losing_strategies(analysis_text),
                "best_exit_reasons": extract_best_exit_reasons(analysis_text),
            }

            analyses.append(analysis)

        except Exception as e:
            logger.debug(f"Error parsing {model_dir.name}: {e}")

    return analyses


def parse_metrics(summary_text: str) -> Dict:
    """Parse metrics from SUMMARY.txt."""
    metrics = {}

    patterns = {
        "pnl_pct": r"P&L.*\(([-\d.]+)%\)",
        "win_rate": r"Win Rate[:\s]+(\d+\.?\d*)%?",
        "trades": r"(?:Total )?Trades[:\s]+(\d+)",
        "per_trade_pnl": r"per.?trade.*\$([-\d.]+)",
    }

    for key, pattern in patterns.items():
        match = re.search(pattern, summary_text, re.IGNORECASE)
        if match:
            try:
                metrics[key] = float(match.group(1))
            except:
                pass

    return metrics


def extract_recommendations(analysis_text: str) -> List[str]:
    """Extract recommendations from analysis."""
    recs = []
    lines = analysis_text.split("\n")

    for line in lines:
        if "🔧" in line or "BLOCK_SIGNAL_STRATEGIES" in line:
            recs.append(line.strip())

    return recs


def extract_losing_strategies(analysis_text: str) -> List[str]:
    """Extract signal strategies that consistently lose money."""
    losing = []

    # Look for patterns like "❌ STRATEGY_NAME: X trades, $-Y total"
    pattern = r"❌\s*([A-Z_]+):\s*\d+\s*trades,\s*\$?([-\d.]+)"
    matches = re.findall(pattern, analysis_text)

    for strategy, pnl in matches:
        if float(pnl) < 0:
            losing.append(strategy)

    return losing


def extract_best_exit_reasons(analysis_text: str) -> List[str]:
    """Extract exit reasons that consistently win."""
    best = []

    # Look for patterns with positive P&L and high win rate
    lines = analysis_text.split("\n")
    for line in lines:
        if "100% win rate" in line.lower() or "💰" in line:
            best.append(line.strip()[:100])

    return best[:5]


def aggregate_patterns(analyses: List[Dict]) -> Dict:
    """Aggregate patterns across all experiments."""
    patterns = {
        "total_experiments": len(analyses),
        "confidence_inversions": 0,
        "losing_strategies": {},
        "winning_configs": [],
        "losing_configs": [],
        "env_var_impacts": {},
    }

    for a in analyses:
        # Count confidence inversions
        if a.get("confidence_inverted"):
            patterns["confidence_inversions"] += 1

        # Aggregate losing strategies
        for strategy in a.get("losing_strategies", []):
            patterns["losing_strategies"][strategy] = patterns["losing_strategies"].get(strategy, 0) + 1

        # Track winning/losing configs
        metrics = a.get("metrics", {})
        pnl = metrics.get("pnl_pct", 0)
        per_trade = metrics.get("per_trade_pnl", 0)

        if pnl > 50 or per_trade > 5:
            patterns["winning_configs"].append({
                "run_name": a["run_name"],
                "pnl_pct": pnl,
                "per_trade_pnl": per_trade,
                "env_vars": a.get("env_vars", {}),
            })
        elif pnl < -50 or per_trade < -5:
            patterns["losing_configs"].append({
                "run_name": a["run_name"],
                "pnl_pct": pnl,
                "per_trade_pnl": per_trade,
                "env_vars": a.get("env_vars", {}),
            })

        # Track env var impacts
        for key, value in a.get("env_vars", {}).items():
            if key not in patterns["env_var_impacts"]:
                patterns["env_var_impacts"][key] = {"values": {}, "pnls": []}

            value_str = str(value)
            if value_str not in patterns["env_var_impacts"][key]["values"]:
                patterns["env_var_impacts"][key]["values"][value_str] = []

            patterns["env_var_impacts"][key]["values"][value_str].append(pnl)
            patterns["env_var_impacts"][key]["pnls"].append(pnl)

    # Sort winning configs by P&L
    patterns["winning_configs"].sort(key=lambda x: x.get("pnl_pct", 0), reverse=True)
    patterns["losing_configs"].sort(key=lambda x: x.get("pnl_pct", 0))

    return patterns


def generate_ai_suggestions(patterns: Dict) -> List[Dict]:
    """Generate new experiment suggestions based on patterns."""
    suggestions = []

    # 1. If confidence is often inverted, suggest entropy-based confidence
    if patterns["confidence_inversions"] > patterns["total_experiments"] * 0.3:
        suggestions.append({
            "hypothesis": "Confidence is frequently inverted - try entropy-based confidence",
            "env_vars": {"USE_ENTROPY_CONFIDENCE_V2": "1", "DECOUPLE_UNCERTAINTY": "1"},
            "priority": 1,
            "category": "confidence_fix",
        })

    # 2. Block commonly losing strategies
    losing_strategies = [s for s, count in patterns["losing_strategies"].items() if count >= 3]
    if losing_strategies:
        strategies_str = ",".join(losing_strategies[:3])
        suggestions.append({
            "hypothesis": f"Block consistently losing strategies: {strategies_str}",
            "env_vars": {"BLOCK_SIGNAL_STRATEGIES": strategies_str},
            "priority": 1,
            "category": "signal_filter",
        })

    # 3. Combine best config elements
    winning = patterns.get("winning_configs", [])[:3]
    if len(winning) >= 2:
        # Find common env vars among winners
        common_vars = {}
        for config in winning:
            for key, value in config.get("env_vars", {}).items():
                if key not in common_vars:
                    common_vars[key] = {}
                value_str = str(value)
                common_vars[key][value_str] = common_vars[key].get(value_str, 0) + 1

        # Pick most common values
        combined_env = {}
        for key, values in common_vars.items():
            if values:
                best_value = max(values.items(), key=lambda x: x[1])[0]
                if values[best_value] >= 2:  # At least 2 winners use this
                    combined_env[key] = best_value

        if combined_env:
            suggestions.append({
                "hypothesis": "Combine common elements from top performing configs",
                "env_vars": combined_env,
                "priority": 2,
                "category": "best_combination",
            })

    # 4. Analyze env var impacts and suggest optimal values
    for var_name, data in patterns.get("env_var_impacts", {}).items():
        if len(data["values"]) < 2:
            continue

        # Find best performing value
        best_value = None
        best_avg_pnl = float("-inf")

        for value, pnls in data["values"].items():
            if len(pnls) >= 3:  # Need at least 3 data points
                avg_pnl = sum(pnls) / len(pnls)
                if avg_pnl > best_avg_pnl:
                    best_avg_pnl = avg_pnl
                    best_value = value

        if best_value and best_avg_pnl > 10:  # Only if significantly positive
            suggestions.append({
                "hypothesis": f"Optimal {var_name}={best_value} shows avg +{best_avg_pnl:.1f}% P&L",
                "env_vars": {var_name: best_value},
                "priority": 3,
                "category": "parameter_optimization",
            })

    return suggestions[:MAX_IDEAS_PER_RUN]


def add_to_optimizer_queue(suggestions: List[Dict]) -> int:
    """Add suggestions to the Layer 1 optimizer queue."""
    queue_file = COLLAB_DIR / "idea_queue.json"

    if not queue_file.exists():
        queue = {"ideas": [], "last_updated": datetime.now().isoformat()}
    else:
        with open(queue_file) as f:
            queue = json.load(f)

    # Find max idea ID
    max_id = 0
    for idea in queue.get("ideas", []):
        if "IDEA-" in idea.get("id", ""):
            try:
                num = int(idea["id"].split("-")[1])
                max_id = max(max_id, num)
            except:
                pass

    # Add new ideas
    added = 0
    for i, suggestion in enumerate(suggestions):
        idea_id = f"IDEA-{max_id + i + 1}"

        # Check for duplicates (same env_vars)
        env_hash = json.dumps(suggestion["env_vars"], sort_keys=True)
        duplicate = False
        for existing in queue.get("ideas", []):
            existing_hash = json.dumps(existing.get("env_vars", {}), sort_keys=True)
            if existing_hash == env_hash:
                duplicate = True
                break

        if duplicate:
            logger.info(f"Skipping duplicate: {suggestion['hypothesis'][:50]}")
            continue

        idea = {
            "id": idea_id,
            "source": "meta_optimizer",
            "timestamp": datetime.now().isoformat(),
            "hypothesis": suggestion["hypothesis"],
            "description": f"Generated by Layer 2 Meta Optimizer based on analysis of {suggestion.get('category', 'patterns')}",
            "env_vars": suggestion["env_vars"],
            "category": suggestion.get("category", "meta"),
            "status": "pending",
            "priority": suggestion.get("priority", 5),
        }

        queue["ideas"].insert(0, idea)  # Add at front (high priority)
        added += 1
        logger.info(f"Added {idea_id}: {suggestion['hypothesis'][:60]}")

    queue["last_updated"] = datetime.now().isoformat()

    with open(queue_file, "w") as f:
        json.dump(queue, f, indent=2)

    return added


def generate_summary_report(analyses: List[Dict], patterns: Dict) -> str:
    """Generate a summary report of findings."""
    report = []
    report.append("=" * 60)
    report.append("META OPTIMIZER SUMMARY REPORT")
    report.append(f"Generated: {datetime.now().isoformat()}")
    report.append("=" * 60)

    report.append(f"\nAnalyzed {patterns['total_experiments']} experiments with ANALYSIS.md")

    # Confidence issues
    if patterns["confidence_inversions"] > 0:
        pct = patterns["confidence_inversions"] / patterns["total_experiments"] * 100
        report.append(f"\n⚠️  Confidence inverted in {patterns['confidence_inversions']} experiments ({pct:.0f}%)")

    # Losing strategies
    if patterns["losing_strategies"]:
        report.append("\n❌ Frequently losing strategies:")
        for strategy, count in sorted(patterns["losing_strategies"].items(), key=lambda x: -x[1])[:5]:
            report.append(f"   {strategy}: loses in {count} experiments")

    # Best configs
    if patterns["winning_configs"]:
        report.append("\n✅ Top performing configurations:")
        for config in patterns["winning_configs"][:5]:
            report.append(f"   {config['run_name']}: {config['pnl_pct']:.1f}% P&L, ${config.get('per_trade_pnl', 0):.2f}/trade")

    # Worst configs
    if patterns["losing_configs"]:
        report.append("\n❌ Worst performing configurations:")
        for config in patterns["losing_configs"][:3]:
            report.append(f"   {config['run_name']}: {config['pnl_pct']:.1f}% P&L")

    return "\n".join(report)


def main():
    parser = argparse.ArgumentParser(description="Meta Optimizer - Layer 2 experiment analysis")
    parser.add_argument("--loop", action="store_true", help="Run continuously")
    parser.add_argument("--dry-run", action="store_true", help="Show suggestions without adding to queue")
    parser.add_argument("--interval", type=int, default=3600, help="Loop interval in seconds")
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("Meta Optimizer (Layer 2) Starting")
    logger.info("=" * 60)

    while True:
        # Load all analyses
        logger.info("Loading experiment analyses...")
        analyses = load_all_analyses()
        logger.info(f"Found {len(analyses)} experiments with analysis")

        if len(analyses) < MIN_EXPERIMENTS_FOR_ANALYSIS:
            logger.warning(f"Need at least {MIN_EXPERIMENTS_FOR_ANALYSIS} experiments for meta-analysis")
            if not args.loop:
                return
            import time
            time.sleep(args.interval)
            continue

        # Aggregate patterns
        logger.info("Aggregating patterns...")
        patterns = aggregate_patterns(analyses)

        # Generate summary report
        report = generate_summary_report(analyses, patterns)
        print(report)

        # Generate AI suggestions
        logger.info("Generating suggestions...")
        suggestions = generate_ai_suggestions(patterns)

        if suggestions:
            print(f"\n📋 Generated {len(suggestions)} new experiment suggestions:")
            for s in suggestions:
                print(f"   [{s.get('priority', 5)}] {s['hypothesis'][:70]}")
                print(f"       env_vars: {s['env_vars']}")

            if not args.dry_run:
                added = add_to_optimizer_queue(suggestions)
                logger.info(f"Added {added} new ideas to optimizer queue")
            else:
                logger.info("Dry run - not adding to queue")
        else:
            logger.info("No new suggestions generated")

        if not args.loop:
            break

        logger.info(f"Sleeping for {args.interval} seconds...")
        import time
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
