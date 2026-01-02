#!/usr/bin/env python3
"""
Tuning Data Analyzer - Overnight Improvement Cycle

Monitors training progress, analyzes trades with tuning data,
and submits improvement ideas to the idea queue for continuous_optimizer to test.

Run: python scripts/tuning_analyzer.py
"""

import sqlite3
import json
import time
import subprocess
from datetime import datetime, timedelta
from pathlib import Path
import logging

BASE_DIR = Path(__file__).parent.parent.resolve()
IDEA_QUEUE = BASE_DIR / ".claude" / "collab" / "idea_queue.json"
PAPER_DB = BASE_DIR / "data" / "paper_trading.db"
RUNNING_EXP = BASE_DIR / "data" / "running_experiment.json"

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def get_running_experiment():
    """Get current running experiment status."""
    try:
        if RUNNING_EXP.exists():
            with open(RUNNING_EXP) as f:
                return json.load(f)
    except:
        pass
    return None


def sync_experiments_db():
    """Sync experiments from models/ to experiments.db for dashboard visibility."""
    try:
        result = subprocess.run(
            ["python", str(BASE_DIR / "tools" / "experiments_db.py"), "scan"],
            cwd=str(BASE_DIR),
            capture_output=True,
            text=True,
            timeout=60
        )
        # Parse output for import count
        for line in result.stdout.split('\n'):
            if 'Imported' in line:
                logger.info(f"[DB SYNC] {line.strip()}")
                return
        logger.info("[DB SYNC] Experiments database synced")
    except Exception as e:
        logger.warning(f"[DB SYNC] Failed to sync: {e}")


def git_commit_and_push():
    """Commit and push results to GitHub."""
    try:
        # Check for changes in key files
        result = subprocess.run(
            ["git", "status", "--porcelain",
             "RESULTS_TRACKER.md",
             ".claude/collab/idea_queue.json",
             "data/experiments.db"],
            cwd=str(BASE_DIR),
            capture_output=True,
            text=True,
            timeout=30
        )

        if not result.stdout.strip():
            logger.debug("[GIT] No changes to commit")
            return False

        # Add the files
        subprocess.run(
            ["git", "add",
             "RESULTS_TRACKER.md",
             "data/experiments.db"],
            cwd=str(BASE_DIR),
            capture_output=True,
            timeout=30
        )

        # Create commit with timestamp
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
        commit_msg = f"[auto] Update experiment results ({timestamp})"

        result = subprocess.run(
            ["git", "commit", "-m", commit_msg],
            cwd=str(BASE_DIR),
            capture_output=True,
            text=True,
            timeout=30
        )

        if result.returncode != 0:
            if "nothing to commit" in result.stdout or "nothing to commit" in result.stderr:
                return False
            logger.warning(f"[GIT] Commit failed: {result.stderr}")
            return False

        logger.info(f"[GIT] Committed: {commit_msg}")

        # Push to remote
        result = subprocess.run(
            ["git", "push"],
            cwd=str(BASE_DIR),
            capture_output=True,
            text=True,
            timeout=60
        )

        if result.returncode == 0:
            logger.info("[GIT] Pushed to GitHub successfully")
            return True
        else:
            logger.warning(f"[GIT] Push failed: {result.stderr}")
            return False

    except subprocess.TimeoutExpired:
        logger.warning("[GIT] Git operation timed out")
        return False
    except Exception as e:
        logger.warning(f"[GIT] Error: {e}")
        return False


def analyze_tuning_data():
    """Analyze trades with tuning data to find improvement opportunities."""
    insights = []

    if not PAPER_DB.exists():
        return insights

    conn = sqlite3.connect(PAPER_DB)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    # Check if tuning columns exist
    cursor.execute("PRAGMA table_info(trades)")
    columns = {row[1] for row in cursor.fetchall()}
    has_tuning = 'hmm_trend' in columns and 'entry_controller' in columns

    if not has_tuning:
        logger.info("Tuning columns not yet populated - using basic analysis")
        return insights

    # 1. Analyze win rate by confidence bucket
    cursor.execute("""
        SELECT
            CASE
                WHEN ml_confidence < 0.6 THEN 'low'
                WHEN ml_confidence < 0.75 THEN 'medium'
                ELSE 'high'
            END as conf_bucket,
            COUNT(*) as trades,
            AVG(profit_loss) as avg_pnl,
            SUM(CASE WHEN profit_loss > 0 THEN 1.0 ELSE 0.0 END) / COUNT(*) as win_rate
        FROM trades
        WHERE ml_confidence IS NOT NULL AND timestamp > datetime('now', '-7 days')
        GROUP BY conf_bucket
        HAVING trades > 10
    """)
    conf_results = cursor.fetchall()
    for row in conf_results:
        if row['conf_bucket'] == 'high' and row['win_rate'] > 0.45:
            insights.append({
                'type': 'confidence_threshold',
                'finding': f"High confidence (>75%) trades have {row['win_rate']*100:.1f}% win rate",
                'suggestion': 'Increase min_confidence_to_trade to 0.75'
            })
        elif row['conf_bucket'] == 'low' and row['win_rate'] < 0.35:
            insights.append({
                'type': 'confidence_threshold',
                'finding': f"Low confidence (<60%) trades have {row['win_rate']*100:.1f}% win rate",
                'suggestion': 'Raise confidence threshold above 0.60'
            })

    # 2. Analyze HMM regime impact
    cursor.execute("""
        SELECT
            CASE
                WHEN hmm_trend > 0.7 THEN 'bullish'
                WHEN hmm_trend < 0.3 THEN 'bearish'
                ELSE 'neutral'
            END as regime,
            option_type,
            COUNT(*) as trades,
            AVG(profit_loss) as avg_pnl,
            SUM(CASE WHEN profit_loss > 0 THEN 1.0 ELSE 0.0 END) / COUNT(*) as win_rate
        FROM trades
        WHERE hmm_trend IS NOT NULL AND timestamp > datetime('now', '-7 days')
        GROUP BY regime, option_type
        HAVING trades > 5
    """)
    regime_results = cursor.fetchall()
    for row in regime_results:
        if row['regime'] == 'bullish' and 'CALL' in str(row['option_type']) and row['win_rate'] > 0.5:
            insights.append({
                'type': 'regime_alignment',
                'finding': f"Bullish regime + CALL has {row['win_rate']*100:.1f}% win rate",
                'suggestion': 'Regime alignment working - keep strict'
            })
        if row['regime'] == 'neutral' and row['win_rate'] < 0.4:
            insights.append({
                'type': 'regime_filter',
                'finding': f"Neutral regime trades have only {row['win_rate']*100:.1f}% win rate",
                'suggestion': 'Filter out trades in neutral HMM regime (0.3-0.7)'
            })

    # 3. Analyze hold time impact
    cursor.execute("""
        SELECT
            CASE
                WHEN hold_minutes < 15 THEN 'quick (<15m)'
                WHEN hold_minutes < 30 THEN 'medium (15-30m)'
                ELSE 'long (>30m)'
            END as hold_bucket,
            COUNT(*) as trades,
            AVG(profit_loss) as avg_pnl,
            SUM(CASE WHEN profit_loss > 0 THEN 1.0 ELSE 0.0 END) / COUNT(*) as win_rate
        FROM trades
        WHERE hold_minutes IS NOT NULL AND timestamp > datetime('now', '-7 days')
        GROUP BY hold_bucket
        HAVING trades > 5
    """)
    hold_results = cursor.fetchall()
    for row in hold_results:
        if 'quick' in row['hold_bucket'] and row['avg_pnl'] > 0:
            insights.append({
                'type': 'hold_time',
                'finding': f"Quick trades (<15m) avg P&L: ${row['avg_pnl']:.2f}",
                'suggestion': 'Consider shorter max_hold_minutes'
            })
        if 'long' in row['hold_bucket'] and row['avg_pnl'] < 0:
            insights.append({
                'type': 'hold_time',
                'finding': f"Long trades (>30m) losing money: ${row['avg_pnl']:.2f}",
                'suggestion': 'Reduce max_hold_minutes to 30 or less'
            })

    # 4. Analyze max drawdown vs final P&L
    cursor.execute("""
        SELECT
            CASE
                WHEN max_drawdown_pct < -5 THEN 'deep_drawdown'
                WHEN max_drawdown_pct < -2 THEN 'moderate_drawdown'
                ELSE 'shallow_drawdown'
            END as dd_bucket,
            COUNT(*) as trades,
            AVG(profit_loss) as avg_pnl,
            SUM(CASE WHEN profit_loss > 0 THEN 1.0 ELSE 0.0 END) / COUNT(*) as win_rate
        FROM trades
        WHERE max_drawdown_pct IS NOT NULL AND timestamp > datetime('now', '-7 days')
        GROUP BY dd_bucket
        HAVING trades > 5
    """)
    dd_results = cursor.fetchall()
    for row in dd_results:
        if 'deep' in row['dd_bucket'] and row['win_rate'] > 0.3:
            insights.append({
                'type': 'stop_loss',
                'finding': f"Trades with >5% drawdown still win {row['win_rate']*100:.1f}%",
                'suggestion': 'Stop loss might be too tight - try wider stops'
            })

    conn.close()
    return insights


def generate_idea_from_insight(insight, next_id):
    """Convert an insight into an idea for the queue."""
    ideas = {
        'confidence_threshold': {
            'category': 'entry',
            'title': f"Adjust confidence threshold based on analysis",
            'description': insight['finding'],
            'hypothesis': insight['suggestion'],
            'env_vars': {'MIN_CONFIDENCE_TO_TRADE': '0.70'}
        },
        'regime_filter': {
            'category': 'entry',
            'title': f"Filter neutral regime trades",
            'description': insight['finding'],
            'hypothesis': insight['suggestion'],
            'env_vars': {'HMM_MIN_TREND_STRENGTH': '0.35'}
        },
        'hold_time': {
            'category': 'exit',
            'title': f"Optimize hold time based on analysis",
            'description': insight['finding'],
            'hypothesis': insight['suggestion'],
            'env_vars': {'MAX_HOLD_MINUTES': '25'}
        },
        'stop_loss': {
            'category': 'exit',
            'title': f"Adjust stop loss based on drawdown analysis",
            'description': insight['finding'],
            'hypothesis': insight['suggestion'],
            'env_vars': {'HARD_STOP_LOSS_PCT': '12'}
        }
    }

    template = ideas.get(insight['type'], {
        'category': 'optimization',
        'title': insight['finding'][:50],
        'description': insight['finding'],
        'hypothesis': insight['suggestion'],
        'env_vars': {}
    })

    return {
        'id': f'IDEA-{next_id:03d}',
        'source': 'tuning_analyzer',
        'timestamp': datetime.now().isoformat(),
        'category': template['category'],
        'title': template['title'],
        'description': template['description'],
        'hypothesis': template['hypothesis'],
        'priority': 'high',
        'status': 'pending',
        'env_vars': template.get('env_vars', {})
    }


def submit_idea(idea):
    """Add idea to the queue."""
    try:
        with open(IDEA_QUEUE, 'r') as f:
            queue = json.load(f)
    except:
        queue = {'ideas': [], 'next_id': 100}

    # Check for duplicates
    existing_titles = {i.get('title', '') for i in queue['ideas']}
    if idea['title'] in existing_titles:
        logger.info(f"Idea already exists: {idea['title']}")
        return False

    queue['ideas'].insert(0, idea)
    queue['next_id'] = max(queue.get('next_id', 100), int(idea['id'].split('-')[1]) + 1)
    queue['last_updated'] = datetime.now().isoformat()

    with open(IDEA_QUEUE, 'w') as f:
        json.dump(queue, f, indent=2)

    logger.info(f"Submitted idea: {idea['id']} - {idea['title']}")
    return True


def monitor_and_improve():
    """Main monitoring loop."""
    logger.info("=" * 60)
    logger.info("Tuning Analyzer started - monitoring for improvements")
    logger.info("=" * 60)

    last_analysis = datetime.now() - timedelta(hours=1)
    ideas_submitted = 0

    while True:
        try:
            # Check running experiment
            exp = get_running_experiment()
            if exp:
                progress = exp.get('current_cycle', 0) / max(exp.get('target_cycles', 1), 1) * 100
                logger.info(f"Running: {exp.get('run_name')} - {progress:.1f}% complete, P&L: ${exp.get('current_pnl', 0):.2f}")

            # Analyze tuning data every 30 minutes
            if datetime.now() - last_analysis > timedelta(minutes=30):
                logger.info("Running tuning data analysis...")
                insights = analyze_tuning_data()

                if insights:
                    logger.info(f"Found {len(insights)} insights from tuning data")
                    for insight in insights[:3]:  # Max 3 ideas per cycle
                        try:
                            with open(IDEA_QUEUE, 'r') as f:
                                queue = json.load(f)
                            next_id = queue.get('next_id', 100)
                        except:
                            next_id = 100

                        idea = generate_idea_from_insight(insight, next_id)
                        if submit_idea(idea):
                            ideas_submitted += 1
                else:
                    logger.info("No new insights from tuning data (may need more trades)")

                last_analysis = datetime.now()

            # Sync experiments to database for dashboard visibility
            sync_experiments_db()

            # Status update
            logger.info(f"Ideas submitted this session: {ideas_submitted}")
            logger.info("Sleeping for 10 minutes...")
            time.sleep(600)  # 10 minutes

        except KeyboardInterrupt:
            logger.info("Shutting down...")
            break
        except Exception as e:
            logger.error(f"Error in monitoring loop: {e}")
            time.sleep(60)


if __name__ == "__main__":
    monitor_and_improve()
