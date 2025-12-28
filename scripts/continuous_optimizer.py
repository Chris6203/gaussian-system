#!/usr/bin/env python3
"""
Continuous Optimizer: Claude-Codex Distributed Experimentation System

Runs 24/7, testing improvements in parallel. Supports multiple machines
syncing via GitHub to avoid duplicate work and share findings.

Usage:
    python scripts/continuous_optimizer.py
    python scripts/continuous_optimizer.py --dry-run  # Test without running experiments
    python scripts/continuous_optimizer.py --single   # Run one batch then exit

Environment Variables:
    QUICK_TEST_CYCLES    - Cycles for quick test (default: 5000)
    VALIDATION_CYCLES    - Cycles for validation (default: 20000)
    MIN_PER_TRADE_PNL    - Minimum per-trade P&L to pass (default: 0.0)
    PARALLEL_EXPERIMENTS - Number of parallel experiments (default: 4)
    CODEX_MODEL          - Model for Codex (default: o3)
    MACHINE_ID           - Unique machine identifier (default: hostname)
    GIT_SYNC_ENABLED     - Enable GitHub sync for distributed mode (default: 1)
"""

import os
import sys
import json
import subprocess
import time
import re
import shutil
import socket
import hashlib
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, List, Any, Set
from concurrent.futures import ProcessPoolExecutor, as_completed
import logging
import argparse

# Configuration
BASE_DIR = Path(__file__).parent.parent.resolve()
COLLAB_DIR = BASE_DIR / ".claude" / "collab"
QUICK_TEST_CYCLES = int(os.environ.get("QUICK_TEST_CYCLES", 5000))
VALIDATION_CYCLES = int(os.environ.get("VALIDATION_CYCLES", 20000))
MIN_PER_TRADE_PNL = float(os.environ.get("MIN_PER_TRADE_PNL", 0.0))
PARALLEL_EXPERIMENTS = int(os.environ.get("PARALLEL_EXPERIMENTS", 4))
PAUSE_BETWEEN_BATCHES = 60  # seconds

# Distributed mode settings
MACHINE_ID = os.environ.get("MACHINE_ID", socket.gethostname()[:20])
GIT_SYNC_ENABLED = os.environ.get("GIT_SYNC_ENABLED", "1") == "1"
LOCK_TIMEOUT_MINUTES = 120  # Consider locks stale after 2 hours

# Setup logging
log_dir = BASE_DIR / "logs"
log_dir.mkdir(exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_dir / "continuous_optimizer.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


# =============================================================================
# DISTRIBUTED SYNC FUNCTIONS
# =============================================================================

def git_pull_latest() -> bool:
    """Pull latest changes from GitHub. Returns True if successful."""
    if not GIT_SYNC_ENABLED:
        return True

    try:
        # Stash any local changes first
        subprocess.run(
            ["git", "stash", "--include-untracked"],
            cwd=str(BASE_DIR),
            capture_output=True,
            timeout=30
        )

        # Pull with rebase to avoid merge commits
        result = subprocess.run(
            ["git", "pull", "--rebase", "origin", "main"],
            cwd=str(BASE_DIR),
            capture_output=True,
            text=True,
            timeout=60
        )

        # Pop stash if we had local changes
        subprocess.run(
            ["git", "stash", "pop"],
            cwd=str(BASE_DIR),
            capture_output=True,
            timeout=30
        )

        if result.returncode == 0:
            if "Already up to date" not in result.stdout:
                logger.info(f"[SYNC] Pulled latest changes from GitHub")
            return True
        else:
            logger.warning(f"[SYNC] Git pull failed: {result.stderr}")
            return False

    except subprocess.TimeoutExpired:
        logger.warning("[SYNC] Git pull timed out")
        return False
    except Exception as e:
        logger.warning(f"[SYNC] Git pull error: {e}")
        return False


def git_push_changes(message: str) -> bool:
    """Push changes to GitHub. Returns True if successful."""
    if not GIT_SYNC_ENABLED:
        return True

    try:
        # Stage collaboration files
        subprocess.run(
            ["git", "add",
             ".claude/collab/",
             "RESULTS_TRACKER.md"],
            cwd=str(BASE_DIR),
            capture_output=True,
            timeout=30
        )

        # Check if there are changes
        result = subprocess.run(
            ["git", "diff", "--cached", "--quiet"],
            cwd=str(BASE_DIR),
            capture_output=True
        )

        if result.returncode == 0:
            # No changes to commit
            return True

        # Commit with machine ID
        full_message = f"[{MACHINE_ID}] {message}"
        subprocess.run(
            ["git", "commit", "-m", full_message],
            cwd=str(BASE_DIR),
            capture_output=True,
            timeout=30
        )

        # Push (may need to pull first if remote changed)
        for attempt in range(3):
            result = subprocess.run(
                ["git", "push", "origin", "main"],
                cwd=str(BASE_DIR),
                capture_output=True,
                text=True,
                timeout=60
            )

            if result.returncode == 0:
                logger.info(f"[SYNC] Pushed: {message}")
                return True
            elif "rejected" in result.stderr or "failed to push" in result.stderr:
                # Need to pull first
                logger.info("[SYNC] Remote has changes, pulling and retrying...")
                git_pull_latest()
            else:
                break

        logger.warning(f"[SYNC] Push failed after retries: {result.stderr}")
        return False

    except Exception as e:
        logger.warning(f"[SYNC] Push error: {e}")
        return False


def idea_to_hash(idea: dict) -> str:
    """Create a unique hash for an idea based on its configuration."""
    # Hash based on env_vars (the actual configuration being tested)
    env_vars = idea.get("env_vars", {})
    config_str = json.dumps(env_vars, sort_keys=True)
    return hashlib.md5(config_str.encode()).hexdigest()[:12]


def get_tested_idea_hashes() -> Set[str]:
    """Get hashes of all ideas that have been tested (from experiment log)."""
    tested = set()
    log_path = COLLAB_DIR / "experiments" / "experiment_log.jsonl"

    if not log_path.exists():
        return tested

    try:
        for line in log_path.read_text(encoding='utf-8').strip().split("\n"):
            if not line:
                continue
            try:
                record = json.loads(line)
                # Only count experiments that actually ran (have results)
                result = record.get("result", {})
                if result.get("win_rate") is not None or result.get("trades") is not None:
                    env_vars = record.get("env_vars", {})
                    config_str = json.dumps(env_vars, sort_keys=True)
                    hash_val = hashlib.md5(config_str.encode()).hexdigest()[:12]
                    tested.add(hash_val)
            except:
                pass
    except:
        pass

    return tested


def get_locked_ideas() -> Dict[str, dict]:
    """Get ideas currently being tested by other machines."""
    locks = {}
    queue = load_json(COLLAB_DIR / "idea_queue.json")

    now = datetime.now()

    for idea in queue.get("ideas", []):
        if idea.get("status") == "in_progress":
            lock_info = idea.get("lock", {})
            lock_time_str = lock_info.get("locked_at")
            lock_machine = lock_info.get("machine_id")

            if lock_time_str and lock_machine:
                try:
                    lock_time = datetime.fromisoformat(lock_time_str)
                    minutes_locked = (now - lock_time).total_seconds() / 60

                    # Only honor locks that aren't stale
                    if minutes_locked < LOCK_TIMEOUT_MINUTES:
                        locks[idea.get("id")] = {
                            "machine": lock_machine,
                            "since": lock_time_str,
                            "minutes": int(minutes_locked)
                        }
                except:
                    pass

    return locks


def claim_ideas(ideas: List[dict]) -> List[dict]:
    """Claim ideas by marking them in_progress with our machine ID."""
    queue = load_json(COLLAB_DIR / "idea_queue.json")
    stored_ideas = queue.get("ideas", [])
    claimed = []

    now = datetime.now().isoformat()

    for idea in ideas:
        for i, stored in enumerate(stored_ideas):
            if stored.get("id") == idea.get("id"):
                stored_ideas[i]["status"] = "in_progress"
                stored_ideas[i]["lock"] = {
                    "machine_id": MACHINE_ID,
                    "locked_at": now
                }
                claimed.append(idea)
                break

    queue["ideas"] = stored_ideas
    queue["last_updated"] = now
    save_json(COLLAB_DIR / "idea_queue.json", queue)

    # Push claim immediately so other machines see it
    git_push_changes(f"Claimed {len(claimed)} ideas for testing")

    return claimed


def release_idea_locks(idea_ids: List[str]):
    """Release locks on ideas (after testing completes)."""
    queue = load_json(COLLAB_DIR / "idea_queue.json")

    for idea in queue.get("ideas", []):
        if idea.get("id") in idea_ids:
            if "lock" in idea:
                del idea["lock"]

    save_json(COLLAB_DIR / "idea_queue.json", queue)


# =============================================================================
# CORE FUNCTIONS
# =============================================================================

def load_json(path: Path) -> dict:
    """Load JSON file, return empty dict if not exists."""
    if path.exists():
        return json.loads(path.read_text(encoding='utf-8'))
    return {}


def save_json(path: Path, data: dict):
    """Save JSON file with pretty printing."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding='utf-8')


def append_jsonl(path: Path, record: dict):
    """Append record to JSONL file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(record) + "\n")


def get_next_experiment_id() -> str:
    """Generate next experiment ID based on experiment log."""
    log_path = COLLAB_DIR / "experiments" / "experiment_log.jsonl"
    if not log_path.exists():
        return "EXP-0001"

    lines = log_path.read_text(encoding='utf-8').strip().split("\n")
    if not lines or not lines[0]:
        return "EXP-0001"

    # Find highest ID
    max_id = 0
    for line in lines:
        if line:
            try:
                record = json.loads(line)
                exp_id = record.get("id", "EXP-0000")
                num = int(exp_id.split("-")[1])
                max_id = max(max_id, num)
            except:
                pass

    return f"EXP-{max_id + 1:04d}"


def run_codex(prompt: str, timeout: int = 600) -> Optional[str]:
    """Run Codex CLI and return output."""
    try:
        cmd = [
            "codex",
            "--full-auto",
            "-m", os.environ.get("CODEX_MODEL", "o3"),
            prompt
        ]
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=str(BASE_DIR)
        )
        return result.stdout
    except subprocess.TimeoutExpired:
        logger.warning("Codex timed out")
        return None
    except FileNotFoundError:
        logger.warning("Codex CLI not found, skipping...")
        return None
    except Exception as e:
        logger.error(f"Codex error: {e}")
        return None


def ask_codex_for_ideas() -> List[dict]:
    """Ask Codex for new improvement ideas."""
    logger.info("Asking Codex for improvement ideas...")

    # Load current state
    learnings = load_json(COLLAB_DIR / "learnings.json")

    # Read prompt template
    prompt_path = COLLAB_DIR / "prompts" / "ask_codex_ideas.txt"
    if not prompt_path.exists():
        logger.warning("Prompt template not found")
        return []

    prompt_template = prompt_path.read_text(encoding='utf-8')

    # Get recent experiments
    experiment_log = COLLAB_DIR / "experiments" / "experiment_log.jsonl"
    recent = []
    if experiment_log.exists():
        lines = experiment_log.read_text(encoding='utf-8').strip().split("\n")[-20:]
        for line in lines:
            if line:
                try:
                    exp = json.loads(line)
                    recent.append({
                        "title": exp.get("title", "Unknown"),
                        "status": exp.get("final_status", "unknown"),
                        "per_trade_pnl": exp.get("quick_test", {}).get("per_trade_pnl", "N/A")
                    })
                except:
                    pass

    # Build prompt
    prompt = prompt_template.replace(
        "{INSERT_LEARNINGS}",
        json.dumps(learnings.get("failed_approaches", [])[-10:], indent=2)
    ).replace(
        "{INSERT_RECENT_EXPERIMENTS}",
        json.dumps(recent, indent=2)
    )

    # Call Codex
    output = run_codex(prompt)
    if not output:
        return []

    # Parse ideas from output
    try:
        # Look for JSON in output
        start = output.find("{")
        end = output.rfind("}") + 1
        if start >= 0 and end > start:
            data = json.loads(output[start:end])
            ideas = data.get("new_ideas", [])
            if ideas:
                logger.info(f"Codex suggested {len(ideas)} new ideas")
                return ideas
    except Exception as e:
        logger.warning(f"Could not parse Codex output: {e}")

    return []


def select_batch_of_ideas(count: int = 4) -> List[dict]:
    """Select next batch of ideas to test in parallel.

    In distributed mode, this:
    1. Pulls latest from GitHub to see other machines' progress
    2. Skips ideas already tested (based on config hash)
    3. Skips ideas locked by other machines
    4. Claims selected ideas with our machine ID
    """
    # Sync with remote first
    git_pull_latest()

    queue = load_json(COLLAB_DIR / "idea_queue.json")
    ideas = queue.get("ideas", [])

    # Get already-tested configurations (by hash)
    tested_hashes = get_tested_idea_hashes()
    logger.info(f"[SYNC] {len(tested_hashes)} configurations already tested")

    # Get ideas locked by other machines
    locked_ideas = get_locked_ideas()
    if locked_ideas:
        logger.info(f"[SYNC] {len(locked_ideas)} ideas locked by other machines:")
        for idea_id, lock in locked_ideas.items():
            logger.info(f"       {idea_id} by {lock['machine']} ({lock['minutes']}m ago)")

    # Find pending ideas that haven't been tested and aren't locked
    available = []
    for idea in ideas:
        if idea.get("status") != "pending":
            continue

        # Skip if locked by another machine
        if idea.get("id") in locked_ideas:
            continue

        # Skip if this configuration was already tested
        idea_hash = idea_to_hash(idea)
        if idea_hash in tested_hashes:
            logger.info(f"[SYNC] Skipping {idea.get('id')} - already tested (hash: {idea_hash})")
            # Mark as tested
            idea["status"] = "already_tested"
            continue

        available.append(idea)

    # Update queue with any status changes
    save_json(COLLAB_DIR / "idea_queue.json", queue)

    if len(available) < count:
        # Ask Codex for more ideas
        logger.info(f"[SYNC] Only {len(available)} ideas available, asking Codex for more...")
        new_ideas = ask_codex_for_ideas()
        for idea in new_ideas:
            # Assign ID if not present
            if "id" not in idea:
                idea["id"] = f"IDEA-{queue.get('next_id', len(ideas) + 1):04d}"
                queue["next_id"] = queue.get("next_id", len(ideas) + 1) + 1
            idea["status"] = "pending"

            # Check if this is a duplicate configuration
            idea_hash = idea_to_hash(idea)
            if idea_hash not in tested_hashes:
                ideas.append(idea)
                available.append(idea)
            else:
                logger.info(f"[SYNC] Skipping Codex idea - config already tested")

        queue["ideas"] = ideas
        queue["last_updated"] = datetime.now().isoformat()
        save_json(COLLAB_DIR / "idea_queue.json", queue)

    # Select up to 'count' ideas
    batch = available[:count]

    if not batch:
        return []

    # Claim them with our machine ID
    claimed = claim_ideas(batch)
    logger.info(f"[SYNC] Claimed {len(claimed)} ideas for {MACHINE_ID}")

    return claimed


def run_experiment(idea: dict, exp_id: str, cycles: int, gpu_id: int = 0) -> dict:
    """Run a single experiment (called in subprocess)."""
    run_name = f"{exp_id}_{idea.get('id', 'unknown')}"
    run_dir = f"models/{run_name}"

    # Build environment
    env = os.environ.copy()
    env["MODEL_RUN_DIR"] = str(BASE_DIR / run_dir)
    env["TT_MAX_CYCLES"] = str(cycles)
    env["TT_PRINT_EVERY"] = str(max(cycles // 10, 100))
    env["PAPER_TRADING"] = "True"
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    # Apply idea's env_vars
    for key, value in idea.get("env_vars", {}).items():
        env[key] = str(value)

    logger.info(f"[GPU {gpu_id}] Running {exp_id}: {idea.get('title', 'Unknown')}")

    try:
        result = subprocess.run(
            [sys.executable, str(BASE_DIR / "scripts" / "train_time_travel.py")],
            env=env,
            cwd=str(BASE_DIR),
            capture_output=True,
            text=True,
            timeout=7200  # 2 hour timeout
        )

        # Parse results from SUMMARY.txt
        summary_path = BASE_DIR / run_dir / "SUMMARY.txt"
        if summary_path.exists():
            return parse_summary(summary_path, run_dir)
        else:
            # Try to extract from stdout
            return parse_stdout(result.stdout, run_dir)

    except subprocess.TimeoutExpired:
        logger.error(f"[GPU {gpu_id}] Experiment {exp_id} timed out")
        return {"error": "timeout", "run_dir": run_dir}
    except Exception as e:
        logger.error(f"[GPU {gpu_id}] Experiment {exp_id} failed: {e}")
        return {"error": str(e), "run_dir": run_dir}


def parse_summary(summary_path: Path, run_dir: str) -> dict:
    """Parse SUMMARY.txt to extract metrics."""
    text = summary_path.read_text(encoding='utf-8')
    result = {"run_dir": run_dir}

    # Extract metrics using regex patterns
    patterns = {
        "win_rate": r"Win Rate[:\s]+(\d+\.?\d*)%?",
        "pnl_dollars": r"P&L[:\s]+\$?([-\d,]+\.?\d*)",
        "pnl_pct": r"P&L.*\(([-\d.]+)%\)",
        "trades": r"(?:Total )?Trades[:\s]+(\d+)",
        "cycles": r"Cycles[:\s]+(\d+)"
    }

    for key, pattern in patterns.items():
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            value = match.group(1).replace(",", "")
            try:
                result[key] = float(value) if "." in value else int(value)
            except:
                pass

    # Calculate per-trade P&L
    if "pnl_dollars" in result and "trades" in result and result["trades"] > 0:
        result["per_trade_pnl"] = result["pnl_dollars"] / result["trades"]

    return result


def parse_stdout(stdout: str, run_dir: str) -> dict:
    """Parse stdout for metrics if SUMMARY.txt not available."""
    result = {"run_dir": run_dir}

    # Look for final results in stdout
    lines = stdout.split("\n")
    for line in lines:
        if "Win Rate:" in line:
            match = re.search(r"(\d+\.?\d*)%", line)
            if match:
                result["win_rate"] = float(match.group(1))
        elif "P&L:" in line and "$" in line:
            match = re.search(r"\$([-\d,]+\.?\d*)", line)
            if match:
                result["pnl_dollars"] = float(match.group(1).replace(",", ""))
        elif "Total Trades:" in line or "Trades:" in line:
            match = re.search(r"(\d+)", line)
            if match:
                result["trades"] = int(match.group(1))

    # Calculate per-trade P&L
    if "pnl_dollars" in result and result.get("trades", 0) > 0:
        result["per_trade_pnl"] = result["pnl_dollars"] / result["trades"]

    return result


def run_parallel_experiments(ideas: List[dict], cycles: int) -> List[tuple]:
    """Run experiments in parallel across GPUs."""
    results = []
    exp_base_id = get_next_experiment_id()
    base_num = int(exp_base_id.split("-")[1])

    # Assign GPU IDs (round-robin across 2 GPUs)
    with ProcessPoolExecutor(max_workers=min(len(ideas), PARALLEL_EXPERIMENTS)) as executor:
        futures = {}
        for i, idea in enumerate(ideas):
            exp_id = f"EXP-{base_num + i:04d}"
            gpu_id = i % 2  # Alternate between GPU 0 and 1
            future = executor.submit(run_experiment, idea, exp_id, cycles, gpu_id)
            futures[future] = (idea, exp_id)

        for future in as_completed(futures):
            idea, exp_id = futures[future]
            try:
                result = future.result()
                results.append((idea, exp_id, result))
                logger.info(f"Completed {exp_id}: per-trade P&L = ${result.get('per_trade_pnl', 'N/A')}")
            except Exception as e:
                logger.error(f"Failed {exp_id}: {e}")
                results.append((idea, exp_id, {"error": str(e)}))

    return results


def update_scoreboard(experiments: List[tuple], is_validation: bool = False):
    """Update RESULTS_TRACKER.md with experiment results."""
    tracker_path = BASE_DIR / "RESULTS_TRACKER.md"
    content = tracker_path.read_text(encoding='utf-8')

    # Find or create automated section
    section_header = "## Automated Optimization Results"
    if section_header not in content:
        content = content + f"\n\n{section_header}\n\n"

    # Build entries
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    test_type = "Validation (20K)" if is_validation else "Quick Test (5K)"

    entries = []
    for idea, exp_id, result in experiments:
        per_trade = result.get("per_trade_pnl", "N/A")
        if isinstance(per_trade, (int, float)):
            status = "PASS" if per_trade > MIN_PER_TRADE_PNL else "FAIL"
            per_trade_str = f"${per_trade:.2f}"
        else:
            status = "ERROR"
            per_trade_str = "N/A"

        entry = f"""
### {exp_id}: {idea.get('title', 'Unknown')} ({timestamp})

| Metric | {test_type} |
|--------|------------|
| Win Rate | {result.get('win_rate', 'N/A')}% |
| P&L | {result.get('pnl_pct', 'N/A')}% |
| Per-Trade P&L | {per_trade_str} |
| Trades | {result.get('trades', 'N/A')} |
| Run Dir | `{result.get('run_dir', 'N/A')}` |

**Source**: {idea.get('source', 'claude').upper()}
**Category**: {idea.get('category', 'unknown')}
**Hypothesis**: {idea.get('hypothesis', 'N/A')}
**Result**: {status}

---
"""
        entries.append(entry)

    # Insert after section header
    idx = content.find(section_header) + len(section_header)
    content = content[:idx] + "\n" + "".join(entries) + content[idx:]

    tracker_path.write_text(content, encoding='utf-8')
    logger.info(f"Updated scoreboard with {len(experiments)} experiments")


def record_experiments(experiments: List[tuple], is_validation: bool = False):
    """Record experiments to JSONL log with machine ID for distributed tracking."""
    for idea, exp_id, result in experiments:
        passed = result.get("per_trade_pnl", -999) > MIN_PER_TRADE_PNL

        record = {
            "id": exp_id,
            "timestamp": datetime.now().isoformat(),
            "machine_id": MACHINE_ID,  # Track which machine ran this
            "idea_id": idea.get("id"),
            "config_hash": idea_to_hash(idea),  # For deduplication
            "source": idea.get("source", "unknown"),
            "category": idea.get("category", "unknown"),
            "title": idea.get("title", "Unknown"),
            "hypothesis": idea.get("hypothesis", ""),
            "env_vars": idea.get("env_vars", {}),
            "test_type": "validation" if is_validation else "quick",
            "result": result,
            "passed": passed,
            "final_status": "accepted" if (is_validation and passed) else ("pending_validation" if passed else "rejected")
        }

        append_jsonl(COLLAB_DIR / "experiments" / "experiment_log.jsonl", record)


def update_idea_status(ideas: List[dict], results: List[tuple]):
    """Update idea statuses based on results."""
    queue = load_json(COLLAB_DIR / "idea_queue.json")
    stored_ideas = queue.get("ideas", [])

    for idea, exp_id, result in results:
        passed = result.get("per_trade_pnl", -999) > MIN_PER_TRADE_PNL
        new_status = "passed_quick" if passed else "rejected"

        for i, stored in enumerate(stored_ideas):
            if stored.get("id") == idea.get("id"):
                stored_ideas[i]["status"] = new_status
                stored_ideas[i]["last_result"] = {
                    "exp_id": exp_id,
                    "per_trade_pnl": result.get("per_trade_pnl"),
                    "win_rate": result.get("win_rate")
                }

    queue["ideas"] = stored_ideas
    queue["last_updated"] = datetime.now().isoformat()
    save_json(COLLAB_DIR / "idea_queue.json", queue)


def git_commit_and_push(message: str):
    """Commit changes and push to GitHub."""
    try:
        # Pull latest first to avoid conflicts
        subprocess.run(["git", "pull", "--rebase"], cwd=str(BASE_DIR), capture_output=True)

        # Stage changes
        subprocess.run(["git", "add", "-A"], cwd=str(BASE_DIR), check=True)

        # Check if there are changes to commit
        result = subprocess.run(
            ["git", "diff", "--cached", "--quiet"],
            cwd=str(BASE_DIR),
            capture_output=True
        )

        if result.returncode != 0:  # There are changes
            # Commit
            full_message = f"{message}\n\nGenerated by Continuous Optimizer\nCo-Authored-By: Claude Code <noreply@anthropic.com>"
            subprocess.run(
                ["git", "commit", "-m", full_message],
                cwd=str(BASE_DIR),
                check=True
            )

            # Push
            subprocess.run(["git", "push"], cwd=str(BASE_DIR), check=True)
            logger.info(f"Committed and pushed: {message}")
        else:
            logger.info("No changes to commit")

    except subprocess.CalledProcessError as e:
        logger.warning(f"Git operation failed: {e}")


def main_loop(dry_run: bool = False, single_run: bool = False):
    """Main optimization loop with distributed sync support."""
    logger.info("=" * 60)
    logger.info("Starting Continuous Optimizer (Distributed Mode)")
    logger.info(f"Machine ID: {MACHINE_ID}")
    logger.info(f"Git Sync: {'ENABLED' if GIT_SYNC_ENABLED else 'DISABLED'}")
    logger.info(f"Parallel experiments: {PARALLEL_EXPERIMENTS}")
    logger.info(f"Quick test cycles: {QUICK_TEST_CYCLES}")
    logger.info(f"Validation cycles: {VALIDATION_CYCLES}")
    logger.info(f"Min per-trade P&L: ${MIN_PER_TRADE_PNL}")
    logger.info("=" * 60)

    # Initial sync
    if GIT_SYNC_ENABLED:
        logger.info("[SYNC] Pulling latest from GitHub...")
        git_pull_latest()

    iteration = 0
    while True:
        iteration += 1
        logger.info(f"\n{'='*60}")
        logger.info(f"=== [{MACHINE_ID}] Iteration {iteration} ===")
        logger.info(f"{'='*60}\n")

        try:
            # 1. Select batch of ideas (includes git sync)
            batch = select_batch_of_ideas(PARALLEL_EXPERIMENTS)
            if not batch:
                logger.info("No ideas available. Waiting 5 minutes...")
                if single_run:
                    break
                time.sleep(300)
                continue

            logger.info(f"Selected {len(batch)} ideas for testing:")
            for idea in batch:
                logger.info(f"  - {idea.get('id')}: {idea.get('title')}")

            if dry_run:
                logger.info("[DRY RUN] Would run experiments here")
                # Release locks
                release_idea_locks([i.get("id") for i in batch])
                if single_run:
                    break
                continue

            # 2. Run quick tests in parallel
            logger.info(f"\nRunning {len(batch)} quick tests ({QUICK_TEST_CYCLES} cycles each)...")
            quick_results = run_parallel_experiments(batch, QUICK_TEST_CYCLES)

            # 3. Record and update scoreboard
            record_experiments(quick_results, is_validation=False)
            update_scoreboard(quick_results, is_validation=False)
            update_idea_status(batch, quick_results)

            # 4. Release locks on tested ideas
            release_idea_locks([i.get("id") for i in batch])

            # 5. Identify winners for validation
            winners = [
                (idea, exp_id, result)
                for idea, exp_id, result in quick_results
                if result.get("per_trade_pnl", -999) > MIN_PER_TRADE_PNL
            ]

            logger.info(f"\nQuick test results: {len(winners)}/{len(batch)} passed")

            # 6. Push quick test results immediately
            git_push_changes(f"Quick tests: {len(winners)}/{len(batch)} passed")

            # 7. Run validation on winners
            if winners:
                logger.info(f"\nRunning validation ({VALIDATION_CYCLES} cycles) on {len(winners)} winners...")
                val_ideas = [w[0] for w in winners]
                val_results = run_parallel_experiments(val_ideas, VALIDATION_CYCLES)

                record_experiments(val_results, is_validation=True)
                update_scoreboard(val_results, is_validation=True)

                # Check for validated improvements
                validated = [
                    (idea, exp_id, result)
                    for idea, exp_id, result in val_results
                    if result.get("per_trade_pnl", -999) > MIN_PER_TRADE_PNL
                ]

                if validated:
                    logger.info(f"\n*** {len(validated)} VALIDATED IMPROVEMENTS FOUND! ***")
                    for idea, exp_id, result in validated:
                        logger.info(f"  {exp_id}: {idea.get('title')} - ${result.get('per_trade_pnl', 'N/A'):.2f}/trade")

                # Push validation results
                git_push_changes(f"Validation: {len(validated)}/{len(winners)} passed")

            if single_run:
                logger.info("Single run complete. Exiting.")
                break

            # 8. Pause before next iteration
            logger.info(f"\nIteration {iteration} complete. Pausing {PAUSE_BETWEEN_BATCHES}s...")
            time.sleep(PAUSE_BETWEEN_BATCHES)

        except KeyboardInterrupt:
            logger.info("\nInterrupted by user. Exiting...")
            # Release any held locks
            if 'batch' in locals():
                release_idea_locks([i.get("id") for i in batch])
            break
        except Exception as e:
            logger.error(f"Error in iteration {iteration}: {e}")
            import traceback
            traceback.print_exc()
            # Release locks on error
            if 'batch' in locals():
                release_idea_locks([i.get("id") for i in batch])
            time.sleep(60)  # Wait before retrying


def main():
    parser = argparse.ArgumentParser(description="Continuous Optimizer for Gaussian Trading Bot")
    parser.add_argument("--dry-run", action="store_true", help="Test without running experiments")
    parser.add_argument("--single", action="store_true", help="Run one batch then exit")
    args = parser.parse_args()

    main_loop(dry_run=args.dry_run, single_run=args.single)


if __name__ == "__main__":
    main()
