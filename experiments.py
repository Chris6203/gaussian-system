#!/usr/bin/env python3
"""
Experiment System
=================
Two-layer automated optimization system for testing trading configurations.

Layer 1 (Continuous Optimizer): Tests specific configurations from the queue
Layer 2 (Meta Optimizer): Analyzes results and suggests new configurations

Usage:
    python experiments.py                    # Run Layer 1 (continuous optimizer)
    python experiments.py --meta             # Run Layer 2 (meta optimizer)
    python experiments.py --both             # Run both layers
    python experiments.py --status           # Show experiment status
    python experiments.py --queue            # Show idea queue
    python experiments.py --add "hypothesis" # Add experiment idea

Configuration:
    Edit .claude/collab/idea_queue.json for experiment ideas
    Edit config.json for experiment parameters
"""

import sys
import os
import argparse
import subprocess
import signal
import time
import json
from pathlib import Path
from datetime import datetime
from typing import List, Optional

# Add project root to path
BASE_DIR = Path(__file__).parent.resolve()
sys.path.insert(0, str(BASE_DIR))

# Paths
COLLAB_DIR = BASE_DIR / ".claude" / "collab"
QUEUE_FILE = COLLAB_DIR / "idea_queue.json"
MODELS_DIR = BASE_DIR / "models"

# Track running processes
processes: List[subprocess.Popen] = []


def signal_handler(sig, frame):
    """Handle shutdown signals."""
    print("\nShutting down experiments...")
    for proc in processes:
        if proc and proc.poll() is None:
            proc.terminate()
    sys.exit(0)


def show_status():
    """Show experiment system status."""
    print("\n=== Experiment System Status ===\n")

    # Count experiments
    if MODELS_DIR.exists():
        runs = [d for d in MODELS_DIR.iterdir() if d.is_dir() and d.name.startswith("run_")]
        analyzed = [d for d in runs if (d / "ANALYSIS.md").exists()]
        print(f"Total experiments: {len(runs)}")
        print(f"With analysis: {len(analyzed)}")
    else:
        print("No models directory found")

    # Queue status
    if QUEUE_FILE.exists():
        try:
            queue = json.loads(QUEUE_FILE.read_text())
            ideas = queue.get("ideas", [])
            pending = [i for i in ideas if i.get("status") == "pending"]
            running = [i for i in ideas if i.get("status") == "running"]
            completed = [i for i in ideas if i.get("status") == "completed"]

            print(f"\nIdea Queue:")
            print(f"  Pending: {len(pending)}")
            print(f"  Running: {len(running)}")
            print(f"  Completed: {len(completed)}")
        except:
            print("\nCould not read idea queue")

    # Check for running processes
    print("\nRunning optimizers:")

    # Check Layer 1
    layer1_log = BASE_DIR / "logs" / "continuous_optimizer.log"
    if layer1_log.exists():
        mtime = datetime.fromtimestamp(layer1_log.stat().st_mtime)
        age_minutes = (datetime.now() - mtime).total_seconds() / 60
        if age_minutes < 5:
            print(f"  Layer 1 (Continuous): ACTIVE (log updated {age_minutes:.1f} min ago)")
        else:
            print(f"  Layer 1 (Continuous): INACTIVE (last update {age_minutes:.0f} min ago)")
    else:
        print("  Layer 1 (Continuous): NOT STARTED")

    # Check Layer 2
    import glob
    meta_logs = glob.glob("/tmp/meta_optimizer*.log")
    if meta_logs:
        latest = max(meta_logs, key=os.path.getmtime)
        mtime = datetime.fromtimestamp(os.path.getmtime(latest))
        age_minutes = (datetime.now() - mtime).total_seconds() / 60
        if age_minutes < 35:  # Meta runs every 30 min
            print(f"  Layer 2 (Meta): ACTIVE (log updated {age_minutes:.1f} min ago)")
        else:
            print(f"  Layer 2 (Meta): INACTIVE (last update {age_minutes:.0f} min ago)")
    else:
        print("  Layer 2 (Meta): NOT STARTED")

    print()


def show_queue():
    """Show the idea queue."""
    if not QUEUE_FILE.exists():
        print("No idea queue found")
        return

    try:
        queue = json.loads(QUEUE_FILE.read_text())
        ideas = queue.get("ideas", [])

        print("\n=== Experiment Idea Queue ===\n")

        pending = [i for i in ideas if i.get("status") == "pending"]
        if pending:
            print(f"PENDING ({len(pending)}):")
            for idea in pending[:10]:
                print(f"  [{idea.get('id', '?')}] {idea.get('hypothesis', 'No hypothesis')[:60]}")
                if idea.get('env_vars'):
                    print(f"       env: {dict(list(idea['env_vars'].items())[:3])}")

        running = [i for i in ideas if i.get("status") == "running"]
        if running:
            print(f"\nRUNNING ({len(running)}):")
            for idea in running:
                print(f"  [{idea.get('id', '?')}] {idea.get('hypothesis', 'No hypothesis')[:60]}")

        print()

    except Exception as e:
        print(f"Error reading queue: {e}")


def add_idea(hypothesis: str, env_vars: dict = None):
    """Add a new experiment idea to the queue."""
    COLLAB_DIR.mkdir(parents=True, exist_ok=True)

    if QUEUE_FILE.exists():
        queue = json.loads(QUEUE_FILE.read_text())
    else:
        queue = {"ideas": [], "last_updated": ""}

    # Find max ID
    max_id = 0
    for idea in queue.get("ideas", []):
        if "IDEA-" in idea.get("id", ""):
            try:
                num = int(idea["id"].split("-")[1])
                max_id = max(max_id, num)
            except:
                pass

    new_idea = {
        "id": f"IDEA-{max_id + 1}",
        "source": "manual",
        "timestamp": datetime.now().isoformat(),
        "hypothesis": hypothesis,
        "env_vars": env_vars or {},
        "status": "pending",
        "priority": 1
    }

    queue["ideas"].insert(0, new_idea)
    queue["last_updated"] = datetime.now().isoformat()

    QUEUE_FILE.write_text(json.dumps(queue, indent=2))
    print(f"Added {new_idea['id']}: {hypothesis}")


def run_layer1():
    """Run Layer 1 (Continuous Optimizer)."""
    script = BASE_DIR / "scripts" / "continuous_optimizer.py"

    if not script.exists():
        print(f"Error: {script} not found")
        sys.exit(1)

    print("\n=== Starting Layer 1: Continuous Optimizer ===\n")
    print("This will run experiments from the idea queue.")
    print("Press Ctrl+C to stop.\n")

    os.execv(sys.executable, [sys.executable, str(script), "--loop"])


def run_layer2():
    """Run Layer 2 (Meta Optimizer)."""
    script = BASE_DIR / "scripts" / "meta_optimizer.py"

    if not script.exists():
        print(f"Error: {script} not found")
        sys.exit(1)

    print("\n=== Starting Layer 2: Meta Optimizer ===\n")
    print("This will analyze experiments and suggest improvements every 30 minutes.")
    print("Press Ctrl+C to stop.\n")

    os.execv(sys.executable, [sys.executable, str(script), "--loop", "--interval", "1800"])


def run_both():
    """Run both layers as background processes."""
    global processes

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    script1 = BASE_DIR / "scripts" / "continuous_optimizer.py"
    script2 = BASE_DIR / "scripts" / "meta_optimizer.py"

    print("\n=== Starting Both Optimization Layers ===\n")

    # Start Layer 1
    print("Starting Layer 1 (Continuous Optimizer)...")
    proc1 = subprocess.Popen(
        [sys.executable, str(script1), "--loop"],
        cwd=str(BASE_DIR)
    )
    processes.append(proc1)

    time.sleep(2)

    # Start Layer 2
    print("Starting Layer 2 (Meta Optimizer)...")
    proc2 = subprocess.Popen(
        [sys.executable, str(script2), "--loop", "--interval", "1800"],
        cwd=str(BASE_DIR)
    )
    processes.append(proc2)

    print("\nBoth layers started.")
    print("  Layer 1: Runs experiments continuously")
    print("  Layer 2: Analyzes results every 30 minutes")
    print("\nPress Ctrl+C to stop both.\n")

    try:
        while True:
            for proc in processes:
                if proc.poll() is not None:
                    print(f"Warning: Process exited with code {proc.returncode}")
            time.sleep(10)
    except KeyboardInterrupt:
        pass


def main():
    parser = argparse.ArgumentParser(
        description="Experiment System - Two-Layer Optimization",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python experiments.py              # Run Layer 1 (continuous optimizer)
    python experiments.py --meta       # Run Layer 2 (meta optimizer)
    python experiments.py --both       # Run both layers
    python experiments.py --status     # Show status
    python experiments.py --queue      # Show idea queue
    python experiments.py --add "Test wider stops"  # Add experiment idea
        """
    )

    parser.add_argument("--meta", action="store_true", help="Run Layer 2 (Meta Optimizer)")
    parser.add_argument("--both", action="store_true", help="Run both layers")
    parser.add_argument("--status", action="store_true", help="Show experiment status")
    parser.add_argument("--queue", action="store_true", help="Show idea queue")
    parser.add_argument("--add", type=str, metavar="HYPOTHESIS", help="Add experiment idea")
    parser.add_argument("--env", type=str, help="Environment vars for --add (JSON format)")

    args = parser.parse_args()

    if args.status:
        show_status()
        return

    if args.queue:
        show_queue()
        return

    if args.add:
        env_vars = {}
        if args.env:
            try:
                env_vars = json.loads(args.env)
            except:
                print(f"Warning: Could not parse --env as JSON: {args.env}")
        add_idea(args.add, env_vars)
        return

    if args.both:
        run_both()
    elif args.meta:
        run_layer2()
    else:
        run_layer1()


if __name__ == "__main__":
    main()
