#!/usr/bin/env python3
"""
Experiment Queue System with Offline Support

Queues experiments locally when the main server (192.168.20.235) is unavailable,
and syncs them when the server comes back online.

Usage:
    # Add experiment to queue
    python tools/experiment_queue.py add --name "IDEA-266 20K Validation" --priority critical --config '{"TT_MAX_CYCLES": "20000"}'

    # Check server and sync
    python tools/experiment_queue.py sync

    # List queued experiments
    python tools/experiment_queue.py list

    # Run next queued experiment locally
    python tools/experiment_queue.py run-local
"""

import os
import sys
import json
import socket
import subprocess
import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

# Configuration
QUEUE_FILE = Path(__file__).parent.parent / ".claude" / "collab" / "offline_queue.json"
SERVER_IP = "192.168.20.235"
SERVER_PORT = 5000  # Dashboard port
IDEA_QUEUE_FILE = Path(__file__).parent.parent / ".claude" / "collab" / "idea_queue.json"


def check_server_available(ip: str = SERVER_IP, port: int = SERVER_PORT, timeout: float = 2.0) -> bool:
    """Check if the server is reachable."""
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(timeout)
        result = sock.connect_ex((ip, port))
        sock.close()
        return result == 0
    except Exception:
        return False


def load_queue() -> Dict:
    """Load the offline queue."""
    if QUEUE_FILE.exists():
        with open(QUEUE_FILE) as f:
            return json.load(f)
    return {"experiments": [], "last_sync": None}


def save_queue(queue: Dict):
    """Save the offline queue."""
    QUEUE_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(QUEUE_FILE, 'w') as f:
        json.dump(queue, f, indent=2)


def add_experiment(
    name: str,
    config: Dict[str, str],
    priority: str = "normal",
    cycles: int = 5000,
    description: str = ""
) -> Dict:
    """Add an experiment to the queue."""
    queue = load_queue()

    experiment = {
        "id": f"OFFLINE-{len(queue['experiments']) + 1:04d}",
        "name": name,
        "description": description,
        "config": config,
        "priority": priority,  # critical, high, normal, low
        "cycles": cycles,
        "status": "queued",
        "queued_at": datetime.now().isoformat(),
        "started_at": None,
        "completed_at": None,
        "result": None
    }

    # Insert based on priority
    priority_order = {"critical": 0, "high": 1, "normal": 2, "low": 3}
    exp_priority = priority_order.get(priority, 2)

    insert_idx = len(queue['experiments'])
    for i, exp in enumerate(queue['experiments']):
        if priority_order.get(exp['priority'], 2) > exp_priority:
            insert_idx = i
            break

    queue['experiments'].insert(insert_idx, experiment)
    save_queue(queue)

    print(f"✓ Added experiment: {experiment['id']} - {name}")
    print(f"  Priority: {priority}")
    print(f"  Position: {insert_idx + 1} of {len(queue['experiments'])}")

    return experiment


def list_experiments():
    """List all queued experiments."""
    queue = load_queue()

    if not queue['experiments']:
        print("No experiments in queue.")
        return

    print(f"\n{'='*80}")
    print(f"EXPERIMENT QUEUE ({len(queue['experiments'])} experiments)")
    print(f"{'='*80}")
    print(f"{'ID':<15} {'Priority':<10} {'Status':<12} {'Name':<40}")
    print(f"{'-'*80}")

    for exp in queue['experiments']:
        print(f"{exp['id']:<15} {exp['priority']:<10} {exp['status']:<12} {exp['name'][:40]:<40}")

    print(f"\nServer status: {'ONLINE' if check_server_available() else 'OFFLINE'}")
    if queue.get('last_sync'):
        print(f"Last sync: {queue['last_sync']}")


def run_local_experiment(experiment: Dict) -> Dict:
    """Run an experiment locally."""
    print(f"\n🚀 Starting experiment: {experiment['name']}")

    # Build environment
    env = os.environ.copy()
    env['MODEL_RUN_DIR'] = f"models/{experiment['id']}"
    env['TT_MAX_CYCLES'] = str(experiment.get('cycles', 5000))
    env['TT_PRINT_EVERY'] = '500'
    env['PAPER_TRADING'] = 'True'

    # Add experiment config
    for k, v in experiment.get('config', {}).items():
        env[k] = str(v)

    # Run experiment
    experiment['started_at'] = datetime.now().isoformat()
    experiment['status'] = 'running'

    try:
        result = subprocess.run(
            ['python', 'scripts/train_time_travel.py'],
            env=env,
            capture_output=True,
            text=True,
            timeout=7200  # 2 hour timeout
        )

        experiment['completed_at'] = datetime.now().isoformat()

        if result.returncode == 0:
            experiment['status'] = 'completed'
            # Try to read results
            summary_path = Path(f"models/{experiment['id']}/SUMMARY.txt")
            if summary_path.exists():
                experiment['result'] = summary_path.read_text()
            print(f"✓ Experiment completed: {experiment['id']}")
        else:
            experiment['status'] = 'failed'
            experiment['error'] = result.stderr[-500:] if result.stderr else "Unknown error"
            print(f"✗ Experiment failed: {experiment['id']}")

    except subprocess.TimeoutExpired:
        experiment['status'] = 'timeout'
        experiment['error'] = "Experiment timed out after 2 hours"
        print(f"⏱ Experiment timed out: {experiment['id']}")
    except Exception as e:
        experiment['status'] = 'error'
        experiment['error'] = str(e)
        print(f"✗ Experiment error: {e}")

    return experiment


def run_next():
    """Run the next queued experiment locally."""
    queue = load_queue()

    # Find next queued experiment
    for exp in queue['experiments']:
        if exp['status'] == 'queued':
            exp = run_local_experiment(exp)
            save_queue(queue)
            return exp

    print("No queued experiments to run.")
    return None


def sync_to_server():
    """Sync completed experiments to the main server."""
    if not check_server_available():
        print("✗ Server offline - cannot sync")
        return False

    queue = load_queue()

    # Sync completed experiments to idea_queue.json
    if IDEA_QUEUE_FILE.exists():
        with open(IDEA_QUEUE_FILE) as f:
            idea_queue = json.load(f)
    else:
        idea_queue = {"ideas": []}

    synced = 0
    for exp in queue['experiments']:
        if exp['status'] == 'completed' and not exp.get('synced'):
            # Add to idea queue as completed
            idea = {
                "id": exp['id'],
                "source": "offline_queue",
                "timestamp": exp['queued_at'],
                "category": "validation",
                "title": exp['name'],
                "description": exp.get('description', ''),
                "status": "completed",
                "env_vars": exp.get('config', {}),
                "last_result": {
                    "exp_id": exp['id'],
                    "completed_at": exp['completed_at']
                }
            }
            idea_queue['ideas'].append(idea)
            exp['synced'] = True
            synced += 1

    if synced > 0:
        with open(IDEA_QUEUE_FILE, 'w') as f:
            json.dump(idea_queue, f, indent=2)
        queue['last_sync'] = datetime.now().isoformat()
        save_queue(queue)
        print(f"✓ Synced {synced} experiments to server")
    else:
        print("No new experiments to sync")

    return True


def main():
    parser = argparse.ArgumentParser(description="Experiment Queue System")
    subparsers = parser.add_subparsers(dest='command', help='Commands')

    # Add command
    add_parser = subparsers.add_parser('add', help='Add experiment to queue')
    add_parser.add_argument('--name', required=True, help='Experiment name')
    add_parser.add_argument('--config', type=json.loads, default={}, help='Config as JSON')
    add_parser.add_argument('--priority', choices=['critical', 'high', 'normal', 'low'], default='normal')
    add_parser.add_argument('--cycles', type=int, default=5000)
    add_parser.add_argument('--description', default='')

    # List command
    subparsers.add_parser('list', help='List queued experiments')

    # Run command
    subparsers.add_parser('run-local', help='Run next queued experiment locally')

    # Sync command
    subparsers.add_parser('sync', help='Sync completed experiments to server')

    # Status command
    subparsers.add_parser('status', help='Check server status')

    args = parser.parse_args()

    if args.command == 'add':
        add_experiment(
            name=args.name,
            config=args.config,
            priority=args.priority,
            cycles=args.cycles,
            description=args.description
        )
    elif args.command == 'list':
        list_experiments()
    elif args.command == 'run-local':
        run_next()
    elif args.command == 'sync':
        sync_to_server()
    elif args.command == 'status':
        status = "ONLINE" if check_server_available() else "OFFLINE"
        print(f"Server {SERVER_IP}: {status}")
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
