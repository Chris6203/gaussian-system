#!/usr/bin/env python3
"""
Unified Dashboard Server
========================
Serves all trading dashboards from a single entry point.

Usage:
    python dashboard.py                    # Start all dashboards (live on 5000, training on 5001)
    python dashboard.py --live             # Live trading dashboard only (port 5000)
    python dashboard.py --training         # Training dashboard only (port 5001)
    python dashboard.py --history          # History dashboard only (port 5002)
    python dashboard.py --port 8080        # Custom port for live dashboard
    python dashboard.py --status           # Show dashboard status

Configuration:
    Edit config.json for dashboard settings
    Edit server_config.json for server addresses
"""

import sys
import os
import argparse
import subprocess
import signal
import time
import json
from pathlib import Path
from typing import List, Optional

# Add project root to path
BASE_DIR = Path(__file__).parent.resolve()
sys.path.insert(0, str(BASE_DIR))

# Dashboard configurations
DASHBOARDS = {
    'unified': {
        'script': 'core/dashboards/unified_dashboard_server.py',
        'port': 5001,
        'description': 'Unified dashboard (hub, live, training, history)'
    },
    'live': {
        'script': 'core/dashboards/dashboard_server.py',
        'port': 5000,
        'description': 'Live trading dashboard (legacy)'
    },
    'training': {
        'script': 'core/dashboards/training_dashboard_server.py',
        'port': 5001,
        'description': 'Training/simulation dashboard (legacy)'
    },
    'history': {
        'script': 'core/dashboards/history_dashboard_server.py',
        'port': 5002,
        'description': 'Historical model browser (legacy)'
    }
}

# Track running processes
processes: List[subprocess.Popen] = []


def load_server_config():
    """Load server configuration."""
    config_path = BASE_DIR / "server_config.json"
    if config_path.exists():
        try:
            return json.loads(config_path.read_text())
        except:
            pass
    return {}


def start_dashboard(name: str, port: Optional[int] = None) -> subprocess.Popen:
    """Start a dashboard server."""
    config = DASHBOARDS[name]
    script_path = BASE_DIR / config['script']

    if not script_path.exists():
        print(f"Error: {script_path} not found")
        return None

    actual_port = port or config['port']

    env = os.environ.copy()
    env['DASHBOARD_PORT'] = str(actual_port)
    # Add project root to Python path for imports
    env['PYTHONPATH'] = str(BASE_DIR) + os.pathsep + env.get('PYTHONPATH', '')

    print(f"Starting {config['description']} on port {actual_port}...")

    proc = subprocess.Popen(
        [sys.executable, str(script_path)],
        env=env,
        cwd=str(BASE_DIR)
    )

    return proc


def stop_all():
    """Stop all running dashboard processes."""
    global processes
    for proc in processes:
        if proc and proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
    processes = []


def signal_handler(sig, frame):
    """Handle shutdown signals."""
    print("\nShutting down dashboards...")
    stop_all()
    sys.exit(0)


def show_status():
    """Show dashboard status."""
    import socket

    server_config = load_server_config()
    main_ip = server_config.get('main_server', {}).get('ip', 'localhost')

    print("\n=== Dashboard Status ===\n")
    print(f"Main Server: {main_ip}")
    print()

    for name, config in DASHBOARDS.items():
        port = config['port']
        status = "Unknown"

        # Try to connect
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(1)
        try:
            sock.connect(('localhost', port))
            status = "RUNNING"
        except:
            status = "NOT RUNNING"
        finally:
            sock.close()

        print(f"  {name.upper():12} Port {port}: {status}")
        print(f"                URL: http://{main_ip}:{port}")

    print()


def run_single_dashboard(name: str, port: Optional[int] = None):
    """Run a single dashboard in the foreground."""
    config = DASHBOARDS[name]
    script_path = BASE_DIR / config['script']

    if not script_path.exists():
        print(f"Error: {script_path} not found")
        sys.exit(1)

    actual_port = port or config['port']

    print(f"\n=== {config['description'].upper()} ===")
    print(f"Port: {actual_port}")

    server_config = load_server_config()
    main_ip = server_config.get('main_server', {}).get('ip', 'localhost')
    print(f"URL: http://{main_ip}:{actual_port}")
    print()

    # Set environment and run as subprocess (foreground)
    env = os.environ.copy()
    env['DASHBOARD_PORT'] = str(actual_port)
    # Add project root to Python path for imports
    env['PYTHONPATH'] = str(BASE_DIR) + os.pathsep + env.get('PYTHONPATH', '')

    try:
        proc = subprocess.run(
            [sys.executable, str(script_path)],
            env=env,
            cwd=str(BASE_DIR)
        )
        sys.exit(proc.returncode)
    except KeyboardInterrupt:
        print("\nDashboard stopped.")
        sys.exit(0)


def run_all_dashboards():
    """Run all legacy dashboards as background processes (not unified)."""
    global processes

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    server_config = load_server_config()
    main_ip = server_config.get('main_server', {}).get('ip', 'localhost')

    print("\n=== Starting Legacy Dashboards ===\n")
    print("Note: Use 'python dashboard.py' (no args) for the unified dashboard.\n")

    legacy_dashboards = {k: v for k, v in DASHBOARDS.items() if k != 'unified'}
    for name, config in legacy_dashboards.items():
        proc = start_dashboard(name)
        if proc:
            processes.append(proc)
            time.sleep(1)  # Give each server time to start

    print("\nLegacy dashboards started:")
    for name, config in legacy_dashboards.items():
        print(f"  {config['description']:30} http://{main_ip}:{config['port']}")

    print("\nPress Ctrl+C to stop all dashboards\n")

    # Wait for processes
    try:
        while True:
            # Check if any process died
            for proc in processes:
                if proc.poll() is not None:
                    print(f"Warning: A dashboard process exited with code {proc.returncode}")
            time.sleep(5)
    except KeyboardInterrupt:
        pass
    finally:
        stop_all()


def main():
    parser = argparse.ArgumentParser(
        description="Unified Dashboard Server",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python dashboard.py                # Start all dashboards
    python dashboard.py --live         # Live dashboard only
    python dashboard.py --training     # Training dashboard only
    python dashboard.py --status       # Check status
        """
    )

    parser.add_argument("--unified", action="store_true", help="Run unified dashboard (default)")
    parser.add_argument("--live", action="store_true", help="Run live trading dashboard (legacy)")
    parser.add_argument("--training", action="store_true", help="Run training dashboard (legacy)")
    parser.add_argument("--history", action="store_true", help="Run history dashboard (legacy)")
    parser.add_argument("--all", action="store_true", help="Run all legacy dashboards")
    parser.add_argument("--port", type=int, help="Custom port (for single dashboard mode)")
    parser.add_argument("--status", action="store_true", help="Show dashboard status")

    args = parser.parse_args()

    if args.status:
        show_status()
        return

    # Determine which dashboard(s) to run
    selected = []
    if args.unified:
        selected.append('unified')
    if args.live:
        selected.append('live')
    if args.training:
        selected.append('training')
    if args.history:
        selected.append('history')

    if len(selected) == 0:
        # Default: run unified dashboard
        run_single_dashboard('unified', args.port)
        return

    if args.all:
        # Run all legacy dashboards
        run_all_dashboards()
    elif len(selected) == 1:
        # Run single dashboard in foreground
        run_single_dashboard(selected[0], args.port)
    else:
        # Run multiple specific dashboards
        global processes
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

        for name in selected:
            proc = start_dashboard(name)
            if proc:
                processes.append(proc)
                time.sleep(1)

        print("\nDashboards started. Press Ctrl+C to stop.\n")

        try:
            while True:
                time.sleep(5)
        except KeyboardInterrupt:
            pass
        finally:
            stop_all()


if __name__ == "__main__":
    main()
