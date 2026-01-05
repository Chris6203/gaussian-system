#!/usr/bin/env python3
"""
Dashboard Remote Client
=======================

Run this on training machines to report status to the central dashboard.

Usage:
    python scripts/dashboard_remote_client.py

Environment Variables:
    DASHBOARD_URL     - Central dashboard URL (default: http://localhost:5003)
    MACHINE_ID        - Unique machine identifier (default: hostname)
    HEARTBEAT_SECONDS - Update interval (default: 30)

Example:
    # On training machine
    DASHBOARD_URL=http://192.168.1.100:5003 python scripts/dashboard_remote_client.py
"""

import os
import sys
import json
import time
import glob
import socket
import logging
from datetime import datetime
from pathlib import Path

import requests

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration
DASHBOARD_URL = os.environ.get('DASHBOARD_URL', 'http://localhost:5003')
MACHINE_ID = os.environ.get('MACHINE_ID', socket.gethostname())
HEARTBEAT_SECONDS = int(os.environ.get('HEARTBEAT_SECONDS', '30'))

# Log pattern to monitor
LOG_PATTERN = os.environ.get('LOG_PATTERN', 'logs/real_bot_simulation*.log')


def get_active_runs():
    """Get list of active training runs on this machine."""
    runs = []
    cutoff_time = time.time() - (2 * 60 * 60)  # Last 2 hours

    for log_path in glob.glob(LOG_PATTERN):
        try:
            path = Path(log_path)
            if not path.exists() or path.stat().st_size == 0:
                continue

            mtime = path.stat().st_mtime
            if mtime < cutoff_time:
                continue

            run_name = path.stem.replace('real_bot_simulation_', '')
            is_active = (time.time() - mtime) < 60  # Modified in last minute

            runs.append({
                'run_name': run_name,
                'log_file': str(path),
                'size_mb': round(path.stat().st_size / (1024 * 1024), 1),
                'last_modified': datetime.fromtimestamp(mtime).isoformat(),
                'is_active': is_active
            })
        except Exception as e:
            logger.warning(f"Error reading {log_path}: {e}")

    return runs


def parse_log_summary(log_path: str):
    """Parse basic stats from log file tail."""
    try:
        with open(log_path, 'r', encoding='utf-8', errors='ignore') as f:
            # Read last 100 lines
            lines = f.readlines()[-100:]

        stats = {
            'cycles': 0,
            'trades': 0,
            'pnl': 0.0,
            'win_rate': 0.0
        }

        for line in reversed(lines):
            if 'Completed cycles:' in line:
                try:
                    stats['cycles'] = int(line.split('Completed cycles:')[1].split()[0])
                except:
                    pass
            elif 'Total trades:' in line:
                try:
                    stats['trades'] = int(line.split('Total trades:')[1].split()[0])
                except:
                    pass
            elif 'Current P&L:' in line:
                try:
                    pnl_str = line.split('Current P&L:')[1].split()[0].replace('$', '').replace(',', '')
                    stats['pnl'] = float(pnl_str)
                except:
                    pass

        return stats
    except Exception as e:
        logger.warning(f"Error parsing {log_path}: {e}")
        return None


def register_machine():
    """Register this machine with the central dashboard."""
    try:
        resp = requests.post(
            f'{DASHBOARD_URL}/api/remote/register',
            json={
                'machine_id': MACHINE_ID,
                'hostname': socket.gethostname(),
                'ip': socket.gethostbyname(socket.gethostname())
            },
            timeout=10
        )

        if resp.status_code == 200:
            data = resp.json()
            if data.get('success'):
                logger.info(f"Registered with dashboard as '{MACHINE_ID}'")
                return True

        logger.error(f"Failed to register: {resp.text}")
        return False
    except Exception as e:
        logger.error(f"Error registering: {e}")
        return False


def send_heartbeat():
    """Send heartbeat with current runs to dashboard."""
    runs = get_active_runs()

    # Get training state from active runs
    training_state = None
    for run in runs:
        if run['is_active']:
            stats = parse_log_summary(run['log_file'])
            if stats:
                training_state = stats
                break

    try:
        resp = requests.post(
            f'{DASHBOARD_URL}/api/remote/heartbeat',
            json={
                'machine_id': MACHINE_ID,
                'hostname': socket.gethostname(),
                'runs': runs,
                'training_state': training_state
            },
            timeout=10
        )

        if resp.status_code == 200:
            active_count = sum(1 for r in runs if r['is_active'])
            logger.debug(f"Heartbeat sent: {len(runs)} runs ({active_count} active)")
            return True

        logger.warning(f"Heartbeat failed: {resp.text}")
        return False
    except Exception as e:
        logger.error(f"Error sending heartbeat: {e}")
        return False


def submit_result(run_id: str, result: dict):
    """Submit completed experiment result."""
    try:
        resp = requests.post(
            f'{DASHBOARD_URL}/api/remote/submit_result',
            json={
                'machine_id': MACHINE_ID,
                'run_id': run_id,
                'result': result
            },
            timeout=10
        )

        if resp.status_code == 200:
            logger.info(f"Submitted result for {run_id}")
            return True

        logger.error(f"Failed to submit result: {resp.text}")
        return False
    except Exception as e:
        logger.error(f"Error submitting result: {e}")
        return False


def main():
    """Main loop."""
    logger.info(f"Dashboard Remote Client starting...")
    logger.info(f"  Dashboard URL: {DASHBOARD_URL}")
    logger.info(f"  Machine ID: {MACHINE_ID}")
    logger.info(f"  Heartbeat: every {HEARTBEAT_SECONDS}s")
    logger.info(f"  Log pattern: {LOG_PATTERN}")

    # Register on startup
    if not register_machine():
        logger.warning("Failed to register, will retry...")

    # Main loop
    while True:
        try:
            send_heartbeat()
        except KeyboardInterrupt:
            logger.info("Shutting down...")
            break
        except Exception as e:
            logger.error(f"Error in main loop: {e}")

        time.sleep(HEARTBEAT_SECONDS)


if __name__ == '__main__':
    main()
