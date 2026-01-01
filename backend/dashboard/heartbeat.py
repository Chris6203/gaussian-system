#!/usr/bin/env python3
"""
Heartbeat System for Running Experiments

Provides functions to register, update, and unregister running experiments
so the dashboard can track active training runs in real-time.

Usage in train_time_travel.py:
    from backend.dashboard.heartbeat import ExperimentHeartbeat

    heartbeat = ExperimentHeartbeat(run_name, target_cycles=5000)
    heartbeat.start()

    for cycle in range(max_cycles):
        # ... training code ...
        if cycle % 100 == 0:
            heartbeat.update(cycle, current_balance - initial_balance)

    heartbeat.stop()
"""

import os
import json
import sqlite3
import socket
import atexit
import signal
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any


class ExperimentHeartbeat:
    """Manages heartbeat for a running experiment."""

    def __init__(
        self,
        run_name: str,
        target_cycles: int = 0,
        env_vars: Optional[Dict[str, str]] = None,
        experiments_db: str = 'data/experiments.db',
        heartbeat_file: str = 'data/running_experiment.json'
    ):
        self.run_name = run_name
        self.target_cycles = target_cycles
        self.env_vars = env_vars or {}
        self.experiments_db = experiments_db
        self.heartbeat_file = Path(heartbeat_file)
        self.started_at = None
        self.pid = os.getpid()
        self.machine_id = socket.gethostname()
        self._registered = False

    def start(self):
        """Register the experiment as running."""
        self.started_at = datetime.now().isoformat()

        # Write to heartbeat file (for quick dashboard access)
        self._write_heartbeat_file(0, 0)

        # Write to database (for persistent tracking)
        self._register_in_db()

        self._registered = True

        # Register cleanup handlers
        atexit.register(self.stop)
        signal.signal(signal.SIGTERM, self._signal_handler)
        signal.signal(signal.SIGINT, self._signal_handler)

    def update(self, current_cycle: int, current_pnl: float = 0):
        """Update heartbeat with current progress."""
        if not self._registered:
            return

        # Update file
        self._write_heartbeat_file(current_cycle, current_pnl)

        # Update database
        self._update_in_db(current_cycle, current_pnl)

    def stop(self):
        """Unregister the experiment (called on completion or exit)."""
        if not self._registered:
            return

        # Remove heartbeat file
        try:
            if self.heartbeat_file.exists():
                self.heartbeat_file.unlink()
        except Exception:
            pass

        # Remove from database
        self._unregister_from_db()

        self._registered = False

    def _write_heartbeat_file(self, cycle: int, pnl: float):
        """Write heartbeat to JSON file."""
        try:
            self.heartbeat_file.parent.mkdir(parents=True, exist_ok=True)
            data = {
                'run_name': self.run_name,
                'pid': self.pid,
                'machine_id': self.machine_id,
                'started_at': self.started_at,
                'last_heartbeat': datetime.now().isoformat(),
                'current_cycle': cycle,
                'current_pnl': pnl,
                'target_cycles': self.target_cycles
            }
            with open(self.heartbeat_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception:
            pass

    def _register_in_db(self):
        """Register experiment in running_experiments table."""
        try:
            conn = sqlite3.connect(self.experiments_db)
            cursor = conn.cursor()

            # Check if table exists
            cursor.execute("""
                SELECT name FROM sqlite_master
                WHERE type='table' AND name='running_experiments'
            """)
            if not cursor.fetchone():
                # Create table if it doesn't exist
                cursor.execute("""
                    CREATE TABLE running_experiments (
                        run_name TEXT PRIMARY KEY,
                        started_at TEXT NOT NULL,
                        pid INTEGER,
                        machine_id TEXT,
                        last_heartbeat TEXT,
                        current_cycle INTEGER DEFAULT 0,
                        current_pnl REAL DEFAULT 0,
                        target_cycles INTEGER,
                        env_vars TEXT
                    )
                """)

            cursor.execute("""
                INSERT OR REPLACE INTO running_experiments
                (run_name, started_at, pid, machine_id, last_heartbeat,
                 current_cycle, current_pnl, target_cycles, env_vars)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                self.run_name,
                self.started_at,
                self.pid,
                self.machine_id,
                datetime.now().isoformat(),
                0,
                0,
                self.target_cycles,
                json.dumps(self.env_vars) if self.env_vars else None
            ))

            conn.commit()
            conn.close()
        except Exception:
            pass

    def _update_in_db(self, cycle: int, pnl: float):
        """Update heartbeat in database."""
        try:
            conn = sqlite3.connect(self.experiments_db)
            cursor = conn.cursor()

            cursor.execute("""
                UPDATE running_experiments
                SET last_heartbeat = ?,
                    current_cycle = ?,
                    current_pnl = ?
                WHERE run_name = ?
            """, (
                datetime.now().isoformat(),
                cycle,
                pnl,
                self.run_name
            ))

            conn.commit()
            conn.close()
        except Exception:
            pass

    def _unregister_from_db(self):
        """Remove experiment from running_experiments table."""
        try:
            conn = sqlite3.connect(self.experiments_db)
            cursor = conn.cursor()

            cursor.execute(
                "DELETE FROM running_experiments WHERE run_name = ?",
                (self.run_name,)
            )

            conn.commit()
            conn.close()
        except Exception:
            pass

    def _signal_handler(self, signum, frame):
        """Handle SIGTERM/SIGINT gracefully."""
        self.stop()
        # Re-raise to allow normal shutdown
        signal.signal(signum, signal.SIG_DFL)
        os.kill(os.getpid(), signum)


def get_running_experiments(experiments_db: str = 'data/experiments.db') -> list:
    """Get list of currently running experiments."""
    try:
        conn = sqlite3.connect(experiments_db)
        conn.row_factory = lambda c, r: {col[0]: r[idx] for idx, col in enumerate(c.description)}
        cursor = conn.cursor()

        cursor.execute("""
            SELECT * FROM running_experiments
            WHERE last_heartbeat > datetime('now', '-5 minutes')
        """)
        results = cursor.fetchall()
        conn.close()
        return results
    except Exception:
        return []


def cleanup_stale_experiments(experiments_db: str = 'data/experiments.db', max_age_minutes: int = 10):
    """Remove experiments that haven't updated their heartbeat."""
    try:
        conn = sqlite3.connect(experiments_db)
        cursor = conn.cursor()

        cursor.execute(f"""
            DELETE FROM running_experiments
            WHERE last_heartbeat < datetime('now', '-{max_age_minutes} minutes')
        """)

        deleted = cursor.rowcount
        conn.commit()
        conn.close()
        return deleted
    except Exception:
        return 0
