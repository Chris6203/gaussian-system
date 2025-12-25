#!/usr/bin/env python3
"""
Compare Entry Controllers via Time-Travel
========================================

Runs `output3/scripts/train_time_travel.py` twice on the same historical window:
- baseline ("old"): whatever the script normally does (bandit/RL stack)
- q_scorer ("new"): ENTRY_CONTROLLER=q_scorer with configurable gates

Outputs:
- <out_root>/baseline/SUMMARY.txt
- <out_root>/q_scorer/SUMMARY.txt
- <out_root>/COMPARE.json  (parsed summary metrics + delta)

Safe-by-default:
- Forces PAPER_TRADING via `.trading_mode_config.json` during subprocess runs.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _output3_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


class _TradingModeGuard:
    def __init__(self, repo_root: Path, *, enabled: bool, mode: str = "PAPER_TRADING"):
        self.repo_root = repo_root
        self.enabled = bool(enabled)
        self.mode = str(mode or "PAPER_TRADING").strip().upper()
        if self.mode not in ("PAPER_TRADING", "SIMULATION"):
            self.mode = "PAPER_TRADING"
        self.path = repo_root / ".trading_mode_config.json"
        self._backup: Optional[str] = None
        self._had_file = False

    def __enter__(self):
        if not self.enabled:
            return self
        try:
            if self.path.exists():
                self._had_file = True
                self._backup = self.path.read_text(encoding="utf-8")
        except Exception:
            self._had_file = False
            self._backup = None
        cfg = {
            "mode": self.mode,
            "description": f"Forced {self.mode} for time-travel compare",
            "live_trading": {"enabled": False, "tradier_sandbox": True},
        }
        _write_text(self.path, json.dumps(cfg, indent=2))
        return self

    def __exit__(self, exc_type, exc, tb):
        if not self.enabled:
            return False
        try:
            if self._had_file and self._backup is not None:
                _write_text(self.path, self._backup)
            else:
                try:
                    self.path.unlink(missing_ok=True)
                except TypeError:
                    if self.path.exists():
                        self.path.unlink()
        except Exception:
            pass
        return False


@dataclass
class SummaryMetrics:
    final_balance: float = 0.0
    pnl: float = 0.0
    total_trades: int = 0
    total_cycles: int = 0
    win_rate: float = 0.0


def _read_summary(summary_path: Path) -> SummaryMetrics:
    m = SummaryMetrics()
    if not summary_path.exists():
        return m
    try:
        lines = summary_path.read_text(encoding="utf-8", errors="ignore").splitlines()
    except Exception:
        return m
    for ln in lines:
        s = ln.strip()
        if s.startswith("Final Balance:"):
            try:
                val = s.split(":", 1)[1].strip().replace("$", "").replace(",", "")
                m.final_balance = float(val)
            except Exception:
                pass
        elif s.startswith("P&L:"):
            try:
                rhs = s.split(":", 1)[1].strip()
                first = rhs.split(" ", 1)[0].replace("$", "").replace(",", "")
                m.pnl = float(first)
            except Exception:
                pass
        elif s.startswith("Total Trades:"):
            try:
                m.total_trades = int(s.split(":", 1)[1].strip())
            except Exception:
                pass
        elif s.startswith("Total Cycles:"):
            try:
                m.total_cycles = int(s.split(":", 1)[1].strip())
            except Exception:
                pass
        elif s.startswith("Win Rate:"):
            try:
                rhs = s.split(":", 1)[1].strip().replace("%", "")
                m.win_rate = float(rhs) / 100.0
            except Exception:
                pass
    return m


def _run_time_travel(py: str, script: Path, *, cwd: Path, env: Dict[str, str], timeout_sec: int) -> None:
    cmd = [py, str(script)]
    print("\n$ " + " ".join(cmd))
    subprocess.run(cmd, cwd=str(cwd), env=env, check=True, timeout=int(timeout_sec))


def main() -> int:
    ap = argparse.ArgumentParser()
    repo_root = _repo_root()
    out3 = _output3_root()
    py = sys.executable

    ap.add_argument("--out-root", type=str, default="", help="Folder to write both runs into.")
    ap.add_argument("--cycles", type=int, default=800)
    ap.add_argument("--max-positions", type=int, default=1)
    ap.add_argument("--timeout-sec", type=int, default=3600)
    ap.add_argument("--force-mode", type=str, default=os.environ.get("PIPELINE_FORCE_MODE", "PAPER_TRADING"))
    ap.add_argument("--no-force-mode", action="store_true")

    # same historical slice start (train_time_travel supports this)
    ap.add_argument("--start-date", type=str, default=os.environ.get("TRAINING_START_DATE", ""))

    # q_scorer knobs
    ap.add_argument("--q-threshold", type=float, default=10.0)
    ap.add_argument("--q-min-conf", type=float, default=0.25)
    ap.add_argument("--q-min-abs-ret", type=float, default=0.0008)

    # model artifacts to use for q_scorer
    ap.add_argument("--model-path", type=str, default=str(out3 / "models" / "q_scorer.pt"))
    ap.add_argument("--meta-path", type=str, default=str(out3 / "models" / "q_scorer_metadata.json"))

    args = ap.parse_args()

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_root = Path(str(args.out_root).strip() or (repo_root / "output3" / "models" / f"compare_entry_{ts}"))
    base_dir = out_root / "baseline"
    q_dir = out_root / "q_scorer"
    base_dir.mkdir(parents=True, exist_ok=True)
    q_dir.mkdir(parents=True, exist_ok=True)

    env_common = dict(os.environ)
    env_common["ENABLE_LIVE_MODE"] = "false"
    env_common["USE_SHADOW_TRADING"] = "0"
    env_common["TT_MAX_CYCLES"] = str(int(args.cycles))
    env_common["TT_TRAIN_MAX_POSITIONS"] = str(int(args.max_positions))
    if str(args.start_date).strip():
        env_common["TRAINING_START_DATE"] = str(args.start_date).strip()

    force_enabled = not bool(args.no_force_mode)
    guard_mode = str(args.force_mode or "PAPER_TRADING").strip().upper()

    script = out3 / "scripts" / "train_time_travel.py"

    with _TradingModeGuard(repo_root, enabled=bool(force_enabled), mode=guard_mode):
        # Baseline
        env_b = dict(env_common)
        env_b["MODEL_RUN_DIR"] = str(base_dir)
        # Ensure we're not accidentally using q mode
        env_b.pop("ENTRY_CONTROLLER", None)
        _run_time_travel(py, script, cwd=repo_root, env=env_b, timeout_sec=int(args.timeout_sec))

        # Q-scorer
        env_q = dict(env_common)
        env_q["MODEL_RUN_DIR"] = str(q_dir)
        env_q["ENTRY_CONTROLLER"] = "q_scorer"
        env_q["ENTRY_Q_THRESHOLD"] = str(float(args.q_threshold))
        env_q["Q_ENTRY_MIN_CONF"] = str(float(args.q_min_conf))
        env_q["Q_ENTRY_MIN_ABS_RET"] = str(float(args.q_min_abs_ret))
        env_q["Q_SCORER_MODEL_PATH"] = str(args.model_path)
        env_q["Q_SCORER_METADATA_PATH"] = str(args.meta_path)
        _run_time_travel(py, script, cwd=repo_root, env=env_q, timeout_sec=int(args.timeout_sec))

    base_sum = _read_summary(base_dir / "SUMMARY.txt")
    q_sum = _read_summary(q_dir / "SUMMARY.txt")

    out = {
        "baseline": base_sum.__dict__,
        "q_scorer": q_sum.__dict__,
        "delta": {
            "pnl": float(q_sum.pnl - base_sum.pnl),
            "final_balance": float(q_sum.final_balance - base_sum.final_balance),
            "total_trades": int(q_sum.total_trades - base_sum.total_trades),
            "win_rate": float(q_sum.win_rate - base_sum.win_rate),
        },
        "paths": {
            "out_root": str(out_root),
            "baseline_summary": str(base_dir / "SUMMARY.txt"),
            "q_scorer_summary": str(q_dir / "SUMMARY.txt"),
        },
        "params": {
            "cycles": int(args.cycles),
            "max_positions": int(args.max_positions),
            "start_date": str(args.start_date).strip(),
            "q_threshold": float(args.q_threshold),
            "q_min_conf": float(args.q_min_conf),
            "q_min_abs_ret": float(args.q_min_abs_ret),
        },
    }
    _write_text(out_root / "COMPARE.json", json.dumps(out, indent=2))

    print("\nDONE")
    print(f"Out root: {out_root}")
    print(f"Baseline SUMMARY: {base_dir / 'SUMMARY.txt'}")
    print(f"Q-scorer SUMMARY: {q_dir / 'SUMMARY.txt'}")
    print(f"COMPARE: {out_root / 'COMPARE.json'}")
    print(f"\nDelta pnl: {out['delta']['pnl']:+.2f} (q - baseline)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())





