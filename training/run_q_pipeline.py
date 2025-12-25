#!/usr/bin/env python3
"""
One-command Offline Q Pipeline
==============================

Runs (optionally) in three stages:
1) Generate dataset via time-travel simulation (DecisionRecord + MissedOpportunityRecord JSONL)
2) Train the offline Q regressor (q_scorer.pt + q_scorer_metadata.json)
3) Evaluate the trained Q-scorer on a log slice

This script is designed to be the single entry-point you run repeatedly while
changing knobs (horizon, label mode, thresholds, etc.) without re-wiring commands.

Safety:
- By default, forces SIMULATION trading mode during dataset generation and restores the
  previous `.trading_mode_config.json` afterwards.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional


def _repo_root() -> Path:
    # .../output3/training/run_q_pipeline.py -> parents[2] == repo root
    return Path(__file__).resolve().parents[2]


def _output3_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


class _TradingModeGuard:
    """
    Temporarily force `.trading_mode_config.json` to SIMULATION for a subprocess run.
    Restores the original file contents afterwards.
    """

    def __init__(self, repo_root: Path, *, enabled: bool):
        self.repo_root = repo_root
        self.enabled = bool(enabled)
        self.path = repo_root / ".trading_mode_config.json"
        self._backup: Optional[str] = None
        self._had_file: bool = False

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
            "mode": "SIMULATION",
            "description": "Forced SIMULATION for offline dataset generation",
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
                # If there was no file originally, remove the temporary file.
                try:
                    self.path.unlink(missing_ok=True)  # py3.8+ on Windows supports missing_ok
                except TypeError:
                    # older semantics: best-effort
                    if self.path.exists():
                        self.path.unlink()
        except Exception:
            # Don't mask the original exception.
            pass
        return False


def _run(cmd: list[str], *, cwd: Path, env: Dict[str, str]) -> None:
    print("\n$ " + " ".join(cmd))
    subprocess.run(cmd, cwd=str(cwd), env=env, check=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    repo_root = _repo_root()
    out3 = _output3_root()

    ap.add_argument("--run-dir", type=str, default="", help="MODEL_RUN_DIR for time-travel outputs")
    ap.add_argument("--horizon", type=int, default=int(os.environ.get("Q_HORIZON_MINUTES", "30")))
    ap.add_argument(
        "--steps",
        type=str,
        default="dataset,train,eval",
        help="Comma list: dataset,train,eval,deploy,bot",
    )

    # Dataset knobs
    ap.add_argument("--q-labels", type=str, default=os.environ.get("TT_Q_LABELS", "flat_only"), help="hold_only|all|flat_only")
    ap.add_argument("--tt-max-cycles", type=int, default=int(os.environ.get("TT_MAX_CYCLES", "0")))
    ap.add_argument("--tt-train-min-conf", type=float, default=float(os.environ.get("TT_TRAIN_MIN_CONF", "0.15")))
    ap.add_argument("--tt-train-min-abs-ret", type=float, default=float(os.environ.get("TT_TRAIN_MIN_ABS_RET", "0.0005")))
    ap.add_argument("--tt-train-max-positions", type=int, default=int(os.environ.get("TT_TRAIN_MAX_POSITIONS", "10")))
    ap.add_argument(
        "--force-mode",
        type=str,
        default=os.environ.get("PIPELINE_FORCE_MODE", "PAPER_TRADING"),
        help="Force trading mode during dataset/bot steps: PAPER_TRADING (no Tradier calls) or SIMULATION.",
    )
    ap.add_argument("--no-force-mode", action="store_true", help="Do not change .trading_mode_config.json")

    # Q label model knobs
    ap.add_argument("--ghost-delta", type=float, default=float(os.environ.get("Q_GHOST_DELTA", "0.5")))
    ap.add_argument("--pos-weight", type=float, default=float(os.environ.get("Q_POS_WEIGHT", "20")))
    ap.add_argument("--label-clip", type=float, default=float(os.environ.get("Q_LABEL_CLIP", "200")))
    ap.add_argument("--min-rows", type=int, default=int(os.environ.get("Q_MIN_MATCHED_ROWS", "100")))
    ap.add_argument(
        "--state-filter",
        type=str,
        default=os.environ.get("Q_STATE_FILTER", "flat"),
        help="Train filter: all | flat | in_trade. Default flat (entry-focused).",
    )

    # Eval knobs
    ap.add_argument("--eval-slice", type=int, default=5000)
    ap.add_argument("--eval-thresholds", type=str, default="0,5,10,25,50")

    # Deploy/bot knobs
    ap.add_argument("--entry-controller", type=str, default=os.environ.get("ENTRY_CONTROLLER", "q_scorer"))
    ap.add_argument("--entry-q-threshold", type=float, default=float(os.environ.get("ENTRY_Q_THRESHOLD", "0.0")))
    ap.add_argument("--bot-symbol", type=str, default=os.environ.get("BOT_SYMBOL", "BITX"))
    ap.add_argument("--bot-initial-balance", type=float, default=float(os.environ.get("BOT_INITIAL_BALANCE", "25000.0")))
    ap.add_argument("--bot-max-cycles", type=int, default=int(os.environ.get("BOT_MAX_CYCLES", "60")))
    ap.add_argument("--bot-timeout-sec", type=int, default=int(os.environ.get("BOT_TIMEOUT_SEC", "600")))

    args = ap.parse_args()

    steps = {s.strip().lower() for s in str(args.steps).split(",") if s.strip()}
    run_dir = str(args.run_dir).strip()
    if not run_dir:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = str(repo_root / "output3" / "models" / f"run_qpipeline_{ts}")

    run_dir_path = Path(run_dir)
    decisions_path = run_dir_path / "state" / "decision_records.jsonl"
    missed_path = run_dir_path / "state" / "missed_opportunities_full.jsonl"

    env = dict(os.environ)
    # Make run deterministic-ish and safe for offline dataset generation
    env["MODEL_RUN_DIR"] = str(run_dir_path)
    env["ENABLE_LIVE_MODE"] = "false"
    env["USE_SHADOW_TRADING"] = "0"
    env["Q_HORIZON_MINUTES"] = str(int(args.horizon))
    env["TT_Q_LABELS"] = str(args.q_labels)
    env["TT_MAX_CYCLES"] = str(int(args.tt_max_cycles))
    env["TT_TRAIN_MIN_CONF"] = str(float(args.tt_train_min_conf))
    env["TT_TRAIN_MIN_ABS_RET"] = str(float(args.tt_train_min_abs_ret))
    env["TT_TRAIN_MAX_POSITIONS"] = str(int(args.tt_train_max_positions))
    env["Q_GHOST_DELTA"] = str(float(args.ghost_delta))
    env["Q_POS_WEIGHT"] = str(float(args.pos_weight))
    env["Q_LABEL_CLIP"] = str(float(args.label_clip))
    env["Q_MIN_MATCHED_ROWS"] = str(int(args.min_rows))
    env["ENTRY_CONTROLLER"] = str(args.entry_controller)
    env["ENTRY_Q_THRESHOLD"] = str(float(args.entry_q_threshold))
    env["BOT_SYMBOL"] = str(args.bot_symbol)
    env["BOT_INITIAL_BALANCE"] = str(float(args.bot_initial_balance))
    env["BOT_MAX_CYCLES"] = str(int(args.bot_max_cycles))

    py = sys.executable

    force_mode = str(getattr(args, "force_mode", "PAPER_TRADING") or "PAPER_TRADING").strip().upper()
    if force_mode not in ("PAPER_TRADING", "SIMULATION"):
        force_mode = "PAPER_TRADING"
    force_enabled = not bool(getattr(args, "no_force_mode", False))

    # Reuse the existing guard but write the desired mode.
    class _ModeGuard(_TradingModeGuard):
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
                "mode": force_mode,
                "description": f"Forced {force_mode} for offline pipeline",
                "live_trading": {"enabled": False, "tradier_sandbox": True},
            }
            _write_text(self.path, json.dumps(cfg, indent=2))
            return self

    try:
        with _ModeGuard(repo_root, enabled=bool(force_enabled)):
            if "dataset" in steps:
                _run([py, str(out3 / "scripts" / "train_time_travel.py")], cwd=repo_root, env=env)
                if not decisions_path.exists():
                    print(f"\nERROR: Missing decisions file: {decisions_path}")
                    return 2
                if not missed_path.exists():
                    print(f"\nERROR: Missing missed file: {missed_path}")
                    return 2
                print("\nDataset ready:")
                print(f"- {decisions_path}")
                print(f"- {missed_path}")

            if "train" in steps:
                _run(
                    [
                        py,
                        str(out3 / "training" / "train_q_scorer.py"),
                        "--decisions",
                        str(decisions_path),
                        "--missed",
                        str(missed_path),
                        "--horizon",
                        str(int(args.horizon)),
                        "--state-filter",
                        str(args.state_filter),
                        "--min-rows",
                        str(int(args.min_rows)),
                        "--pos-weight",
                        str(float(args.pos_weight)),
                        "--label-clip",
                        str(float(args.label_clip)),
                    ],
                    cwd=repo_root,
                    env=env,
                )

            if "eval" in steps:
                _run(
                    [
                        py,
                        str(out3 / "tools" / "eval_q_scorer.py"),
                        "--run-dir",
                        str(run_dir_path),
                        "--horizon",
                        str(int(args.horizon)),
                        "--slice",
                        str(int(args.eval_slice)),
                        f"--thresholds={str(args.eval_thresholds)}",
                    ],
                    cwd=repo_root,
                    env=env,
                )

            if "deploy" in steps:
                model_path = repo_root / "output3" / "models" / "q_scorer.pt"
                meta_path = repo_root / "output3" / "models" / "q_scorer_metadata.json"
                if not model_path.exists():
                    print(f"\nERROR: Missing model artifact: {model_path}")
                    return 3
                if not meta_path.exists():
                    print(f"\nERROR: Missing metadata artifact: {meta_path}")
                    return 3
                print("\nDeployed artifacts present:")
                print(f"- {model_path}")
                print(f"- {meta_path}")

            if "bot" in steps:
                cmd = [py, str(out3 / "unified_options_trading_bot.py")]
                print("\n$ " + " ".join(cmd))
                try:
                    subprocess.run(cmd, cwd=str(repo_root), env=env, check=True, timeout=int(args.bot_timeout_sec))
                except subprocess.TimeoutExpired:
                    print(f"\n[WARN] Bot step timed out after {int(args.bot_timeout_sec)}s; stopping pipeline run.")

    except subprocess.CalledProcessError as e:
        print(f"\nPipeline failed with exit code {e.returncode}")
        return int(e.returncode or 1)

    print("\nDONE")
    print(f"Run dir: {run_dir_path}")
    print("Model artifacts (default):")
    print(f"- {repo_root / 'output3' / 'models' / 'q_scorer.pt'}")
    print(f"- {repo_root / 'output3' / 'models' / 'q_scorer_metadata.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())





