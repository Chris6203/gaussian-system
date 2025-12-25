#!/usr/bin/env python3
"""
Q Entry Optimization Runner
==========================

Goal: automate the loop:
  dataset -> train -> (backtest sweep over entry gates) -> pick best -> optional deploy

This is designed for *offline* optimization and is safe-by-default:
- Forces PAPER_TRADING via .trading_mode_config.json during subprocess runs (no Tradier calls).
- Writes each experiment into its own run directory with a SUMMARY.txt.

What it optimizes (grid search):
- ENTRY_Q_THRESHOLD
- Q_ENTRY_MIN_CONF
- Q_ENTRY_MIN_ABS_RET

It produces:
- <out_root>/leaderboard.csv
- <out_root>/best_params.json
- <out_root>/best/SUMMARY.txt (copied from the best run)

Notes:
- The time-travel SUMMARY.txt reflects *simulated trading P&L* under the chosen entry policy.
- For speed, use smaller --eval-cycles; for confidence, increase it.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _output3_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


class _TradingModeGuard:
    """
    Temporarily force `.trading_mode_config.json` to PAPER_TRADING for subprocess runs.
    Restores original contents afterwards.
    """

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
            "description": f"Forced {self.mode} for Q optimization",
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


def _run(cmd: List[str], *, cwd: Path, env: Dict[str, str], timeout_sec: int) -> None:
    print("\n$ " + " ".join(cmd))
    subprocess.run(cmd, cwd=str(cwd), env=env, check=True, timeout=int(timeout_sec))


def _parse_list(s: str) -> List[float]:
    out: List[float] = []
    for part in (s or "").split(","):
        part = part.strip()
        if not part:
            continue
        out.append(float(part))
    return out


def _parse_list_str(s: str) -> List[str]:
    return [p.strip() for p in (s or "").split(",") if p.strip()]


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
            # "Final Balance:   $4,977.30"
            try:
                val = s.split(":", 1)[1].strip().replace("$", "").replace(",", "")
                m.final_balance = float(val)
            except Exception:
                pass
        elif s.startswith("P&L:"):
            # "P&L:             $-22.70 (-0.45%)"
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


def main() -> int:
    ap = argparse.ArgumentParser()
    repo_root = _repo_root()
    out3 = _output3_root()
    py = sys.executable

    ap.add_argument("--out-root", type=str, default="", help="Where to write optimization runs (folder).")
    ap.add_argument("--steps", type=str, default="train,sweep", help="Comma list: dataset,train,sweep,deploy_best")
    ap.add_argument("--timeout-sec", type=int, default=3600, help="Per-subprocess timeout (seconds).")
    ap.add_argument("--force-mode", type=str, default=os.environ.get("PIPELINE_FORCE_MODE", "PAPER_TRADING"))
    ap.add_argument("--no-force-mode", action="store_true")

    # Dataset generation (optional)
    ap.add_argument("--dataset-cycles", type=int, default=int(os.environ.get("TT_MAX_CYCLES", "0") or "0"))
    ap.add_argument("--q-labels", type=str, default=os.environ.get("TT_Q_LABELS", "all"))
    ap.add_argument("--tt-train-min-conf", type=float, default=float(os.environ.get("TT_TRAIN_MIN_CONF", "0.30")))
    ap.add_argument("--tt-train-min-abs-ret", type=float, default=float(os.environ.get("TT_TRAIN_MIN_ABS_RET", "0.0010")))
    ap.add_argument("--tt-train-max-positions", type=int, default=int(os.environ.get("TT_TRAIN_MAX_POSITIONS", "1")))

    # Train settings
    ap.add_argument("--train-run-dir", type=str, default="", help="Use an existing dataset run dir (has state/*.jsonl).")
    ap.add_argument("--horizon", type=int, default=int(os.environ.get("Q_HORIZON_MINUTES", "30")))
    ap.add_argument("--state-filter", type=str, default=os.environ.get("Q_STATE_FILTER", "flat"))
    ap.add_argument("--min-rows", type=int, default=int(os.environ.get("Q_MIN_MATCHED_ROWS", "1500")))
    ap.add_argument("--pos-weight", type=float, default=float(os.environ.get("Q_POS_WEIGHT", "20")))
    ap.add_argument("--label-clip", type=float, default=float(os.environ.get("Q_LABEL_CLIP", "200")))

    # Sweep settings
    ap.add_argument("--eval-cycles", type=int, default=800)
    ap.add_argument("--thresholds", type=str, default="5,7.5,10")
    ap.add_argument("--min-confs", type=str, default="0.25,0.30,0.35")
    ap.add_argument("--min-abs-rets", type=str, default="0.0008,0.0010,0.0012")
    ap.add_argument("--max-positions", type=int, default=1)
    ap.add_argument(
        "--min-trades",
        type=int,
        default=1,
        help="Ignore parameter sets that produce fewer trades than this (avoids 'best=do nothing').",
    )
    ap.add_argument("--stage2", action="store_true", help="Re-evaluate top K configs on a longer window.")
    ap.add_argument("--stage2-cycles", type=int, default=2500, help="Cycles for stage2 re-evaluation.")
    ap.add_argument("--stage2-topk", type=int, default=5, help="How many top configs to re-run in stage2.")
    ap.add_argument("--stage2-min-trades", type=int, default=30, help="Min trades for stage2 best selection.")

    # Model paths (so we don't need to overwrite global artifacts unless we want to)
    ap.add_argument("--model-path", type=str, default="", help="Where to write trained model for this optimization.")
    ap.add_argument("--meta-path", type=str, default="", help="Where to write trained metadata for this optimization.")

    args = ap.parse_args()

    steps = {s.strip().lower() for s in str(args.steps).split(",") if s.strip()}

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_root = Path(str(args.out_root).strip() or (repo_root / "output3" / "models" / f"opt_q_entry_{ts}"))
    out_root.mkdir(parents=True, exist_ok=True)

    # Where the optimized model artifacts live (default under out_root/model/)
    model_path = Path(str(args.model_path).strip() or (out_root / "model" / "q_scorer.pt"))
    meta_path = Path(str(args.meta_path).strip() or (out_root / "model" / "q_scorer_metadata.json"))
    model_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.parent.mkdir(parents=True, exist_ok=True)

    env_base = dict(os.environ)
    env_base["ENABLE_LIVE_MODE"] = "false"
    env_base["USE_SHADOW_TRADING"] = "0"
    env_base["Q_HORIZON_MINUTES"] = str(int(args.horizon))

    force_enabled = not bool(args.no_force_mode)
    guard_mode = str(args.force_mode or "PAPER_TRADING").strip().upper()

    # Dataset run dir (where decision_records + missed_opps live)
    dataset_run_dir: Optional[Path] = None
    if args.train_run_dir:
        dataset_run_dir = Path(str(args.train_run_dir).strip())
    else:
        dataset_run_dir = out_root / "dataset"

    with _TradingModeGuard(repo_root, enabled=bool(force_enabled), mode=guard_mode):
        if "dataset" in steps:
            env = dict(env_base)
            env["MODEL_RUN_DIR"] = str(dataset_run_dir)
            env["TT_MAX_CYCLES"] = str(int(args.dataset_cycles))
            env["TT_Q_LABELS"] = str(args.q_labels)
            env["TT_TRAIN_MIN_CONF"] = str(float(args.tt_train_min_conf))
            env["TT_TRAIN_MIN_ABS_RET"] = str(float(args.tt_train_min_abs_ret))
            env["TT_TRAIN_MAX_POSITIONS"] = str(int(args.tt_train_max_positions))
            _run([py, str(out3 / "scripts" / "train_time_travel.py")], cwd=repo_root, env=env, timeout_sec=int(args.timeout_sec))

        # Train Q scorer from dataset
        if "train" in steps:
            decisions = Path(dataset_run_dir) / "state" / "decision_records.jsonl"
            missed = Path(dataset_run_dir) / "state" / "missed_opportunities_full.jsonl"
            if not decisions.exists() or not missed.exists():
                print(f"\nERROR: dataset files missing under: {dataset_run_dir}")
                print(f"- {decisions}")
                print(f"- {missed}")
                return 2
            env = dict(env_base)
            env["Q_SCORER_MODEL_PATH"] = str(model_path)
            env["Q_SCORER_METADATA_PATH"] = str(meta_path)
            _run(
                [
                    py,
                    str(out3 / "training" / "train_q_scorer.py"),
                    "--decisions",
                    str(decisions),
                    "--missed",
                    str(missed),
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
                    "--out-model",
                    str(model_path),
                    "--out-meta",
                    str(meta_path),
                ],
                cwd=repo_root,
                env=env,
                timeout_sec=int(args.timeout_sec),
            )

        if "sweep" in steps:
            thresholds = _parse_list(str(args.thresholds))
            min_confs = _parse_list(str(args.min_confs))
            min_abs_rets = _parse_list(str(args.min_abs_rets))

            results: List[Dict[str, object]] = []
            runs_dir = out_root / "runs"
            runs_dir.mkdir(parents=True, exist_ok=True)

            for thr in thresholds:
                for mc in min_confs:
                    for mar in min_abs_rets:
                        run_name = f"thr{thr:g}_conf{mc:g}_ret{mar:g}".replace(".", "p")
                        run_dir = runs_dir / run_name
                        env = dict(env_base)
                        env["MODEL_RUN_DIR"] = str(run_dir)
                        env["TT_MAX_CYCLES"] = str(int(args.eval_cycles))
                        env["TT_TRAIN_MAX_POSITIONS"] = str(int(args.max_positions))
                        env["ENTRY_CONTROLLER"] = "q_scorer"
                        env["ENTRY_Q_THRESHOLD"] = str(float(thr))
                        env["Q_ENTRY_MIN_CONF"] = str(float(mc))
                        env["Q_ENTRY_MIN_ABS_RET"] = str(float(mar))
                        env["Q_SCORER_MODEL_PATH"] = str(model_path)
                        env["Q_SCORER_METADATA_PATH"] = str(meta_path)

                        _run([py, str(out3 / "scripts" / "train_time_travel.py")], cwd=repo_root, env=env, timeout_sec=int(args.timeout_sec))

                        summary_path = run_dir / "SUMMARY.txt"
                        met = _read_summary(summary_path)
                        row: Dict[str, object] = {
                            "run_dir": str(run_dir),
                            "summary": str(summary_path),
                            "thr": float(thr),
                            "min_conf": float(mc),
                            "min_abs_ret": float(mar),
                            "final_balance": float(met.final_balance),
                            "pnl": float(met.pnl),
                            "total_trades": int(met.total_trades),
                            "total_cycles": int(met.total_cycles),
                            "win_rate": float(met.win_rate),
                        }
                        # Derived metrics
                        try:
                            trades_i = int(row["total_trades"])
                            pnl_f = float(row["pnl"])
                            row["pnl_per_trade"] = float(pnl_f / max(1, trades_i))
                        except Exception:
                            row["pnl_per_trade"] = 0.0
                        results.append(row)
                        print(f"[OK] {run_name} -> pnl={met.pnl:+.2f} trades={met.total_trades}")

            # Eligibility + sorting:
            # - first, prefer configs that meet min-trades (avoids "best=do nothing")
            # - then maximize pnl
            # - then prefer fewer trades as a tie-breaker (more conservative)
            min_trades = int(max(0, args.min_trades))
            for r in results:
                try:
                    r["eligible"] = bool(int(r.get("total_trades", 0) or 0) >= min_trades)
                except Exception:
                    r["eligible"] = False
            results.sort(
                key=lambda r: (
                    1 if bool(r.get("eligible", False)) else 0,
                    float(r.get("pnl", 0.0)),
                    -int(r.get("total_trades", 0) or 0),
                ),
                reverse=True,
            )

            # Pick a best that actually trades (by default).
            tradable = [r for r in results if int(r.get("total_trades", 0) or 0) >= min_trades]
            best = tradable[0] if tradable else (results[0] if results else None)

            leaderboard_path = out_root / "leaderboard.csv"
            with leaderboard_path.open("w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(
                    f,
                    fieldnames=[
                        "pnl",
                        "final_balance",
                        "total_trades",
                        "win_rate",
                        "pnl_per_trade",
                        "eligible",
                        "thr",
                        "min_conf",
                        "min_abs_ret",
                        "run_dir",
                        "summary",
                    ],
                )
                w.writeheader()
                for r in results:
                    w.writerow({k: r.get(k) for k in w.fieldnames})

            best_params_path = out_root / "best_params.json"
            _write_text(best_params_path, json.dumps(best or {}, indent=2))

            # Copy best summary for convenience
            best_dir = out_root / "best"
            best_dir.mkdir(parents=True, exist_ok=True)
            if best and best.get("summary"):
                try:
                    shutil.copyfile(str(best["summary"]), str(best_dir / "SUMMARY.txt"))
                except Exception:
                    pass

            # ---------------------------
            # Stage 2: longer re-eval
            # ---------------------------
            if bool(getattr(args, "stage2", False)) and results:
                stage2_dir = out_root / "stage2"
                stage2_runs = stage2_dir / "runs"
                stage2_dir.mkdir(parents=True, exist_ok=True)
                stage2_runs.mkdir(parents=True, exist_ok=True)

                topk = int(max(1, getattr(args, "stage2_topk", 5)))
                # Prefer eligible configs from stage1; fall back to top results.
                stage1_min_trades = int(max(0, args.min_trades))
                eligible_stage1 = [r for r in results if int(r.get("total_trades", 0) or 0) >= stage1_min_trades]
                seeds = (eligible_stage1[:topk] if eligible_stage1 else results[:topk])

                stage2_results: List[Dict[str, object]] = []
                for r in seeds:
                    thr = float(r.get("thr", 0.0))
                    mc = float(r.get("min_conf", 0.0))
                    mar = float(r.get("min_abs_ret", 0.0))
                    run_name = f"thr{thr:g}_conf{mc:g}_ret{mar:g}".replace(".", "p")
                    run_dir = stage2_runs / run_name
                    env = dict(env_base)
                    env["MODEL_RUN_DIR"] = str(run_dir)
                    env["TT_MAX_CYCLES"] = str(int(getattr(args, "stage2_cycles", 2500)))
                    env["TT_TRAIN_MAX_POSITIONS"] = str(int(args.max_positions))
                    env["ENTRY_CONTROLLER"] = "q_scorer"
                    env["ENTRY_Q_THRESHOLD"] = str(float(thr))
                    env["Q_ENTRY_MIN_CONF"] = str(float(mc))
                    env["Q_ENTRY_MIN_ABS_RET"] = str(float(mar))
                    env["Q_SCORER_MODEL_PATH"] = str(model_path)
                    env["Q_SCORER_METADATA_PATH"] = str(meta_path)
                    _run([py, str(out3 / "scripts" / "train_time_travel.py")], cwd=repo_root, env=env, timeout_sec=int(args.timeout_sec))

                    summary_path = run_dir / "SUMMARY.txt"
                    met = _read_summary(summary_path)
                    row2: Dict[str, object] = {
                        "run_dir": str(run_dir),
                        "summary": str(summary_path),
                        "thr": float(thr),
                        "min_conf": float(mc),
                        "min_abs_ret": float(mar),
                        "final_balance": float(met.final_balance),
                        "pnl": float(met.pnl),
                        "total_trades": int(met.total_trades),
                        "total_cycles": int(met.total_cycles),
                        "win_rate": float(met.win_rate),
                    }
                    try:
                        trades_i = int(row2["total_trades"])
                        pnl_f = float(row2["pnl"])
                        row2["pnl_per_trade"] = float(pnl_f / max(1, trades_i))
                    except Exception:
                        row2["pnl_per_trade"] = 0.0
                    stage2_results.append(row2)
                    print(f"[STAGE2] {run_name} -> pnl={met.pnl:+.2f} trades={met.total_trades}")

                # Choose best stage2 with its own min-trades threshold.
                s2_min_trades = int(max(0, getattr(args, "stage2_min_trades", 30)))
                for r2 in stage2_results:
                    try:
                        r2["eligible"] = bool(int(r2.get("total_trades", 0) or 0) >= s2_min_trades)
                    except Exception:
                        r2["eligible"] = False
                stage2_results.sort(
                    key=lambda r2: (
                        1 if bool(r2.get("eligible", False)) else 0,
                        float(r2.get("pnl", 0.0)),
                        -int(r2.get("total_trades", 0) or 0),
                    ),
                    reverse=True,
                )
                best2 = None
                tradable2 = [r2 for r2 in stage2_results if int(r2.get("total_trades", 0) or 0) >= s2_min_trades]
                best2 = tradable2[0] if tradable2 else (stage2_results[0] if stage2_results else None)

                # Write stage2 leaderboard
                leaderboard2 = stage2_dir / "leaderboard.csv"
                with leaderboard2.open("w", newline="", encoding="utf-8") as f:
                    w = csv.DictWriter(
                        f,
                        fieldnames=[
                            "pnl",
                            "final_balance",
                            "total_trades",
                            "win_rate",
                            "pnl_per_trade",
                            "eligible",
                            "thr",
                            "min_conf",
                            "min_abs_ret",
                            "run_dir",
                            "summary",
                        ],
                    )
                    w.writeheader()
                    for r2 in stage2_results:
                        w.writerow({k: r2.get(k) for k in w.fieldnames})

                _write_text(stage2_dir / "best_params.json", json.dumps(best2 or {}, indent=2))
                best2_dir = stage2_dir / "best"
                best2_dir.mkdir(parents=True, exist_ok=True)
                if best2 and best2.get("summary"):
                    try:
                        shutil.copyfile(str(best2["summary"]), str(best2_dir / "SUMMARY.txt"))
                    except Exception:
                        pass

            print("\nTop 10:")
            for r in results[:10]:
                print(
                    f"- pnl={float(r['pnl']):+.2f} trades={int(r['total_trades'])} eligible={bool(r.get('eligible', False))} "
                    f"thr={float(r['thr']):g} min_conf={float(r['min_conf']):g} min_abs_ret={float(r['min_abs_ret']):g} "
                    f"run={r['run_dir']}"
                )

        if "deploy_best" in steps:
            # Deploy the *trained* model artifacts (not the per-sweep runs) into output3/models/
            dest_model = out3 / "models" / "q_scorer.pt"
            dest_meta = out3 / "models" / "q_scorer_metadata.json"
            dest_model.parent.mkdir(parents=True, exist_ok=True)
            try:
                shutil.copyfile(str(model_path), str(dest_model))
                shutil.copyfile(str(meta_path), str(dest_meta))
                print("\nDeployed:")
                print(f"- {dest_model}")
                print(f"- {dest_meta}")
            except Exception as e:
                print(f"\nERROR: deploy failed: {e}")
                return 3

    print("\nDONE")
    print(f"Optimization root: {out_root}")
    print(f"Leaderboard: {out_root / 'leaderboard.csv'}")
    print(f"Best params: {out_root / 'best_params.json'}")
    if (out_root / "best" / "SUMMARY.txt").exists():
        print(f"Best summary: {out_root / 'best' / 'SUMMARY.txt'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())





