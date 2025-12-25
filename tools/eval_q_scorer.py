#!/usr/bin/env python3
"""
Evaluate Q Scorer on logged decision streams.

Reports:
- trade rate at different thresholds
- avg predicted Q vs realized/ghost reward (ghost-based, horizon-consistent)
- missed positive-EV count remaining after thresholding
- top veto reasons when Q wanted trade but executed HOLD
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Ensure `output3/` is on sys.path when running this file directly.
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from backend.q_entry_controller import QEntryController  # noqa: E402


def _sf(x: Any, default: float = 0.0) -> float:
    try:
        if x is None:
            return float(default)
        return float(x)
    except Exception:
        return float(default)


def _parse_ts(s: Any) -> Optional[datetime]:
    if isinstance(s, datetime):
        return s
    if not isinstance(s, str):
        return None
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00"))
    except Exception:
        try:
            return datetime.fromisoformat(s)
        except Exception:
            return None


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    out: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except Exception:
                continue
    return out


def _extract_horizon_rewards(missed_row: Dict[str, Any], horizon: int) -> Tuple[float, float]:
    gd = missed_row.get("ghost_details") or {}
    hz = (gd.get("horizons") or {}).get(str(horizon))
    if isinstance(hz, dict):
        uret = hz.get("underlying_return", None)
        if uret is not None:
            try:
                delta = float(os.environ.get("Q_GHOST_DELTA", "0.5"))
            except Exception:
                delta = 0.5
            delta = float(max(0.05, min(0.95, delta)))
            S = _sf(missed_row.get("current_price", 0.0), 0.0)
            contract_cost = _sf(gd.get("contract_cost_est", 0.0), 0.0)
            friction_dollars = _sf(gd.get("friction_dollars", 0.0), 0.0)
            theta_cost_pct = _sf(hz.get("theta_cost_pct", 0.0), 0.0)
            theta_dollars = contract_cost * theta_cost_pct
            underlying_change = S * _sf(uret, 0.0)
            gross_call = delta * underlying_change * 100.0
            gross_put = -delta * underlying_change * 100.0
            call_r = gross_call - theta_dollars - friction_dollars
            put_r = gross_put - theta_dollars - friction_dollars
            return float(call_r), float(put_r)
        return _sf(hz.get("call_reward", 0.0), 0.0), _sf(hz.get("put_reward", 0.0), 0.0)
    return _sf(missed_row.get("ghost_reward_call", 0.0), 0.0), _sf(missed_row.get("ghost_reward_put", 0.0), 0.0)


def main() -> int:
    ap = argparse.ArgumentParser()
    base_dir = _ROOT
    default_decisions = str(base_dir / "data" / "decision_records.jsonl")
    default_missed = str(base_dir / "data" / "missed_opportunities_full.jsonl")
    ap.add_argument(
        "--run-dir",
        type=str,
        default=os.environ.get("MODEL_RUN_DIR", ""),
        help="If set, prefer <run_dir>/state/{decision_records.jsonl,missed_opportunities_full.jsonl} when defaults are used.",
    )
    ap.add_argument("--decisions", type=str, default=os.environ.get("DECISION_RECORDS_PATH", default_decisions))
    ap.add_argument("--missed", type=str, default=os.environ.get("MISSED_OPPS_PATH", default_missed))
    ap.add_argument("--horizon", type=int, default=int(os.environ.get("Q_HORIZON_MINUTES", "30")))
    ap.add_argument("--slice", type=int, default=5000, help="Evaluate last N decision rows (time-ordered)")
    ap.add_argument("--thresholds", type=str, default="0,1,2,5,10")
    args = ap.parse_args()

    decisions_path = Path(args.decisions)
    missed_path = Path(args.missed)
    run_dir = str(args.run_dir or "").strip()
    if run_dir:
        cand_dec = Path(run_dir) / "state" / "decision_records.jsonl"
        cand_miss = Path(run_dir) / "state" / "missed_opportunities_full.jsonl"
        if str(decisions_path) == str(default_decisions) and cand_dec.exists():
            decisions_path = cand_dec
        if str(missed_path) == str(default_missed) and cand_miss.exists():
            missed_path = cand_miss

    dec_rows = _read_jsonl(decisions_path)
    miss_rows = _read_jsonl(missed_path)
    if not dec_rows:
        print(f"No decision rows at {decisions_path}")
        return 1
    if not miss_rows:
        print(f"No missed rows at {missed_path}")
        return 1

    # Build quick index for missed by (symbol, timestamp) with tolerance fallback
    miss_by_sym: Dict[str, List[Tuple[datetime, Dict[str, Any]]]] = {}
    for m in miss_rows:
        ts = _parse_ts(m.get("timestamp"))
        sym = str(m.get("symbol", ""))
        if ts is None or not sym:
            continue
        miss_by_sym.setdefault(sym, []).append((ts, m))
    for sym in miss_by_sym:
        miss_by_sym[sym].sort(key=lambda t: t[0])

    def find_missed(sym: str, ts: datetime, tol_sec: int = 5) -> Optional[Dict[str, Any]]:
        cand = miss_by_sym.get(sym, [])
        if not cand:
            return None
        times = [t[0] for t in cand]
        idx = int(np.searchsorted(np.array(times, dtype="datetime64[ns]"), np.datetime64(ts)))
        best = None
        best_dt = 1e18
        for j in (idx - 1, idx, idx + 1):
            if 0 <= j < len(cand):
                dt = abs((cand[j][0] - ts).total_seconds())
                if dt < best_dt:
                    best_dt = dt
                    best = cand[j][1]
        if best is None or best_dt > tol_sec:
            return None
        return best

    # Load model
    qc = QEntryController()
    if not qc.load():
        print("Q scorer not available. Ensure models/q_scorer.pt and models/q_scorer_metadata.json exist.")
        return 2

    # Slice recent decisions
    dec_rows = [r for r in dec_rows if _parse_ts(r.get("timestamp")) is not None]
    dec_rows.sort(key=lambda r: _parse_ts(r.get("timestamp")))
    dec_rows = dec_rows[-int(args.slice) :]

    thresholds = [float(x.strip()) for x in args.thresholds.split(",") if x.strip()]

    print("=" * 80)
    print(f"Evaluating last {len(dec_rows)} decisions | horizon={args.horizon}m")
    print(f"Thresholds: {thresholds}")
    print("=" * 80)

    # Precompute q + ghost labels for rows where available
    rows_eval = []
    for r in dec_rows:
        ts = _parse_ts(r.get("timestamp"))
        sym = str(r.get("symbol", ""))
        if ts is None or not sym:
            continue
        trade_state = r.get("trade_state") or {}
        qd = qc.decide(trade_state, q_threshold=-1e18)  # no threshold yet; we sweep
        if qd is None:
            continue
        miss = find_missed(sym, ts)
        if miss is None:
            continue
        call_r, put_r = _extract_horizon_rewards(miss, args.horizon)
        rows_eval.append((r, qd, call_r, put_r))

    if not rows_eval:
        print("No rows with both Q inputs and horizon-consistent ghost labels. (Check embedding logging + missed logs.)")
        return 0

    for thr in thresholds:
        wanted_trade = 0
        executed_trade = 0
        total = 0
        sum_pred_best_all = 0.0
        sum_realized_best_all = 0.0
        sum_pred_best_wanted = 0.0
        sum_realized_best_wanted = 0.0
        wanted_pos = 0
        missed_pos_remaining = 0
        veto_reasons = defaultdict(int)

        for r, qd, call_r, put_r in rows_eval:
            total += 1
            q_call = qd.q_values["BUY_CALLS"]
            q_put = qd.q_values["BUY_PUTS"]
            best_trade_q = max(q_call, q_put)
            best_action = "BUY_CALLS" if q_call >= q_put else "BUY_PUTS"
            # Compare to executed action
            executed_action = str(r.get("executed_action", "HOLD"))
            if executed_action != "HOLD":
                executed_trade += 1

            # realized/ghost reward if we took best_action
            realized = call_r if best_action == "BUY_CALLS" else put_r
            sum_pred_best_all += best_trade_q
            sum_realized_best_all += realized

            if best_trade_q > thr:
                wanted_trade += 1
                sum_pred_best_wanted += best_trade_q
                sum_realized_best_wanted += realized
                if realized > 0.0:
                    wanted_pos += 1

            # missed positive-EV remaining after thresholding:
            best_real = max(call_r, put_r)
            if best_real > 0.0 and not (best_trade_q > thr):
                missed_pos_remaining += 1

            # veto reasons: Q wants trade, executed HOLD, and there are rejection reasons
            if best_trade_q > thr and executed_action == "HOLD":
                reasons = r.get("rejection_reasons") or []
                if reasons:
                    for rr in reasons:
                        veto_reasons[str(rr)] += 1

        trade_rate = wanted_trade / max(1, total)
        avg_pred_all = sum_pred_best_all / max(1, total)
        avg_real_all = sum_realized_best_all / max(1, total)
        avg_pred_wanted = sum_pred_best_wanted / max(1, wanted_trade)
        avg_real_wanted = sum_realized_best_wanted / max(1, wanted_trade)
        pos_rate_wanted = wanted_pos / max(1, wanted_trade)

        print("\n" + "-" * 80)
        print(f"thr={thr:+.2f} | trade_rate={trade_rate:.1%} | wanted_trade={wanted_trade}/{total} | executed_trade={executed_trade}/{total}")
        print(f"avg predicted best-trade Q (all): {avg_pred_all:+.2f}")
        print(f"avg horizon ghost reward (all): {avg_real_all:+.2f}")
        if wanted_trade > 0:
            print(f"avg predicted best-trade Q (wanted): {avg_pred_wanted:+.2f}")
            print(f"avg horizon ghost reward (wanted): {avg_real_wanted:+.2f}")
            print(f"wanted positive rate (realized>0): {pos_rate_wanted:.1%}")
        print(f"missed positive-EV remaining (ghost best > 0 but suppressed by threshold): {missed_pos_remaining}")
        if veto_reasons:
            print("Top veto reasons when Q wanted trade but executed HOLD:")
            for k, v in sorted(veto_reasons.items(), key=lambda kv: kv[1], reverse=True)[:10]:
                print(f"- {v:6d}  {k}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())





