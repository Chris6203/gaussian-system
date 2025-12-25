#!/usr/bin/env python3
"""
Summarize missed opportunity records (JSONL).

Reads MissedOpportunityRecord rows produced by backend/decision_pipeline.py and reports:
- total missed positive EV events by veto reason
- avg ghost_best_reward when vetoed vs policy HOLD
- top feature correlations (simple Pearson) with ghost_best_reward

Usage:
  python tools/summarize_missed.py --path data/missed_opportunities_full.jsonl
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        if x is None:
            return float(default)
        return float(x)
    except Exception:
        return float(default)


def load_rows(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception:
                continue
    return rows


def compute_feature_matrix(rows: List[Dict[str, Any]]) -> Tuple[np.ndarray, List[str], np.ndarray]:
    """
    Returns (X, feature_names, y) where y is ghost_best_reward.
    Rows lacking features are skipped.
    """
    Xs: List[List[float]] = []
    ys: List[float] = []
    names: List[str] = []

    for r in rows:
        fv = r.get("feature_vector")
        fn = r.get("feature_names")
        y = r.get("ghost_best_reward")
        if not isinstance(fv, list) or not isinstance(fn, list) or len(fv) != len(fn):
            continue
        if not names:
            names = [str(x) for x in fn]
        # Enforce consistent schema/feature length
        if len(fn) != len(names):
            continue
        Xs.append([_safe_float(v, 0.0) for v in fv])
        ys.append(_safe_float(y, 0.0))

    if not Xs:
        return np.zeros((0, 0), dtype=np.float32), [], np.zeros((0,), dtype=np.float32)
    return np.asarray(Xs, dtype=np.float32), names, np.asarray(ys, dtype=np.float32)


def pearson_corr(x: np.ndarray, y: np.ndarray) -> float:
    if x.size == 0 or y.size == 0:
        return 0.0
    if np.std(x) < 1e-9 or np.std(y) < 1e-9:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", type=str, default="data/missed_opportunities_full.jsonl")
    ap.add_argument("--topk", type=int, default=15)
    args = ap.parse_args()

    path = Path(args.path)
    rows = load_rows(path)

    if not rows:
        print(f"No rows found at: {path}")
        return 0

    # Missed positive EV = best reward > 0
    missed_pos = [r for r in rows if _safe_float(r.get("ghost_best_reward", 0.0)) > 0.0]

    by_reason = defaultdict(int)
    for r in missed_pos:
        reasons = r.get("veto_reasons") or []
        if not reasons:
            by_reason["(policy_hold)"] += 1
        else:
            for reason in reasons:
                by_reason[str(reason)] += 1

    # Vetoed vs policy hold averages
    vetoed_rewards = [_safe_float(r.get("ghost_best_reward", 0.0)) for r in missed_pos if (r.get("veto_reasons") or [])]
    policy_rewards = [_safe_float(r.get("ghost_best_reward", 0.0)) for r in missed_pos if not (r.get("veto_reasons") or [])]

    print("=" * 80)
    print(f"Missed opportunities file: {path}")
    print(f"Total HOLD decisions logged: {len(rows)}")
    print(f"Missed positive-EV events (ghost_best_reward > 0): {len(missed_pos)}")
    print("=" * 80)

    print("\nMissed positive-EV by veto reason:")
    for reason, cnt in sorted(by_reason.items(), key=lambda kv: kv[1], reverse=True)[:50]:
        print(f"- {cnt:6d}  {reason}")

    def _avg(xs: List[float]) -> float:
        return float(np.mean(xs)) if xs else 0.0

    print("\nAvg ghost_best_reward (only positive-EV misses):")
    print(f"- vetoed:      {_avg(vetoed_rewards):+.2f}  (n={len(vetoed_rewards)})")
    print(f"- policy_hold: {_avg(policy_rewards):+.2f}  (n={len(policy_rewards)})")

    # Correlations with ghost_best_reward across all rows (not just positive-EV)
    X, names, y = compute_feature_matrix(rows)
    if X.size and len(names) == X.shape[1]:
        corrs = []
        for i, name in enumerate(names):
            corrs.append((name, pearson_corr(X[:, i], y)))
        corrs.sort(key=lambda t: abs(t[1]), reverse=True)

        print(f"\nTop {args.topk} absolute feature correlations with ghost_best_reward:")
        for name, c in corrs[: max(1, args.topk)]:
            print(f"- {c:+.3f}  {name}")
    else:
        print("\nNo feature vectors found to compute correlations.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())





