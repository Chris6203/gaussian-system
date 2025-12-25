#!/usr/bin/env python3
"""
Offline Q Regression Trainer (Supervised)
========================================

Trains a conservative, regularized MLP to predict:
  Q_hold, Q_call, Q_put
as expected net reward over a fixed horizon (e.g. 30 minutes).

This is NOT classic RL bootstrapping. Labels come from GhostTradeEvaluator logs.

Inputs:
- predictor_embedding (64-d) from DecisionRecord.trade_state
- a small set of scalar features (from DecisionRecord.trade_state)

Labels:
- Q_hold = 0 (initially)
- Q_call = ghost call reward at the configured horizon
- Q_put  = ghost put  reward at the configured horizon

Splitting:
- Walk-forward split by timestamp (no random shuffle).

Outputs (configurable paths):
- q_scorer.pt
- q_scorer_metadata.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn

# Ensure `output3/` is on sys.path when running this file directly.
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


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


class QScorerNet(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: Sequence[int] = (128, 64), dropout: float = 0.25):
        super().__init__()
        layers: List[nn.Module] = []
        d = int(input_dim)
        for h in hidden_dims:
            layers.append(nn.Linear(d, int(h)))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(float(dropout)))
            d = int(h)
        layers.append(nn.Linear(d, 3))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


@dataclass
class DatasetRow:
    ts: datetime
    x: np.ndarray
    y: np.ndarray


def _extract_horizon_rewards(missed_row: Dict[str, Any], horizon: int) -> Tuple[float, float]:
    """
    Extract horizon-consistent labels.

    Preferred:
    - If ghost_details includes `underlying_return` for the horizon, recompute option P&L
      using a delta-$ approximation:
        pnl_dollars ~= delta * (S * underlying_return) * 100
      then subtract theta + friction_dollars from the log.

    Fallback:
    - Use stored call_reward/put_reward in ghost_details.horizons[h].
    - Else use record-level ghost_reward_call/ghost_reward_put (may be inconsistent).
    """
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


def build_dataset(
    decisions: List[Dict[str, Any]],
    missed: List[Dict[str, Any]],
    *,
    horizon: int,
    embedding_key: str,
    embedding_dim: int,
    scalar_features: Sequence[str],
    match_tolerance_seconds: int = 5,
    state_filter: str = "all",  # all | flat | in_trade
) -> List[DatasetRow]:
    """
    Build supervised dataset by matching missed-opportunity rows to the nearest decision row
    (same symbol) within tolerance.
    """
    # Index decisions by symbol and time
    dec_by_symbol: Dict[str, List[Tuple[datetime, Dict[str, Any]]]] = {}
    for d in decisions:
        ts = _parse_ts(d.get("timestamp"))
        sym = str(d.get("symbol", ""))
        if ts is None or not sym:
            continue
        dec_by_symbol.setdefault(sym, []).append((ts, d))
    for sym in dec_by_symbol:
        dec_by_symbol[sym].sort(key=lambda t: t[0])

    rows: List[DatasetRow] = []
    tol = float(match_tolerance_seconds)

    for m in missed:
        ts_m = _parse_ts(m.get("timestamp"))
        sym = str(m.get("symbol", ""))
        if ts_m is None or not sym:
            continue

        # Find closest decision record by time (linear scan forward using sorted list)
        cand = dec_by_symbol.get(sym, [])
        if not cand:
            continue

        # binary search-ish
        times = [t[0] for t in cand]
        idx = int(np.searchsorted(np.array(times, dtype="datetime64[ns]"), np.datetime64(ts_m)))
        best = None
        best_dt = 1e18
        for j in (idx - 1, idx, idx + 1):
            if 0 <= j < len(cand):
                dt = abs((cand[j][0] - ts_m).total_seconds())
                if dt < best_dt:
                    best_dt = dt
                    best = cand[j][1]
        if best is None or best_dt > tol:
            continue

        trade_state = best.get("trade_state") or {}
        # Optional state filtering (useful to isolate ENTRY vs EXIT datasets)
        try:
            is_in_trade = bool(trade_state.get("is_in_trade", False))
        except Exception:
            is_in_trade = False
        sf = str(state_filter or "all").strip().lower()
        if sf in ("flat", "flat_only", "entry", "entry_only") and is_in_trade:
            continue
        if sf in ("in_trade", "in_trade_only", "exit", "exit_only") and (not is_in_trade):
            continue

        emb = trade_state.get(embedding_key)
        if not isinstance(emb, list) or len(emb) != embedding_dim:
            continue
        try:
            emb_vec = np.asarray([float(v) for v in emb], dtype=np.float32)
        except Exception:
            continue

        scalars: List[float] = []
        for k in scalar_features:
            v = trade_state.get(k)
            if isinstance(v, bool):
                scalars.append(1.0 if v else 0.0)
            else:
                scalars.append(_sf(v, 0.0))

        x = np.concatenate([emb_vec, np.asarray(scalars, dtype=np.float32)], axis=0)
        call_r, put_r = _extract_horizon_rewards(m, horizon)
        y = np.asarray([0.0, call_r, put_r], dtype=np.float32)
        rows.append(DatasetRow(ts=ts_m, x=x, y=y))

    rows.sort(key=lambda r: r.ts)
    return rows


def walk_forward_split(rows: List[DatasetRow], val_fraction: float) -> Tuple[List[DatasetRow], List[DatasetRow]]:
    if not rows:
        return [], []
    n = len(rows)
    n_val = max(1, int(n * val_fraction))
    split = max(1, n - n_val)
    return rows[:split], rows[split:]


def metrics_best_action(y: np.ndarray) -> int:
    # 0=HOLD, 1=CALL, 2=PUT by argmax of reward vector
    return int(np.argmax(y))


def main() -> int:
    ap = argparse.ArgumentParser()
    base_dir = Path(__file__).resolve().parents[1]
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
    ap.add_argument(
        "--state-filter",
        type=str,
        default=os.environ.get("Q_STATE_FILTER", "all"),
        help="Which states to train on: all | flat | in_trade. (flat is best for entry-Q.)",
    )
    ap.add_argument("--val-fraction", type=float, default=0.2)
    ap.add_argument(
        "--min-rows",
        type=int,
        default=int(os.environ.get("Q_MIN_MATCHED_ROWS", "100")),
        help="Minimum matched (DecisionRecord x MissedOpportunity) rows required to train.",
    )
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--patience", type=int, default=15)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--dropout", type=float, default=0.25)
    ap.add_argument("--hidden", type=str, default="128,64")
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument(
        "--pos-weight",
        type=float,
        default=float(os.environ.get("Q_POS_WEIGHT", "5.0")),
        help="Upweight samples where max(call,put) > 0. Helps learn rare positive tails.",
    )
    ap.add_argument(
        "--label-clip",
        type=float,
        default=float(os.environ.get("Q_LABEL_CLIP", "250.0")),
        help="Clip call/put labels to [-label_clip, +label_clip] for stability.",
    )
    default_out_model = str(base_dir / "models" / "q_scorer.pt")
    default_out_meta = str(base_dir / "models" / "q_scorer_metadata.json")
    ap.add_argument("--out-model", type=str, default=os.environ.get("Q_SCORER_MODEL_PATH", default_out_model))
    ap.add_argument("--out-meta", type=str, default=os.environ.get("Q_SCORER_METADATA_PATH", default_out_meta))
    args = ap.parse_args()

    # Convenience: if running right after time-travel, use that run's state/ JSONL.
    decisions_path = Path(args.decisions)
    missed_path = Path(args.missed)
    run_dir = str(args.run_dir or "").strip()
    if run_dir:
        cand_dec = Path(run_dir) / "state" / "decision_records.jsonl"
        cand_miss = Path(run_dir) / "state" / "missed_opportunities_full.jsonl"
        # Only override when caller is still using the defaults (or empty env vars).
        if str(decisions_path) == str(default_decisions) and cand_dec.exists():
            decisions_path = cand_dec
        if str(missed_path) == str(default_missed) and cand_miss.exists():
            missed_path = cand_miss
    out_model = Path(args.out_model)
    out_meta = Path(args.out_meta)
    out_model.parent.mkdir(parents=True, exist_ok=True)
    out_meta.parent.mkdir(parents=True, exist_ok=True)

    decisions = _read_jsonl(decisions_path)
    missed = _read_jsonl(missed_path)

    if not decisions:
        print(f"No DecisionRecord rows found at {decisions_path}")
        return 1
    if not missed:
        print(f"No MissedOpportunity rows found at {missed_path}")
        return 1

    # Feature selection (conservative, small scalars)
    embedding_key = "predictor_embedding"
    embedding_dim = 64
    scalar_features = [
        "vix_bb_pos",
        "hmm_trend",
        "hmm_volatility",
        "hmm_liquidity",
        "hmm_confidence",
        "fillability",
        "exp_slippage",
        "exp_ttf",
        "prediction_uncertainty",
        "predicted_return",
        "predicted_volatility",
        "prediction_confidence",
        "volume_spike",
        "liquidity_proxy",
        "spread_proxy",
        "is_in_trade",
        "time_in_trade_minutes",
        "unrealized_pnl_pct",
    ]

    rows = build_dataset(
        decisions,
        missed,
        horizon=args.horizon,
        embedding_key=embedding_key,
        embedding_dim=embedding_dim,
        scalar_features=scalar_features,
        state_filter=str(args.state_filter),
    )

    if len(rows) < int(args.min_rows):
        print(
            f"Not enough matched rows to train (matched={len(rows)} < min_rows={int(args.min_rows)}). "
            "Need more logged HOLD/missed rows (or lower --min-rows for a quick smoke test)."
        )
        return 1

    train_rows, val_rows = walk_forward_split(rows, args.val_fraction)
    print(f"Matched dataset rows: {len(rows)} | train={len(train_rows)} | val={len(val_rows)} | horizon={args.horizon}m")

    X_train = np.stack([r.x for r in train_rows]).astype(np.float32)
    y_train = np.stack([r.y for r in train_rows]).astype(np.float32)
    X_val = np.stack([r.x for r in val_rows]).astype(np.float32)
    y_val = np.stack([r.y for r in val_rows]).astype(np.float32)

    # Optional label clipping (call/put only; hold stays 0).
    clip = float(max(0.0, args.label_clip))
    if clip > 0:
        y_train[:, 1:] = np.clip(y_train[:, 1:], -clip, clip)
        y_val[:, 1:] = np.clip(y_val[:, 1:], -clip, clip)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    hidden_dims = [int(x) for x in args.hidden.split(",") if x.strip()]
    model = QScorerNet(input_dim=X_train.shape[1], hidden_dims=hidden_dims, dropout=args.dropout).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    # We'll compute a weighted MSE manually to upweight positive-EV tails.
    pos_w = float(max(1.0, args.pos_weight))

    # Simple early stopping
    best_val = float("inf")
    best_state = None
    bad = 0

    def _batches(X: np.ndarray, y: np.ndarray, bs: int):
        n = X.shape[0]
        idx = np.arange(n)
        np.random.shuffle(idx)
        for i in range(0, n, bs):
            j = idx[i : i + bs]
            yield X[j], y[j]

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        seen = 0
        for xb, yb in _batches(X_train, y_train, args.batch):
            xt = torch.from_numpy(xb).to(device)
            yt = torch.from_numpy(yb).to(device)
            pred = model(xt)
            per = torch.mean((pred - yt) ** 2, dim=1)  # per-sample MSE over 3 outputs
            best = torch.max(yt[:, 1:], dim=1).values  # best(call,put)
            w = torch.where(best > 0.0, torch.full_like(best, pos_w), torch.ones_like(best))
            loss = torch.mean(per * w)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            train_loss += float(loss.item()) * xb.shape[0]
            seen += xb.shape[0]
        train_loss /= max(1, seen)

        model.eval()
        with torch.no_grad():
            pv = model(torch.from_numpy(X_val).to(device))
            yt = torch.from_numpy(y_val).to(device)
            per = torch.mean((pv - yt) ** 2, dim=1)
            best = torch.max(yt[:, 1:], dim=1).values
            w = torch.where(best > 0.0, torch.full_like(best, pos_w), torch.ones_like(best))
            vloss = float(torch.mean(per * w).item())

        if vloss < best_val - 1e-6:
            best_val = vloss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1

        if epoch % 10 == 0 or epoch == 1:
            print(f"epoch {epoch:4d} | train_mse={train_loss:.4f} | val_mse={vloss:.4f} | best={best_val:.4f} | bad={bad}/{args.patience}")

        if bad >= args.patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    # Final metrics
    model.eval()
    with torch.no_grad():
        yhat_val = model(torch.from_numpy(X_val).to(device)).cpu().numpy().astype(np.float32)

    mse = float(np.mean((yhat_val - y_val) ** 2))
    mse_hold = float(np.mean((yhat_val[:, 0] - y_val[:, 0]) ** 2))
    mse_call = float(np.mean((yhat_val[:, 1] - y_val[:, 1]) ** 2))
    mse_put = float(np.mean((yhat_val[:, 2] - y_val[:, 2]) ** 2))

    # Best-action accuracy on val
    y_best = np.argmax(y_val, axis=1)
    yhat_best = np.argmax(yhat_val, axis=1)
    best_acc = float(np.mean((y_best == yhat_best).astype(np.float32)))

    print("=" * 80)
    print(f"VAL MSE: total={mse:.4f} | hold={mse_hold:.4f} | call={mse_call:.4f} | put={mse_put:.4f}")
    print(f"VAL best_action accuracy: {best_acc:.3f}")

    # Best-trade offset calibration:
    # We care about "is there a trade worth taking?" so calibrate the *best-trade* score
    # to match the realized/ghost best reward distribution. This is robust even if per-action
    # labels are noisy/imbalanced.
    pred_best_q = np.max(yhat_val[:, 1:], axis=1)
    actual_best_r = np.max(y_val[:, 1:], axis=1)
    try:
        best_offset = float(np.mean(actual_best_r - pred_best_q))
    except Exception:
        best_offset = 0.0
    best_offset = float(np.clip(best_offset, -200.0, 200.0))
    print("\nBest-trade calibration (val):")
    print(f"- best_offset: {best_offset:+.2f}  (add to CALL/PUT Qs)")

    # Simple calibration-ish text: bin by predicted best Q and compare mean actual best reward
    bins = np.percentile(pred_best_q, [0, 20, 40, 60, 80, 100])
    print("\nCalibration (bins by predicted best-trade Q):")
    for i in range(5):
        lo, hi = bins[i], bins[i + 1]
        mask = (pred_best_q >= lo) & (pred_best_q <= hi) if i == 4 else (pred_best_q >= lo) & (pred_best_q < hi)
        if mask.sum() == 0:
            continue
        print(
            f"- bin {i+1}: predQ in [{lo:+.2f},{hi:+.2f}) n={int(mask.sum())} | "
            f"mean_pred={float(np.mean(pred_best_q[mask])):+.2f} | mean_actual={float(np.mean(actual_best_r[mask])):+.2f}"
        )

    # Save artifacts
    # Atomic writes so a live bot can hot-reload safely.
    tmp_model = out_model.with_suffix(out_model.suffix + ".tmp")
    torch.save(model.state_dict(), tmp_model)
    os.replace(str(tmp_model), str(out_model))
    meta = {
        "schema_version": "trade_state_v1",
        "horizon_minutes": int(args.horizon),
        "embedding_key": embedding_key,
        "embedding_dim": int(embedding_dim),
        "scalar_features": list(scalar_features),
        "state_filter": str(args.state_filter),
        "input_dim": int(X_train.shape[1]),
        "hidden_dims": hidden_dims,
        "dropout": float(args.dropout),
        "trained_at": datetime.utcnow().isoformat(),
        "paths": {"decisions": str(decisions_path), "missed": str(missed_path)},
        "calibration": {
            "best_offset": float(best_offset),
        },
    }
    tmp_meta = out_meta.with_suffix(out_meta.suffix + ".tmp")
    tmp_meta.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    os.replace(str(tmp_meta), str(out_meta))

    print("\nSaved:")
    print(f"- {out_model}")
    print(f"- {out_meta}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())





