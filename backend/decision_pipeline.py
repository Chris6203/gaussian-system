#!/usr/bin/env python3
"""
Decision Pipeline Sidecar (Observability + Missed Opportunity Logging)
=====================================================================

This module is **additive** and must not change trading behavior.

It provides:
- Full-state visibility: build a rich TradeState snapshot and encode it to a stable feature vector.
- DecisionRecord JSONL writer: store encoded features + schema metadata per decision.
- GhostTradeEvaluator: counterfactual (expected/approx) trade evaluation for HOLD decisions.
- MissedOpportunityRecord JSONL writer + optional auxiliary dataset writer.

Notes:
- The "ghost" evaluation here is deliberately lightweight and uses a delta-based approximation.
- It subtracts friction (spread/slippage + fees) as required.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np


SCHEMA_VERSION = "trade_state_v1"


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        if x is None:
            return float(default)
        return float(x)
    except Exception:
        return float(default)


def _iso_ts(ts: Any) -> str:
    if isinstance(ts, datetime):
        return ts.isoformat()
    try:
        return str(ts)
    except Exception:
        return ""


@dataclass(frozen=True)
class FullTradeState:
    """
    Full-state snapshot for decision visibility.

    Keep fields additive; do not remove/rename without bumping SCHEMA_VERSION.
    """

    # Identity / time
    timestamp: str
    symbol: str
    current_price: float

    # VIX / vol context
    vix_level: float = 0.0
    vix_bb_pos: float = 0.5
    vix_bb_width: float = 0.0

    # Predictions (multi-timeframe)
    predicted_return_by_tf: Dict[str, float] = field(default_factory=dict)
    predicted_std_by_tf: Dict[str, float] = field(default_factory=dict)
    prediction_confidence: float = 0.0
    predicted_return: float = 0.0  # consensus / primary
    predicted_volatility: float = 0.0
    prediction_uncertainty: float = 0.0

    # Execution head predictions (world-model realism)
    fillability: float = 0.5
    exp_slippage: float = 0.0
    exp_ttf: float = 0.0

    # Predictor latent embedding (preferred input for learned controllers)
    # Stored in the JSON trade_state for training; NOT encoded into feature_vector by default.
    predictor_embedding: Optional[List[float]] = None

    # Past prediction summaries
    pred_prev: float = 0.0
    pred_delta: float = 0.0
    rolling_mean: float = 0.0
    rolling_slope: float = 0.0

    # Market microstructure proxies
    momentum_1m: float = 0.0
    momentum_5m: float = 0.0
    momentum_15m: float = 0.0
    volume_spike: float = 1.0
    liquidity_proxy: float = 0.5  # 0..1 (higher=better)
    spread_proxy: float = 0.5     # 0..1 (higher=larger spread)

    # Regime / HMM
    hmm_trend: float = 0.5
    hmm_volatility: float = 0.5
    hmm_liquidity: float = 0.5
    hmm_confidence: float = 0.5

    # Position state
    is_in_trade: bool = False
    time_in_trade_minutes: float = 0.0
    unrealized_pnl_pct: float = 0.0

    # Misc / existing commonly-used fields
    raw_confidence: float = 0.0
    calibrated_confidence: float = 0.0
    action: str = "HOLD"
    strategy: str = ""
    risk_level: str = ""

    # Optional / extensible payload (keep small; use for sparse extras)
    extras: Dict[str, Any] = field(default_factory=dict)


def build_trade_state(
    *,
    timestamp: datetime,
    symbol: str,
    current_price: float,
    signal: Optional[Mapping[str, Any]] = None,
    bot: Optional[Any] = None,
    rejection_reasons: Optional[Sequence[str]] = None,
) -> FullTradeState:
    """
    Build a FullTradeState snapshot from available inputs.

    This must not perform additional trading-relevant gating or mutate state.
    """
    sig = dict(signal or {})
    rej = list(rejection_reasons or [])

    # VIX features (prefer bot cached values)
    vix_level = _safe_float(getattr(bot, "current_vix", None), default=_safe_float(sig.get("vix_level", 0.0), 0.0))
    vix_bb_pos = _safe_float(getattr(bot, "current_vix_bb_pos", None), default=0.5)

    # Not currently tracked explicitly; keep 0 unless caller provides
    vix_bb_width = _safe_float(sig.get("vix_bb_width", None), default=0.0)

    # Multi-timeframe predictions
    predicted_return_by_tf: Dict[str, float] = {}
    predicted_std_by_tf: Dict[str, float] = {}

    multi_tf = getattr(bot, "last_multi_timeframe_predictions", None)
    if isinstance(multi_tf, dict):
        for tf_name, pred in multi_tf.items():
            if isinstance(pred, dict):
                predicted_return_by_tf[str(tf_name)] = _safe_float(pred.get("predicted_return", 0.0), 0.0)
                # uncertainty / std field names vary
                predicted_std_by_tf[str(tf_name)] = _safe_float(
                    pred.get("predicted_std", pred.get("uncertainty", pred.get("std", 0.0))), 0.0
                )

    # Primary predicted_return / confidence (from signal)
    predicted_return = _safe_float(sig.get("predicted_return", 0.0), 0.0)
    predicted_volatility = _safe_float(sig.get("predicted_volatility", 0.0), 0.0)
    prediction_uncertainty = _safe_float(sig.get("prediction_uncertainty", 0.0), 0.0)
    raw_conf = _safe_float(sig.get("raw_confidence", sig.get("neural_confidence", sig.get("confidence", 0.0))), 0.0)
    calibrated_conf = _safe_float(sig.get("confidence", raw_conf), 0.0)

    fillability = _safe_float(sig.get("fillability", 0.5), 0.5)
    exp_slippage = _safe_float(sig.get("exp_slippage", 0.0), 0.0)
    exp_ttf = _safe_float(sig.get("exp_ttf", 0.0), 0.0)
    predictor_embedding = sig.get("predictor_embedding", None)
    # Time-travel mode often nests this under signal["neural_prediction"].
    if predictor_embedding is None:
        npred = sig.get("neural_prediction") or sig.get("neural_pred") or {}
        if isinstance(npred, dict):
            predictor_embedding = npred.get("predictor_embedding", npred.get("embedding", None))
            # Also allow execution head fallbacks from neural_prediction.
            if "fillability" not in sig and "fillability_mean" in npred:
                fillability = _safe_float(npred.get("fillability_mean", fillability), fillability)
            if "exp_slippage" not in sig and "exp_slippage_mean" in npred:
                exp_slippage = _safe_float(npred.get("exp_slippage_mean", exp_slippage), exp_slippage)
            if "exp_ttf" not in sig and "exp_ttf_mean" in npred:
                exp_ttf = _safe_float(npred.get("exp_ttf_mean", exp_ttf), exp_ttf)

    # Coerce embedding to List[float] if possible
    if predictor_embedding is not None and not isinstance(predictor_embedding, list):
        try:
            predictor_embedding = [float(v) for v in list(predictor_embedding)]
        except Exception:
            predictor_embedding = None
    elif isinstance(predictor_embedding, list):
        try:
            predictor_embedding = [float(v) for v in predictor_embedding]
        except Exception:
            predictor_embedding = None

    # Momentum / volume (signal already carries 5m/15m; keep 1m if present)
    momentum_1m = _safe_float(sig.get("momentum_1m", 0.0), 0.0)
    momentum_5m = _safe_float(sig.get("momentum_5m", sig.get("momentum_5min", 0.0)), 0.0)
    momentum_15m = _safe_float(sig.get("momentum_15m", sig.get("momentum_15min", 0.0)), 0.0)
    volume_spike = _safe_float(sig.get("volume_spike", 1.0), 1.0)

    # Liquidity/spread proxies: best-effort from bot / signal
    liquidity_proxy = _safe_float(getattr(bot, "current_liquidity_proxy", None), default=_safe_float(sig.get("liquidity_proxy", 0.5), 0.5))
    spread_proxy = _safe_float(getattr(bot, "current_spread_proxy", None), default=_safe_float(sig.get("spread_proxy", 0.5), 0.5))

    # HMM regime: map from bot.current_hmm_regime if dict, else defaults
    hmm_trend = 0.5
    hmm_vol = 0.5
    hmm_liq = 0.5
    hmm_conf = 0.5
    hmm = getattr(bot, "current_hmm_regime", None) if bot is not None else None
    if isinstance(hmm, dict):
        # unified_options_trading_bot stores names like 'Bullish'/'Bearish' etc
        trend_name = str(hmm.get("trend", "Neutral"))
        vol_name = str(hmm.get("volatility", "Normal"))
        liq_name = str(hmm.get("liquidity", "Normal"))
        trend_map = {"down": 0.0, "bear": 0.0, "neutral": 0.5, "side": 0.5, "no": 0.5, "up": 1.0, "bull": 1.0}
        vol_map = {"low": 0.0, "normal": 0.5, "high": 1.0}
        liq_map = {"low": 0.0, "normal": 0.5, "high": 1.0}

        def _map_name(name: str, mapping: Dict[str, float], default: float) -> float:
            n = name.strip().lower()
            for k, v in mapping.items():
                if k in n:
                    return v
            return default

        hmm_trend = _map_name(trend_name, trend_map, 0.5)
        hmm_vol = _map_name(vol_name, vol_map, 0.5)
        hmm_liq = _map_name(liq_name, liq_map, 0.5)
        hmm_conf = _safe_float(hmm.get("confidence", hmm.get("combined_confidence", 0.5)), 0.5)
    else:
        # If signal carries scalar HMM fields, prefer those
        hmm_trend = _safe_float(sig.get("hmm_trend", 0.5), 0.5)
        hmm_conf = _safe_float(sig.get("hmm_confidence", 0.5), 0.5)

    # Past prediction summaries from consensus_history (if present)
    pred_prev = 0.0
    pred_delta = 0.0
    rolling_mean = 0.0
    rolling_slope = 0.0

    hist = getattr(bot, "consensus_history", None) if bot is not None else None
    if isinstance(hist, Iterable):
        try:
            items = list(hist)
            if items:
                last = items[-1]
                prev = items[-2] if len(items) >= 2 else None
                pred_prev = _safe_float(prev.get("weighted_return") if isinstance(prev, dict) else 0.0, 0.0)
                cur = _safe_float(last.get("weighted_return") if isinstance(last, dict) else 0.0, 0.0)
                pred_delta = cur - pred_prev
                window = items[-10:] if len(items) >= 3 else items
                vals = [_safe_float(x.get("weighted_return"), 0.0) for x in window if isinstance(x, dict)]
                if vals:
                    rolling_mean = float(np.mean(vals))
                    # simple slope vs index
                    if len(vals) >= 3:
                        xs = np.arange(len(vals), dtype=np.float32)
                        ys = np.asarray(vals, dtype=np.float32)
                        # slope of least squares line
                        cov = float(np.cov(xs, ys, ddof=0)[0, 1])
                        var = float(np.var(xs))
                        rolling_slope = cov / (var + 1e-9)
        except Exception:
            pass

    # Position state (best-effort)
    is_in_trade = False
    time_in_trade_minutes = 0.0
    unrealized_pnl_pct = 0.0
    try:
        if bot is not None and getattr(bot, "paper_trader", None) is not None:
            trades = getattr(bot.paper_trader, "active_trades", []) or []
            filled = [t for t in trades if getattr(getattr(t, "status", None), "value", getattr(t, "status", "")) == "FILLED"]
            is_in_trade = len(filled) > 0
            # Use oldest trade as representative (visibility only)
            if filled:
                entry_ts = getattr(filled[0], "timestamp", None)
                if isinstance(entry_ts, datetime):
                    time_in_trade_minutes = max(0.0, (timestamp - entry_ts).total_seconds() / 60.0)
            acct = bot.paper_trader.get_account_summary()
            invested = _safe_float(acct.get("open_positions_invested", 0.0), 0.0)
            unreal = _safe_float(acct.get("unrealized_pnl", 0.0), 0.0)
            unrealized_pnl_pct = (unreal / invested) if invested > 0 else 0.0
    except Exception:
        pass

    action = str(sig.get("action", "HOLD") or "HOLD")
    strategy = str(sig.get("strategy", "") or "")
    risk_level = str(sig.get("risk_level", "") or "")

    extras: Dict[str, Any] = {}
    if rej:
        extras["rejection_reasons"] = rej

    return FullTradeState(
        timestamp=_iso_ts(timestamp),
        symbol=str(symbol),
        current_price=_safe_float(current_price, 0.0),
        vix_level=_safe_float(vix_level, 0.0),
        vix_bb_pos=_safe_float(vix_bb_pos, 0.5),
        vix_bb_width=_safe_float(vix_bb_width, 0.0),
        predicted_return_by_tf=predicted_return_by_tf,
        predicted_std_by_tf=predicted_std_by_tf,
        prediction_confidence=_safe_float(raw_conf, 0.0),
        predicted_return=_safe_float(predicted_return, 0.0),
        predicted_volatility=_safe_float(predicted_volatility, 0.0),
        prediction_uncertainty=_safe_float(prediction_uncertainty, 0.0),
        fillability=float(min(max(fillability, 0.0), 1.0)),
        exp_slippage=_safe_float(exp_slippage, 0.0),
        exp_ttf=_safe_float(exp_ttf, 0.0),
        predictor_embedding=predictor_embedding,
        pred_prev=_safe_float(pred_prev, 0.0),
        pred_delta=_safe_float(pred_delta, 0.0),
        rolling_mean=_safe_float(rolling_mean, 0.0),
        rolling_slope=_safe_float(rolling_slope, 0.0),
        momentum_1m=_safe_float(momentum_1m, 0.0),
        momentum_5m=_safe_float(momentum_5m, 0.0),
        momentum_15m=_safe_float(momentum_15m, 0.0),
        volume_spike=_safe_float(volume_spike, 1.0),
        liquidity_proxy=float(min(max(liquidity_proxy, 0.0), 1.0)),
        spread_proxy=float(min(max(spread_proxy, 0.0), 1.0)),
        hmm_trend=float(min(max(hmm_trend, 0.0), 1.0)),
        hmm_volatility=float(min(max(hmm_vol, 0.0), 1.0)),
        hmm_liquidity=float(min(max(hmm_liq, 0.0), 1.0)),
        hmm_confidence=float(min(max(hmm_conf, 0.0), 1.0)),
        is_in_trade=bool(is_in_trade),
        time_in_trade_minutes=_safe_float(time_in_trade_minutes, 0.0),
        unrealized_pnl_pct=_safe_float(unrealized_pnl_pct, 0.0),
        raw_confidence=_safe_float(raw_conf, 0.0),
        calibrated_confidence=_safe_float(calibrated_conf, 0.0),
        action=action,
        strategy=strategy,
        risk_level=risk_level,
        extras=extras,
    )


def encode_state(state: FullTradeState) -> Tuple[np.ndarray, List[str], str]:
    """
    Encode FullTradeState to a stable numeric feature vector.

    Returns (feature_vector, feature_names, schema_version).
    """
    feats: List[float] = []
    names: List[str] = []

    # Base numeric fields
    base_fields: List[Tuple[str, float]] = [
        ("current_price", state.current_price),
        ("vix_level", state.vix_level),
        ("vix_bb_pos", state.vix_bb_pos),
        ("vix_bb_width", state.vix_bb_width),
        ("predicted_return", state.predicted_return),
        ("predicted_volatility", state.predicted_volatility),
        ("prediction_uncertainty", state.prediction_uncertainty),
        ("prediction_confidence", state.prediction_confidence),
        ("fillability", state.fillability),
        ("exp_slippage", state.exp_slippage),
        ("exp_ttf", state.exp_ttf),
        ("pred_prev", state.pred_prev),
        ("pred_delta", state.pred_delta),
        ("rolling_mean", state.rolling_mean),
        ("rolling_slope", state.rolling_slope),
        ("momentum_1m", state.momentum_1m),
        ("momentum_5m", state.momentum_5m),
        ("momentum_15m", state.momentum_15m),
        ("volume_spike", state.volume_spike),
        ("liquidity_proxy", state.liquidity_proxy),
        ("spread_proxy", state.spread_proxy),
        ("hmm_trend", state.hmm_trend),
        ("hmm_volatility", state.hmm_volatility),
        ("hmm_liquidity", state.hmm_liquidity),
        ("hmm_confidence", state.hmm_confidence),
        ("is_in_trade", 1.0 if state.is_in_trade else 0.0),
        ("time_in_trade_minutes", state.time_in_trade_minutes),
        ("unrealized_pnl_pct", state.unrealized_pnl_pct),
        ("raw_confidence", state.raw_confidence),
        ("calibrated_confidence", state.calibrated_confidence),
    ]
    for n, v in base_fields:
        names.append(n)
        feats.append(_safe_float(v, 0.0))

    # Multi-timeframe predictions (stable order)
    tf_order = ["5min", "10min", "15min", "20min", "30min", "1h", "2h"]
    for tf in tf_order:
        names.append(f"pred_return_{tf}")
        feats.append(_safe_float(state.predicted_return_by_tf.get(tf, 0.0), 0.0))
        names.append(f"pred_std_{tf}")
        feats.append(_safe_float(state.predicted_std_by_tf.get(tf, 0.0), 0.0))

    vec = np.asarray(feats, dtype=np.float32)
    return vec, names, SCHEMA_VERSION


@dataclass
class DecisionRecord:
    timestamp: str
    symbol: str
    current_price: float

    proposed_action: str
    executed_action: str
    trade_placed: bool

    raw_confidence: float = 0.0
    calibrated_confidence: float = 0.0

    rejection_reasons: List[str] = field(default_factory=list)
    paper_rejection: Optional[Dict[str, Any]] = None

    schema_version: str = SCHEMA_VERSION
    feature_names: List[str] = field(default_factory=list)
    feature_vector: List[float] = field(default_factory=list)

    # keep full state for visibility/debugging (small, but includes dicts)
    trade_state: Dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)


@dataclass
class MissedOpportunityRecord:
    timestamp: str
    symbol: str
    current_price: float
    executed_action: str  # expected "HOLD"

    veto_reasons: List[str] = field(default_factory=list)
    ghost_mode: str = "expected"  # expected | realized (if future prices were available)

    ghost_reward_call: float = 0.0
    ghost_reward_put: float = 0.0
    ghost_best_action: str = "HOLD"
    ghost_best_reward: float = 0.0
    ghost_details: Dict[str, Any] = field(default_factory=dict)

    schema_version: str = SCHEMA_VERSION
    feature_names: List[str] = field(default_factory=list)
    feature_vector: List[float] = field(default_factory=list)

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)


class JsonlWriter:
    """Simple JSONL append-only writer with directory creation."""

    def __init__(self, path: str):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, obj_json_line: str) -> None:
        with self.path.open("a", encoding="utf-8") as f:
            f.write(obj_json_line)
            f.write("\n")


def _black_scholes_call(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """
    Black-Scholes call option price.

    Args:
        S: Current stock price
        K: Strike price
        T: Time to expiration (years)
        r: Risk-free rate (annualized)
        sigma: Volatility (annualized)

    Returns:
        Call option price
    """
    if T <= 0:
        return max(0.0, S - K)

    from scipy.stats import norm

    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)

    call_price = S * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2)
    return max(0.0, call_price)


def _black_scholes_put(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """
    Black-Scholes put option price.

    Args:
        S: Current stock price
        K: Strike price
        T: Time to expiration (years)
        r: Risk-free rate (annualized)
        sigma: Volatility (annualized)

    Returns:
        Put option price
    """
    if T <= 0:
        return max(0.0, K - S)

    from scipy.stats import norm

    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)

    put_price = K * math.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    return max(0.0, put_price)


class GhostTradeEvaluator:
    """
    Lightweight counterfactual evaluator.

    Uses an options proxy:
    - 1 contract
    - premium ~ premium_pct * underlying_price * 100
    - gross option PnL (dollars) ~= delta * (underlying_price * underlying_return) * 100
      (delta-$ approximation; good enough for labeling)
    - subtract friction: spread/slippage (pct of premium, round-trip) + fees (round-trip)
    """

    def __init__(
        self,
        *,
        # Include 15m because many parts of the system use 15m (confidence calibration,
        # missed-play tracking, and multi-timeframe predictions).
        horizons_minutes: Sequence[int] = (10, 15, 20, 30, 60),
        delta_assumed: float = 0.5,
        # NOTE: 0.02 (2%) makes contracts unrealistically expensive for short-dated options.
        # Use a much smaller default; override via env if you want more punitive labels.
        premium_pct: float = 0.003,
        spread_slippage_roundtrip_pct: float = 0.04,
        fee_per_contract: float = 0.73,
    ):
        self.horizons_minutes = list(horizons_minutes)
        self.delta_assumed = float(max(0.05, min(0.95, delta_assumed)))
        self.premium_pct = float(max(0.001, premium_pct))
        self.spread_slippage_roundtrip_pct = float(max(0.0, spread_slippage_roundtrip_pct))
        self.fee_per_contract = float(max(0.0, fee_per_contract))

    def _theta_cost_pct(self, minutes: int, *, theta_daily_pct: float = 0.03) -> float:
        """
        Rough theta decay as percent of premium over the horizon.

        Default ~3% of premium per trading day (390 minutes).
        Override with env `GHOST_THETA_DAILY_PCT`.
        """
        try:
            theta_daily_pct = float(os.environ.get("GHOST_THETA_DAILY_PCT", str(theta_daily_pct)))
        except Exception:
            theta_daily_pct = float(theta_daily_pct)
        theta_daily_pct = float(max(0.0, theta_daily_pct))
        return float(theta_daily_pct * (minutes / 390.0))

    def evaluate(
        self,
        *,
        timestamp: str,
        symbol: str,
        current_price: float,
        predicted_returns_by_horizon: Mapping[int, float],
        vix_level: float,
        liquidity_proxy: float,
    ) -> Dict[str, Any]:
        # Allow env overrides (useful for calibrating label distributions without code changes).
        try:
            delta = float(os.environ.get("GHOST_DELTA_ASSUMED", str(self.delta_assumed)))
        except Exception:
            delta = float(self.delta_assumed)
        delta = float(max(0.05, min(0.95, delta)))
        try:
            base_prem = float(os.environ.get("GHOST_PREMIUM_PCT", str(self.premium_pct)))
        except Exception:
            base_prem = float(self.premium_pct)
        base_prem = float(max(0.0005, min(0.02, base_prem)))
        try:
            spread_rt = float(os.environ.get("GHOST_SPREAD_SLIPPAGE_RT_PCT", str(self.spread_slippage_roundtrip_pct)))
        except Exception:
            spread_rt = float(self.spread_slippage_roundtrip_pct)
        spread_rt = float(max(0.0, spread_rt))
        try:
            fee = float(os.environ.get("GHOST_FEE_PER_CONTRACT", str(self.fee_per_contract)))
        except Exception:
            fee = float(self.fee_per_contract)
        fee = float(max(0.0, fee))

        # Entry premium approximation
        vix = _safe_float(vix_level, 18.0)
        # Increase premium slightly with higher VIX (still bounded)
        premium_pct = float(min(0.02, max(0.0005, base_prem * (1.0 + (vix - 18.0) / 80.0))))
        contract_cost = max(10.0, _safe_float(current_price, 0.0) * premium_pct * 100.0)

        # Liquidity adjustment: worse liquidity -> more friction
        liq = float(min(max(_safe_float(liquidity_proxy, 0.5), 0.0), 1.0))
        friction_pct = spread_rt * (1.0 + (1.0 - liq) * 1.5)
        fees_rt = 2.0 * fee
        friction_dollars = contract_cost * friction_pct + fees_rt

        details = {
            "timestamp": timestamp,
            "symbol": symbol,
            "contract_cost_est": contract_cost,
            "friction_pct_rt": friction_pct,
            "fees_roundtrip": fees_rt,
            "friction_dollars": friction_dollars,
            "delta_assumed": delta,
            "premium_pct_est": premium_pct,
            "horizons": {},
        }

        best_call = -1e18
        best_put = -1e18
        best_call_h = None
        best_put_h = None

        # Check if we should use Black-Scholes pricing (simulator-aligned)
        use_bs = os.environ.get("USE_BS_GHOST_PRICING", "False").lower() in ("true", "1", "yes")

        for h in self.horizons_minutes:
            uret = _safe_float(predicted_returns_by_horizon.get(h, 0.0), 0.0)

            if use_bs:
                # Black-Scholes pricing (simulator-aligned)
                # Parameters for Black-Scholes
                S0 = _safe_float(current_price, 0.0)
                S1 = S0 * (1.0 + uret)  # Future price based on predicted return

                # Use VIX as volatility proxy (VIX is in percentage points, convert to decimal)
                sigma = max(0.10, min(1.0, vix / 100.0))

                # DTE: Use a realistic short-term DTE (7-30 days based on typical strategy)
                # Override with GHOST_DTE_DAYS env var
                try:
                    dte_days = float(os.environ.get("GHOST_DTE_DAYS", "14"))
                except Exception:
                    dte_days = 14.0
                dte_days = max(1.0, min(60.0, dte_days))

                # Time to expiration at entry (years)
                T0 = dte_days / 365.0
                # Time to expiration at exit (years) - reduced by horizon minutes
                T1 = max(0.0, (dte_days - h / (60.0 * 24.0)) / 365.0)

                # Risk-free rate (approximate)
                r = float(os.environ.get("GHOST_RISK_FREE_RATE", "0.04"))

                # Strike prices: Use 0.5-1% OTM for realistic ATM-ish options
                # Override with GHOST_STRIKE_OFFSET_PCT env var
                try:
                    strike_offset_pct = float(os.environ.get("GHOST_STRIKE_OFFSET_PCT", "0.005"))
                except Exception:
                    strike_offset_pct = 0.005

                K_call = S0 * (1.0 + strike_offset_pct)  # Slightly OTM call
                K_put = S0 * (1.0 - strike_offset_pct)   # Slightly OTM put

                # Entry prices
                try:
                    call_entry = _black_scholes_call(S0, K_call, T0, r, sigma) * 100.0
                    put_entry = _black_scholes_put(S0, K_put, T0, r, sigma) * 100.0

                    # Exit prices (future underlying price, less time to expiration)
                    call_exit = _black_scholes_call(S1, K_call, T1, r, sigma) * 100.0
                    put_exit = _black_scholes_put(S1, K_put, T1, r, sigma) * 100.0

                    # Gross P&L (per contract = 100 shares)
                    gross_call = call_exit - call_entry
                    gross_put = put_exit - put_entry

                    # Use entry price for friction calculation (more accurate)
                    friction_dollars_call = call_entry * friction_pct + fees_rt
                    friction_dollars_put = put_entry * friction_pct + fees_rt

                    call_reward = gross_call - friction_dollars_call
                    put_reward = gross_put - friction_dollars_put

                    contract_cost = (call_entry + put_entry) / 2.0  # Average for details

                except Exception as e:
                    # Fallback to delta approximation if Black-Scholes fails
                    theta = self._theta_cost_pct(h)
                    theta_dollars = contract_cost * theta
                    underlying_change = _safe_float(current_price, 0.0) * uret
                    gross_call = delta * underlying_change * 100.0
                    gross_put = -delta * underlying_change * 100.0
                    call_reward = gross_call - theta_dollars - friction_dollars
                    put_reward = gross_put - theta_dollars - friction_dollars
            else:
                # Original delta-$ approximation for gross PnL
                theta = self._theta_cost_pct(h)
                theta_dollars = contract_cost * theta
                underlying_change = _safe_float(current_price, 0.0) * uret
                gross_call = delta * underlying_change * 100.0
                gross_put = -delta * underlying_change * 100.0
                call_reward = gross_call - theta_dollars - friction_dollars
                put_reward = gross_put - theta_dollars - friction_dollars

            details["horizons"][str(h)] = {
                "underlying_return": uret,
                "call_reward": call_reward,
                "put_reward": put_reward,
                "pricing_method": "black_scholes" if use_bs else "delta_approx",
            }

            if call_reward > best_call:
                best_call = call_reward
                best_call_h = h
            if put_reward > best_put:
                best_put = put_reward
                best_put_h = h

        ghost_best_action = "HOLD"
        ghost_best_reward = 0.0
        if best_call > ghost_best_reward and best_call > best_put:
            ghost_best_action = "BUY_CALLS"
            ghost_best_reward = float(best_call)
        elif best_put > ghost_best_reward and best_put > best_call:
            ghost_best_action = "BUY_PUTS"
            ghost_best_reward = float(best_put)

        details["best_call_horizon"] = best_call_h
        details["best_put_horizon"] = best_put_h

        return {
            "ghost_reward_call": float(best_call),
            "ghost_reward_put": float(best_put),
            "ghost_best_action": ghost_best_action,
            "ghost_best_reward": float(ghost_best_reward),
            "ghost_details": details,
        }


class DecisionPipeline:
    """
    Sidecar pipeline: write DecisionRecord and, for HOLD, MissedOpportunityRecord.

    Safe-by-default:
    - All methods catch and swallow exceptions at call sites (recommended).
    - Only append-only writes.
    """

    def __init__(
        self,
        *,
        decisions_path: Optional[str] = None,
        missed_path: Optional[str] = None,
        aux_dataset_path: Optional[str] = None,
    ):
        # Default paths are anchored to the output3 package root (stable regardless of cwd).
        base_dir = Path(__file__).resolve().parents[1]
        default_decisions = str(base_dir / "data" / "decision_records.jsonl")
        default_missed = str(base_dir / "data" / "missed_opportunities_full.jsonl")

        self.decisions_path = decisions_path or os.environ.get("DECISION_RECORDS_PATH", default_decisions)
        self.missed_path = missed_path or os.environ.get("MISSED_OPPS_PATH", default_missed)
        self.aux_dataset_path = aux_dataset_path or os.environ.get("MISSED_AUX_DATASET_PATH", "")

        self._decisions = JsonlWriter(self.decisions_path)
        self._missed = JsonlWriter(self.missed_path)
        self._aux = JsonlWriter(self.aux_dataset_path) if self.aux_dataset_path else None

        self.ghost = GhostTradeEvaluator()

    def record_decision(
        self,
        *,
        timestamp: datetime,
        symbol: str,
        current_price: float,
        signal: Mapping[str, Any],
        proposed_action: str,
        executed_action: str,
        trade_placed: bool,
        rejection_reasons: Sequence[str],
        paper_rejection: Optional[Dict[str, Any]] = None,
        bot: Optional[Any] = None,
        write_missed: bool = True,
    ) -> None:
        state = build_trade_state(
            timestamp=timestamp,
            symbol=symbol,
            current_price=current_price,
            signal=signal,
            bot=bot,
            rejection_reasons=rejection_reasons,
        )
        vec, names, schema = encode_state(state)

        rec = DecisionRecord(
            timestamp=state.timestamp,
            symbol=state.symbol,
            current_price=state.current_price,
            proposed_action=str(proposed_action),
            executed_action=str(executed_action),
            trade_placed=bool(trade_placed),
            raw_confidence=state.raw_confidence,
            calibrated_confidence=state.calibrated_confidence,
            rejection_reasons=list(rejection_reasons or []),
            paper_rejection=paper_rejection,
            schema_version=schema,
            feature_names=names,
            feature_vector=vec.astype(float).tolist(),
            trade_state=asdict(state),
        )
        self._decisions.append(rec.to_json())

        # Missed opportunity logging (HOLD only)
        if write_missed and str(executed_action).upper() == "HOLD":
            self._record_missed_from_hold(state, vec, names, schema, rejection_reasons)

        # Return useful intermediates (ignored by existing callers)
        return state, vec, names, schema

    def _record_missed_from_hold(
        self,
        state: FullTradeState,
        vec: np.ndarray,
        names: List[str],
        schema: str,
        rejection_reasons: Sequence[str],
    ) -> None:
        # Map predicted returns by horizon from available multi-TF predictions
        horizon_map: Dict[int, float] = {}
        tf_to_h = {
            # common keys produced by the bot
            "10min": 10,
            "15min": 15,
            "20min": 20,
            "30min": 30,
            "1h": 60,
            "1hour": 60,
            # optional (if evaluator includes it)
            "2hour": 120,
        }
        for tf, h in tf_to_h.items():
            if tf in state.predicted_return_by_tf:
                horizon_map[h] = _safe_float(state.predicted_return_by_tf.get(tf, 0.0), 0.0)
        # Fallback: use consensus predicted_return for missing horizons
        for h in (10, 15, 20, 30, 60):
            horizon_map.setdefault(h, _safe_float(state.predicted_return, 0.0))

        ghost = self.ghost.evaluate(
            timestamp=state.timestamp,
            symbol=state.symbol,
            current_price=state.current_price,
            predicted_returns_by_horizon=horizon_map,
            vix_level=state.vix_level,
            liquidity_proxy=state.liquidity_proxy,
        )

        miss = MissedOpportunityRecord(
            timestamp=state.timestamp,
            symbol=state.symbol,
            current_price=state.current_price,
            executed_action="HOLD",
            veto_reasons=list(rejection_reasons or []),
            ghost_mode="expected",
            ghost_reward_call=_safe_float(ghost.get("ghost_reward_call", 0.0), 0.0),
            ghost_reward_put=_safe_float(ghost.get("ghost_reward_put", 0.0), 0.0),
            ghost_best_action=str(ghost.get("ghost_best_action", "HOLD")),
            ghost_best_reward=_safe_float(ghost.get("ghost_best_reward", 0.0), 0.0),
            ghost_details=dict(ghost.get("ghost_details", {}) or {}),
            schema_version=schema,
            feature_names=names,
            feature_vector=vec.astype(float).tolist(),
        )
        self._missed.append(miss.to_json())

        # Optional auxiliary dataset writer for missed opportunities
        if self._aux is not None:
            row = {
                "schema_version": schema,
                "feature_names": names,
                "feature_vector": vec.astype(float).tolist(),
                "best_action": miss.ghost_best_action,
                "best_reward": miss.ghost_best_reward,
                "timestamp": miss.timestamp,
                "symbol": miss.symbol,
            }
            self._aux.append(json.dumps(row, ensure_ascii=False))

    def record_missed_opportunity(
        self,
        *,
        state: FullTradeState,
        vec: np.ndarray,
        names: List[str],
        schema: str,
        veto_reasons: Sequence[str],
        horizon_returns: Mapping[int, float],
        ghost_mode: str = "realized",
        executed_action: str = "HOLD",
    ) -> None:
        """
        Record a missed opportunity row with caller-provided horizon returns.

        This is used by time-travel / simulation when future prices are known and we want
        *realized* ghost labels, rather than "expected" labels derived from model predictions.
        """
        ghost = self.ghost.evaluate(
            timestamp=state.timestamp,
            symbol=state.symbol,
            current_price=state.current_price,
            predicted_returns_by_horizon=horizon_returns,
            vix_level=state.vix_level,
            liquidity_proxy=state.liquidity_proxy,
        )
        miss = MissedOpportunityRecord(
            timestamp=state.timestamp,
            symbol=state.symbol,
            current_price=state.current_price,
            executed_action=str(executed_action or "HOLD"),
            veto_reasons=list(veto_reasons or []),
            ghost_mode=str(ghost_mode or "realized"),
            ghost_reward_call=_safe_float(ghost.get("ghost_reward_call", 0.0), 0.0),
            ghost_reward_put=_safe_float(ghost.get("ghost_reward_put", 0.0), 0.0),
            ghost_best_action=str(ghost.get("ghost_best_action", "HOLD")),
            ghost_best_reward=_safe_float(ghost.get("ghost_best_reward", 0.0), 0.0),
            ghost_details=dict(ghost.get("ghost_details", {}) or {}),
            schema_version=schema,
            feature_names=names,
            feature_vector=vec.astype(float).tolist(),
        )
        self._missed.append(miss.to_json())





