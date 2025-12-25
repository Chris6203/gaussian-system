#!/usr/bin/env python3
"""
Edge-based Entry Controller
==========================

Goal: improve entries by using the predictor as a world model:
- predicted return/vol/confidence
- execution heads: fillability / exp_slippage / exp_ttf
- liquidity/spread proxies + theta decay proxy

This controller is intended to replace "pure RL picks entries" while leaving exits unchanged.
It does not place orders; it only chooses action (BUY_CALLS / BUY_PUTS / HOLD) and suggested size.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Tuple


def _sf(x: Any, default: float = 0.0) -> float:
    try:
        if x is None:
            return float(default)
        return float(x)
    except Exception:
        return float(default)


def _clamp(x: float, lo: float, hi: float) -> float:
    return float(max(lo, min(hi, x)))


@dataclass
class EdgeDecision:
    action: str  # BUY_CALLS | BUY_PUTS | HOLD
    position_size: int
    reason: str
    edge_call: float
    edge_put: float
    best_edge: float


class EdgeEntryController:
    """
    Simple, deterministic scorer for CALL/PUT/HOLD.

    Scores are approximated expected dollars per 1 contract after friction, using:
    - option premium proxy (scaled by VIX/volatility percentile)
    - delta leverage approximation
    - theta decay proxy
    - execution heads (fillability / slippage / time-to-fill)
    - spread/liquidity proxies (if provided)
    """

    def __init__(
        self,
        *,
        min_confidence: float = 0.55,
        min_edge_dollars: float = 1.0,
        delta_assumed: float = 0.5,
        base_premium_pct: float = 0.02,
        base_spread_rt_pct: float = 0.04,
        fee_roundtrip: float = 1.46,  # ~2 * 0.73 (Tradier-ish)
        ttf_penalty_per_second: float = 0.002,  # dollars penalty per second expected TTF
    ):
        self.min_confidence = float(min_confidence)
        self.min_edge_dollars = float(min_edge_dollars)
        self.delta_assumed = float(_clamp(delta_assumed, 0.1, 0.9))
        self.base_premium_pct = float(_clamp(base_premium_pct, 0.005, 0.06))
        self.base_spread_rt_pct = float(_clamp(base_spread_rt_pct, 0.0, 0.20))
        self.fee_roundtrip = float(max(0.0, fee_roundtrip))
        self.ttf_penalty_per_second = float(max(0.0, ttf_penalty_per_second))

    def decide(
        self,
        *,
        current_price: float,
        predicted_return: float,
        predicted_volatility: float,
        confidence: float,
        fillability: float,
        exp_slippage: float,
        exp_ttf: float,
        volume_spike: float = 1.0,
        momentum_5m: float = 0.0,
        momentum_15m: float = 0.0,
        vix_level: float = 18.0,
        volatility_percentile: float = 50.0,
        liquidity_proxy: float = 0.5,
        spread_proxy: float = 0.5,
        minutes_horizon: int = 15,
    ) -> EdgeDecision:
        px = max(0.01, _sf(current_price, 0.0))
        conf = _clamp(_sf(confidence, 0.0), 0.0, 1.0)
        p_fill = _clamp(_sf(fillability, 0.5), 0.0, 1.0)

        # Premium proxy (per share) scaled by volatility regime / VIX
        vol_pct = _clamp(_sf(volatility_percentile, 50.0), 0.0, 100.0)
        vix = max(8.0, _sf(vix_level, 18.0))

        premium_pct = self.base_premium_pct * (0.75 + (vol_pct / 100.0) * 0.75) * (0.9 + (vix - 18.0) / 80.0)
        premium_pct = _clamp(premium_pct, 0.008, 0.05)

        # Cost of 1 contract (approx)
        contract_cost = max(1.0, px * premium_pct * 100.0)

        # Theta proxy: higher vol -> slightly higher theta; scale to minutes_horizon
        # (This is intentionally conservative; paper_trader has its own exit/backstops.)
        theta_daily_pct = 0.02 + (vix / 100.0)  # ~0.20 to 0.35 in extreme; bounded below
        theta_cost_pct = _clamp(theta_daily_pct * (minutes_horizon / 390.0), 0.0, 0.15)

        # Spread/liquidity friction (round trip)
        liq = _clamp(_sf(liquidity_proxy, 0.5), 0.0, 1.0)
        spr = _clamp(_sf(spread_proxy, 0.5), 0.0, 1.0)
        spread_rt_pct = self.base_spread_rt_pct * (1.0 + (1.0 - liq) * 1.5) * (1.0 + spr * 0.75)
        spread_rt_pct = _clamp(spread_rt_pct, 0.0, 0.20)

        # Execution head friction
        slip = _sf(exp_slippage, 0.0)
        # Assume slippage head is dollars per contract (can be negative); take abs as cost.
        slippage_cost = abs(slip)
        ttf_cost = self.ttf_penalty_per_second * max(0.0, _sf(exp_ttf, 0.0))

        friction_dollars = contract_cost * spread_rt_pct + self.fee_roundtrip + slippage_cost + ttf_cost

        # Expected option PnL proxy (call vs put): underlying return scaled by delta + theta
        uret = _sf(predicted_return, 0.0)
        call_pct = (uret / self.delta_assumed) - theta_cost_pct
        put_pct = (-uret / self.delta_assumed) - theta_cost_pct

        # Scale reward by fillability (world model says some trades just won't fill well)
        edge_call = p_fill * (contract_cost * call_pct) - friction_dollars
        edge_put = p_fill * (contract_cost * put_pct) - friction_dollars

        best_edge = max(edge_call, edge_put)
        best_action = "BUY_CALLS" if edge_call >= edge_put else "BUY_PUTS"

        # Quality gate: require confidence, and positive edge above a floor
        if conf < self.min_confidence:
            return EdgeDecision(
                action="HOLD",
                position_size=0,
                reason=f"edge_hold: low_conf ({conf:.2f} < {self.min_confidence:.2f})",
                edge_call=float(edge_call),
                edge_put=float(edge_put),
                best_edge=float(best_edge),
            )

        if best_edge < self.min_edge_dollars:
            return EdgeDecision(
                action="HOLD",
                position_size=0,
                reason=f"edge_hold: low_edge (${best_edge:+.2f} < ${self.min_edge_dollars:.2f})",
                edge_call=float(edge_call),
                edge_put=float(edge_put),
                best_edge=float(best_edge),
            )

        # Simple sizing: scale 1..5 contracts by edge strength and fillability
        size = 1
        # Edge per contract relative to cost
        edge_ratio = best_edge / max(1.0, contract_cost)
        size = int(_clamp(1.0 + edge_ratio * 10.0, 1.0, 5.0))
        if p_fill < 0.4:
            size = max(1, int(size * 0.5))

        return EdgeDecision(
            action=best_action,
            position_size=int(size),
            reason=f"edge_ok: {best_action} edge=${best_edge:+.2f} p_fill={p_fill:.2f} slip=${slippage_cost:.2f} ttf={_sf(exp_ttf,0):.0f}s",
            edge_call=float(edge_call),
            edge_put=float(edge_put),
            best_edge=float(best_edge),
        )


def decide_from_signal(
    *,
    signal: Mapping[str, Any],
    neural_pred: Optional[Mapping[str, Any]],
    options_metrics: Mapping[str, Any],
    bot: Optional[Any] = None,
    controller: Optional[EdgeEntryController] = None,
) -> Tuple[EdgeDecision, Dict[str, Any]]:
    """
    Convenience helper that extracts fields from existing bot structures.
    Returns (decision, signal_updates).
    """
    ctrl = controller or EdgeEntryController()

    npred = dict(neural_pred or {})
    sig = dict(signal or {})
    om = dict(options_metrics or {})

    vix_level = _sf(getattr(bot, "current_vix", None), _sf(sig.get("vix_level", 18.0), 18.0))
    volatility_percentile = _sf(om.get("volatility_percentile", 50.0), 50.0)

    decision = ctrl.decide(
        current_price=_sf(sig.get("current_price", 0.0), 0.0),
        predicted_return=_sf(npred.get("predicted_return", sig.get("predicted_return", 0.0)), 0.0),
        predicted_volatility=_sf(npred.get("predicted_volatility", 0.0), 0.0),
        confidence=_sf(sig.get("confidence", npred.get("neural_confidence", 0.0)), 0.0),
        fillability=_sf(npred.get("fillability_mean", npred.get("fillability", 0.5)), 0.5),
        exp_slippage=_sf(npred.get("exp_slippage_mean", npred.get("exp_slippage", 0.0)), 0.0),
        exp_ttf=_sf(npred.get("exp_ttf_mean", npred.get("exp_ttf", 0.0)), 0.0),
        volume_spike=_sf(sig.get("volume_spike", 1.0), 1.0),
        momentum_5m=_sf(sig.get("momentum_5min", sig.get("momentum_5m", 0.0)), 0.0),
        momentum_15m=_sf(sig.get("momentum_15min", sig.get("momentum_15m", 0.0)), 0.0),
        vix_level=vix_level,
        volatility_percentile=volatility_percentile,
        liquidity_proxy=_sf(getattr(bot, "current_liquidity_proxy", None), _sf(sig.get("liquidity_proxy", 0.5), 0.5)),
        spread_proxy=_sf(getattr(bot, "current_spread_proxy", None), _sf(sig.get("spread_proxy", 0.5), 0.5)),
        minutes_horizon=int(_sf(sig.get("timeframe_minutes", 15), 15)),
    )

    updates: Dict[str, Any] = {
        "edge_decision": {
            "edge_call": decision.edge_call,
            "edge_put": decision.edge_put,
            "best_edge": decision.best_edge,
            "reason": decision.reason,
        }
    }
    if decision.action != "HOLD":
        updates["action"] = decision.action
        updates["strategy"] = "EDGE_SCORER"
        updates["rl_position_size"] = decision.position_size
        updates.setdefault("reasoning", [])
        try:
            updates["reasoning"] = list(sig.get("reasoning", [])) + [decision.reason]
        except Exception:
            updates["reasoning"] = [decision.reason]
    return decision, updates





