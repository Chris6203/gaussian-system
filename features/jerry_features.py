#!/usr/bin/env python3
"""
Jerry's Fuzzy Logic Features

Based on Jerry's Quantor-MTFuzz framework. Computes fuzzy membership scores
that can be used as:
1. Additional input features to neural networks
2. Confirmation filters for trade decisions

All outputs are normalized to [0,1] range for easy interpretation.

Reference: docs/jerry-info/EQUATIONS.md
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, List, Tuple
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class JerryFeatures:
    """Container for Jerry's fuzzy membership scores."""
    mu_iv: float = 0.5          # IV Favorability [0,1]
    mu_regime: float = 0.5      # Regime Stability [0,1]
    mu_mtf: float = 0.5         # Multi-Timeframe Alignment [0,1]
    mu_liq: float = 0.5         # Liquidity Quality [0,1]
    mu_delta: float = 0.5       # Delta Balance [0,1]
    f_t: float = 0.5            # Composite Fuzzy Score [0,1]

    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary for feature pipeline."""
        return {
            'jerry_mu_iv': self.mu_iv,
            'jerry_mu_regime': self.mu_regime,
            'jerry_mu_mtf': self.mu_mtf,
            'jerry_mu_liq': self.mu_liq,
            'jerry_mu_delta': self.mu_delta,
            'jerry_f_t': self.f_t,
        }


# Default weights for composite score (from Jerry's framework)
DEFAULT_WEIGHTS = {
    'iv': 0.20,
    'regime': 0.25,
    'mtf': 0.35,
    'liq': 0.15,
    'delta': 0.05,
}


def compute_mu_iv(
    current_iv: float,
    iv_percentile: Optional[float] = None,
    iv_history: Optional[np.ndarray] = None
) -> float:
    """
    Compute IV Favorability membership.

    For BUYING options: low IV is favorable (cheaper premiums).
    For SELLING options: high IV is favorable (more premium).

    We assume buying (directional trades), so low IV = high score.

    Args:
        current_iv: Current implied volatility (VIX or ATM IV)
        iv_percentile: IV rank/percentile [0-100] if available
        iv_history: Historical IV values for computing percentile

    Returns:
        μ_IV ∈ [0,1] where 1 = very favorable
    """
    if iv_percentile is not None:
        # Direct percentile: invert so low IV = high score
        return 1.0 - (iv_percentile / 100.0)

    if iv_history is not None and len(iv_history) >= 20:
        # Compute percentile from history
        percentile = np.sum(iv_history < current_iv) / len(iv_history)
        return 1.0 - percentile

    # Fallback: use VIX heuristic
    # VIX 10-15: very low, VIX 20-25: normal, VIX 30+: high
    if current_iv < 12:
        return 1.0
    elif current_iv < 16:
        return 0.9
    elif current_iv < 20:
        return 0.7
    elif current_iv < 25:
        return 0.5
    elif current_iv < 30:
        return 0.3
    else:
        return 0.1


def compute_mu_regime(
    bars_in_regime: int,
    regime_changes_last_n: int = 0,
    lookback_bars: int = 100
) -> float:
    """
    Compute Regime Stability membership.

    Stable regimes = more predictable = higher score.
    Frequent regime changes = choppy market = lower score.

    Args:
        bars_in_regime: Number of bars in current regime
        regime_changes_last_n: Number of regime changes in lookback
        lookback_bars: Lookback period for change counting

    Returns:
        μ_R ∈ [0,1] where 1 = very stable
    """
    # Score based on time in regime (max out at 30 bars = 30 min)
    time_score = min(1.0, bars_in_regime / 30.0)

    # Penalty for frequent changes
    if lookback_bars > 0 and regime_changes_last_n > 0:
        change_rate = regime_changes_last_n / lookback_bars
        stability_score = max(0.0, 1.0 - change_rate * 10)  # 10% changes = 0 score
    else:
        stability_score = 1.0

    # Combine: both must be good
    return time_score * 0.6 + stability_score * 0.4


def compute_mu_mtf(
    predictions: Dict[str, float],
    weights: Optional[Dict[str, float]] = None
) -> float:
    """
    Compute Multi-Timeframe Alignment membership.

    Based on Jerry's MTF consensus with longer TF = more weight.

    Args:
        predictions: Dict of horizon -> direction prediction
            e.g., {'5m': 0.3, '15m': 0.5, '30m': 0.4, '45m': 0.6}
            Direction should be in [-1, 1] range
        weights: Optional custom weights (default: Jerry's weights)

    Returns:
        μ_MTF ∈ [0,1] where 1 = strong agreement
    """
    if not predictions:
        return 0.5

    # Default weights: longer horizons weighted more
    if weights is None:
        weights = {
            '5m': 0.10,
            '15m': 0.20,
            '30m': 0.30,
            '45m': 0.40,
        }

    # Compute weighted consensus
    weighted_sum = 0.0
    total_weight = 0.0

    for horizon, direction in predictions.items():
        w = weights.get(horizon, 0.25)  # Default equal weight if not specified
        weighted_sum += w * direction
        total_weight += w

    if total_weight == 0:
        return 0.5

    consensus = weighted_sum / total_weight

    # Convert to alignment score:
    # Strong agreement (all same direction) = |consensus| close to 1 = high score
    # Disagreement (mixed directions) = |consensus| close to 0 = low score
    alignment = abs(consensus)

    return alignment


def compute_mu_liq(
    bid_ask_spread_pct: float = 0.0,
    volume_ratio: float = 1.0,
    open_interest: Optional[int] = None
) -> float:
    """
    Compute Liquidity Quality membership.

    Good liquidity = tight spreads + high volume = higher score.

    Args:
        bid_ask_spread_pct: Bid-ask spread as percentage of mid
        volume_ratio: Current volume / average volume
        open_interest: Option open interest (optional)

    Returns:
        μ_Liq ∈ [0,1] where 1 = excellent liquidity
    """
    # Spread score: tighter = better
    # 0.5% spread = 1.0, 5% spread = 0.0
    spread_score = max(0.0, 1.0 - bid_ask_spread_pct / 5.0)

    # Volume score: higher than average = better
    # Cap at 2.0 (200% of average)
    volume_score = min(1.0, volume_ratio / 2.0)

    # Open interest score (if available)
    if open_interest is not None:
        # 1000+ OI = good, 100 = marginal, <100 = poor
        oi_score = min(1.0, open_interest / 1000.0)
    else:
        oi_score = 0.5  # Neutral if not available

    # Combine: spread is most important for execution
    return spread_score * 0.5 + volume_score * 0.3 + oi_score * 0.2


def compute_mu_delta(
    portfolio_delta: float = 0.0,
    delta_limit: float = 50.0
) -> float:
    """
    Compute Delta Balance membership.

    Jerry's system targets delta-neutral. For directional trading,
    we want some delta exposure but not extreme.

    Args:
        portfolio_delta: Total portfolio delta
        delta_limit: Maximum acceptable delta

    Returns:
        μ_Δ ∈ [0,1] where 1 = well balanced
    """
    # For directional trading, some delta is expected
    # Penalize only extreme positions
    delta_ratio = abs(portfolio_delta) / delta_limit

    if delta_ratio <= 0.5:
        return 1.0  # Comfortable
    elif delta_ratio <= 1.0:
        return 1.0 - (delta_ratio - 0.5)  # Linear decay
    else:
        return max(0.0, 0.5 - (delta_ratio - 1.0) * 0.5)  # Steep penalty


def compute_composite_score(
    mu_iv: float,
    mu_regime: float,
    mu_mtf: float,
    mu_liq: float,
    mu_delta: float = 0.5,
    weights: Optional[Dict[str, float]] = None
) -> float:
    """
    Compute composite fuzzy score F_t.

    F_t = Σ w_j × μ_j

    Args:
        mu_*: Individual membership scores
        weights: Custom weights (default: Jerry's weights)

    Returns:
        F_t ∈ [0,1] where 1 = all conditions favorable
    """
    w = weights or DEFAULT_WEIGHTS

    f_t = (
        w['iv'] * mu_iv +
        w['regime'] * mu_regime +
        w['mtf'] * mu_mtf +
        w['liq'] * mu_liq +
        w['delta'] * mu_delta
    )

    return np.clip(f_t, 0.0, 1.0)


def compute_jerry_features(
    vix_level: float = 18.0,
    iv_percentile: Optional[float] = None,
    bars_in_regime: int = 10,
    regime_changes: int = 0,
    multi_horizon_predictions: Optional[Dict[str, float]] = None,
    bid_ask_spread_pct: float = 1.0,
    volume_ratio: float = 1.0,
    portfolio_delta: float = 0.0,
    hmm_confidence: float = 0.5,
) -> JerryFeatures:
    """
    Compute all Jerry's features from available inputs.

    This is the main entry point for the feature pipeline.

    Args:
        vix_level: Current VIX level
        iv_percentile: IV rank [0-100] if available
        bars_in_regime: Bars in current HMM regime
        regime_changes: Regime changes in last 100 bars
        multi_horizon_predictions: V3 predictor outputs
        bid_ask_spread_pct: Bid-ask spread percentage
        volume_ratio: Current/average volume
        portfolio_delta: Total delta exposure
        hmm_confidence: HMM model confidence

    Returns:
        JerryFeatures with all membership scores
    """
    # Compute individual memberships
    mu_iv = compute_mu_iv(vix_level, iv_percentile)

    mu_regime = compute_mu_regime(
        bars_in_regime=bars_in_regime,
        regime_changes_last_n=regime_changes
    )

    # MTF alignment
    if multi_horizon_predictions:
        mu_mtf = compute_mu_mtf(multi_horizon_predictions)
    else:
        # Use HMM confidence as proxy
        mu_mtf = hmm_confidence

    mu_liq = compute_mu_liq(
        bid_ask_spread_pct=bid_ask_spread_pct,
        volume_ratio=volume_ratio
    )

    mu_delta = compute_mu_delta(portfolio_delta)

    # Composite score
    f_t = compute_composite_score(
        mu_iv=mu_iv,
        mu_regime=mu_regime,
        mu_mtf=mu_mtf,
        mu_liq=mu_liq,
        mu_delta=mu_delta
    )

    return JerryFeatures(
        mu_iv=mu_iv,
        mu_regime=mu_regime,
        mu_mtf=mu_mtf,
        mu_liq=mu_liq,
        mu_delta=mu_delta,
        f_t=f_t
    )


def jerry_filter_check(
    f_t: float,
    threshold: float = 0.5,
    min_mtf: float = 0.3,
    mu_mtf: float = 0.5
) -> Tuple[bool, str]:
    """
    Check if trade passes Jerry's filter criteria.

    Args:
        f_t: Composite fuzzy score
        threshold: Minimum F_t to pass
        min_mtf: Minimum MTF alignment
        mu_mtf: Current MTF alignment score

    Returns:
        (pass, reason) tuple
    """
    if f_t < threshold:
        return False, f"F_t {f_t:.2f} < threshold {threshold}"

    if mu_mtf < min_mtf:
        return False, f"MTF alignment {mu_mtf:.2f} < min {min_mtf}"

    return True, "All Jerry checks passed"


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def get_jerry_features_from_context(context: Dict) -> JerryFeatures:
    """
    Extract Jerry features from a trading context dictionary.

    Useful for integration with existing signal flow.
    """
    return compute_jerry_features(
        vix_level=context.get('vix_level', 18.0),
        iv_percentile=context.get('iv_percentile'),
        bars_in_regime=context.get('bars_in_regime', 10),
        regime_changes=context.get('regime_changes', 0),
        multi_horizon_predictions=context.get('multi_horizon_predictions'),
        bid_ask_spread_pct=context.get('bid_ask_spread_pct', 1.0),
        volume_ratio=context.get('volume_ratio', 1.0),
        portfolio_delta=context.get('portfolio_delta', 0.0),
        hmm_confidence=context.get('hmm_confidence', 0.5),
    )


if __name__ == "__main__":
    """Test Jerry's features."""
    print("=" * 60)
    print("JERRY FEATURES TEST")
    print("=" * 60)

    # Test with sample data
    features = compute_jerry_features(
        vix_level=22.0,
        iv_percentile=65.0,
        bars_in_regime=25,
        regime_changes=2,
        multi_horizon_predictions={
            '5m': 0.3,
            '15m': 0.4,
            '30m': 0.5,
            '45m': 0.6
        },
        bid_ask_spread_pct=0.8,
        volume_ratio=1.5,
        portfolio_delta=20.0,
        hmm_confidence=0.75
    )

    print("\nComputed Features:")
    for name, value in features.to_dict().items():
        print(f"  {name}: {value:.3f}")

    # Test filter
    passed, reason = jerry_filter_check(
        f_t=features.f_t,
        threshold=0.5,
        min_mtf=0.3,
        mu_mtf=features.mu_mtf
    )
    print(f"\nFilter Check: {'PASS' if passed else 'FAIL'}")
    print(f"Reason: {reason}")

    print("\n" + "=" * 60)
