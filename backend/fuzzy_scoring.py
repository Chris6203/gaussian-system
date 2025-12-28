#!/usr/bin/env python3
"""
Fuzzy Logic Confidence Scoring

Jerry's "Fuzzy Logic Inference Engine" feature:
- Replace binary thresholds with membership functions μ ∈ [0,1]
- Compute weighted aggregation F_t = Σ w_j × μ_j
- Use F_t as trade quality filter

From Jerry's PDF:
    "Membership degrees μ_j ∈ [0,1] for each soft constraint,
     weighted aggregation F_t = Σ w_j μ_j."

Membership Functions:
- μ_iv: IV Favorable (triangular around optimal IV range)
- μ_regime: Regime Stability (from HMM confidence)
- μ_direction: Direction Agreement (predicted vs HMM trend)
- μ_confidence: Model Confidence (neural network output)

Usage:
    from backend.fuzzy_scoring import compute_fuzzy_score, FuzzyScorer

    score = compute_fuzzy_score(vix=18, hmm_conf=0.8, direction_agree=True, nn_conf=0.65)
    if score > 0.6:
        # High quality trade setup
        pass
"""

import os
import json
import logging
import math
from typing import Dict, Optional, Tuple
from pathlib import Path

logger = logging.getLogger(__name__)


def triangular_membership(x: float, a: float, b: float, c: float) -> float:
    """
    Triangular membership function.

    Args:
        x: Input value
        a: Left foot (μ=0)
        b: Peak (μ=1)
        c: Right foot (μ=0)

    Returns:
        Membership degree μ ∈ [0,1]
    """
    if x <= a or x >= c:
        return 0.0
    elif x < b:
        return (x - a) / (b - a)
    else:
        return (c - x) / (c - b)


def trapezoidal_membership(x: float, a: float, b: float, c: float, d: float) -> float:
    """
    Trapezoidal membership function.

    Args:
        x: Input value
        a: Left foot (μ=0)
        b: Left shoulder (μ=1)
        c: Right shoulder (μ=1)
        d: Right foot (μ=0)

    Returns:
        Membership degree μ ∈ [0,1]
    """
    if x <= a or x >= d:
        return 0.0
    elif x < b:
        return (x - a) / (b - a)
    elif x <= c:
        return 1.0
    else:
        return (d - x) / (d - c)


def sigmoid_membership(x: float, center: float, steepness: float = 10.0) -> float:
    """
    Sigmoid membership function (S-shaped).

    Args:
        x: Input value
        center: Center point where μ=0.5
        steepness: How steep the transition is

    Returns:
        Membership degree μ ∈ [0,1]
    """
    return 1.0 / (1.0 + math.exp(-steepness * (x - center)))


class FuzzyScorer:
    """
    Computes fuzzy trade quality scores using membership functions.

    Jerry's Fuzzy Logic Engine:
    - Each factor has a membership function μ ∈ [0,1]
    - Weighted aggregation: F_t = Σ w_j × μ_j
    - High F_t = high-quality trade setup
    """

    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize fuzzy scorer.

        Config options (jerry.fuzzy_logic section):
            enabled: bool - Enable/disable fuzzy scoring
            threshold: float - Minimum F_t to allow trade
            weights: Dict[str, float] - Weight for each membership function
        """
        self.config = config or {}

        # Load from jerry config section
        jerry_cfg = self.config.get('jerry', {}).get('fuzzy_logic', {})

        self.enabled = jerry_cfg.get('enabled', False)
        self.threshold = jerry_cfg.get('threshold', 0.6)

        # Default weights sum to 1.0
        default_weights = {
            'iv_favorable': 0.25,
            'regime_stability': 0.30,
            'direction_agreement': 0.25,
            'model_confidence': 0.20
        }
        self.weights = jerry_cfg.get('weights', default_weights)

        # Membership function parameters (tuned for SPY options)
        self.iv_optimal_low = 14.0    # VIX below this is too low (no premium)
        self.iv_optimal_peak = 18.0   # VIX around here is ideal
        self.iv_optimal_high = 25.0   # VIX above this is getting risky
        self.iv_danger = 35.0         # VIX above this = danger zone

        logger.info(f"Fuzzy Scorer initialized (enabled={self.enabled})")
        if self.enabled:
            logger.info(f"   Threshold: {self.threshold}")
            logger.info(f"   Weights: {self.weights}")

    def compute_iv_favorable(self, vix: float) -> float:
        """
        Compute IV favorable membership.

        Optimal VIX range: 14-25 (enough premium, not too volatile)
        Peak at VIX=18 (ideal conditions)
        """
        if vix <= self.iv_optimal_low:
            # Too low volatility - no premium
            return vix / self.iv_optimal_low
        elif vix <= self.iv_optimal_peak:
            # Rising toward optimal
            return triangular_membership(
                vix,
                self.iv_optimal_low - 2,
                self.iv_optimal_peak,
                self.iv_optimal_high
            )
        elif vix <= self.iv_optimal_high:
            # Good range but past peak
            return trapezoidal_membership(
                vix,
                self.iv_optimal_low,
                self.iv_optimal_low + 2,
                self.iv_optimal_peak + 2,
                self.iv_optimal_high + 5
            )
        elif vix <= self.iv_danger:
            # Elevated volatility - risky
            return (self.iv_danger - vix) / (self.iv_danger - self.iv_optimal_high)
        else:
            # Danger zone
            return 0.0

    def compute_regime_stability(self, hmm_confidence: float) -> float:
        """
        Compute regime stability membership from HMM confidence.

        High HMM confidence = stable regime = good for trading
        """
        # Sigmoid: rises from 0.5 to 1.0 as confidence goes from 0.5 to 0.9
        return sigmoid_membership(hmm_confidence, center=0.6, steepness=8.0)

    def compute_direction_agreement(
        self,
        predicted_direction: float,
        hmm_trend: float,
        proposed_action: str
    ) -> float:
        """
        Compute direction agreement membership.

        All signals should agree:
        - Neural network prediction
        - HMM trend
        - Proposed trade direction

        Returns 1.0 if all agree, 0.0 if conflicting
        """
        # Determine expected direction from proposed action
        if 'CALL' in proposed_action.upper():
            action_direction = 1.0
        elif 'PUT' in proposed_action.upper():
            action_direction = -1.0
        else:
            return 1.0  # HOLD - no direction requirement

        # Normalize HMM trend to direction (-1 to +1)
        hmm_direction = (hmm_trend - 0.5) * 2  # 0.5 -> 0, 1 -> 1, 0 -> -1

        # Check agreement
        nn_agrees = (predicted_direction * action_direction) > 0
        hmm_agrees = (hmm_direction * action_direction) > 0

        if nn_agrees and hmm_agrees:
            # Full agreement
            return 1.0
        elif nn_agrees or hmm_agrees:
            # Partial agreement
            return 0.5
        else:
            # Full disagreement
            return 0.0

    def compute_model_confidence(self, confidence: float) -> float:
        """
        Compute model confidence membership.

        Uses sigmoid to smooth the confidence into a membership score.
        """
        # Sigmoid centered at 0.55 (our typical confidence threshold)
        return sigmoid_membership(confidence, center=0.55, steepness=12.0)

    def compute_score(
        self,
        vix: float = 18.0,
        hmm_confidence: float = 0.5,
        predicted_direction: float = 0.0,
        hmm_trend: float = 0.5,
        model_confidence: float = 0.5,
        proposed_action: str = 'HOLD'
    ) -> Tuple[float, Dict[str, float]]:
        """
        Compute the fuzzy trade quality score.

        Args:
            vix: Current VIX level
            hmm_confidence: HMM model's confidence in regime
            predicted_direction: Neural network predicted direction (-1 to +1)
            hmm_trend: HMM trend state (0=bearish, 0.5=neutral, 1=bullish)
            model_confidence: Neural network confidence (0 to 1)
            proposed_action: Proposed trade action (BUY_CALL, BUY_PUT, HOLD)

        Returns:
            (score, breakdown) - Overall score and individual membership values
        """
        if not self.enabled:
            return 1.0, {}  # Disabled = always pass

        # Compute individual membership values
        μ_iv = self.compute_iv_favorable(vix)
        μ_regime = self.compute_regime_stability(hmm_confidence)
        μ_direction = self.compute_direction_agreement(
            predicted_direction, hmm_trend, proposed_action
        )
        μ_confidence = self.compute_model_confidence(model_confidence)

        # Weighted aggregation: F_t = Σ w_j × μ_j
        score = (
            self.weights.get('iv_favorable', 0.25) * μ_iv +
            self.weights.get('regime_stability', 0.30) * μ_regime +
            self.weights.get('direction_agreement', 0.25) * μ_direction +
            self.weights.get('model_confidence', 0.20) * μ_confidence
        )

        breakdown = {
            'μ_iv': μ_iv,
            'μ_regime': μ_regime,
            'μ_direction': μ_direction,
            'μ_confidence': μ_confidence,
            'F_t': score
        }

        return score, breakdown

    def should_allow_trade(
        self,
        vix: float = 18.0,
        hmm_confidence: float = 0.5,
        predicted_direction: float = 0.0,
        hmm_trend: float = 0.5,
        model_confidence: float = 0.5,
        proposed_action: str = 'HOLD'
    ) -> Tuple[bool, float, Dict]:
        """
        Check if trade should be allowed based on fuzzy score.

        Returns:
            (allowed, score, breakdown)
        """
        score, breakdown = self.compute_score(
            vix=vix,
            hmm_confidence=hmm_confidence,
            predicted_direction=predicted_direction,
            hmm_trend=hmm_trend,
            model_confidence=model_confidence,
            proposed_action=proposed_action
        )

        allowed = not self.enabled or score >= self.threshold

        if self.enabled and not allowed:
            logger.info(
                f"FUZZY FILTER: Score {score:.3f} < threshold {self.threshold} "
                f"(μ_iv={breakdown['μ_iv']:.2f}, μ_regime={breakdown['μ_regime']:.2f}, "
                f"μ_dir={breakdown['μ_direction']:.2f}, μ_conf={breakdown['μ_confidence']:.2f})"
            )

        return allowed, score, breakdown


# =============================================================================
# Global Scorer Instance
# =============================================================================

_scorer_instance: Optional[FuzzyScorer] = None


def get_fuzzy_scorer(config: Optional[Dict] = None) -> FuzzyScorer:
    """Get or create the global fuzzy scorer instance."""
    global _scorer_instance

    if _scorer_instance is None:
        # Try to load config from file
        if config is None:
            config_path = Path('config.json')
            if config_path.exists():
                try:
                    with open(config_path) as f:
                        config = json.load(f)
                except Exception as e:
                    logger.warning(f"Could not load config.json: {e}")
                    config = {}

        _scorer_instance = FuzzyScorer(config)

    return _scorer_instance


def compute_fuzzy_score(
    vix: float = 18.0,
    hmm_confidence: float = 0.5,
    predicted_direction: float = 0.0,
    hmm_trend: float = 0.5,
    model_confidence: float = 0.5,
    proposed_action: str = 'HOLD'
) -> Tuple[float, Dict]:
    """
    Quick function to compute fuzzy score.

    Returns:
        (score, breakdown)
    """
    scorer = get_fuzzy_scorer()
    return scorer.compute_score(
        vix=vix,
        hmm_confidence=hmm_confidence,
        predicted_direction=predicted_direction,
        hmm_trend=hmm_trend,
        model_confidence=model_confidence,
        proposed_action=proposed_action
    )


def should_allow_trade(
    vix: float = 18.0,
    hmm_confidence: float = 0.5,
    predicted_direction: float = 0.0,
    hmm_trend: float = 0.5,
    model_confidence: float = 0.5,
    proposed_action: str = 'HOLD'
) -> Tuple[bool, float, Dict]:
    """
    Check if trade should be allowed based on fuzzy score.

    Usage:
        allowed, score, breakdown = should_allow_trade(vix=20, ...)
        if not allowed:
            return HOLD
    """
    scorer = get_fuzzy_scorer()
    return scorer.should_allow_trade(
        vix=vix,
        hmm_confidence=hmm_confidence,
        predicted_direction=predicted_direction,
        hmm_trend=hmm_trend,
        model_confidence=model_confidence,
        proposed_action=proposed_action
    )


def reset_scorer():
    """Reset the global scorer instance (for testing)."""
    global _scorer_instance
    _scorer_instance = None
