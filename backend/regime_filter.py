#!/usr/bin/env python3
"""
Unified Regime Filter - Gates entries based on market regime quality

Combines three key checks:
1. Regime Quality Score - Composite metric for tradability
2. Regime Transition Detection - Identifies uncertain/transitioning periods
3. VIX-HMM Reconciliation - Ensures volatility signals agree

Usage:
    from backend.regime_filter import RegimeFilter

    filter = RegimeFilter()
    decision = filter.should_trade(
        hmm_trend=0.75,
        hmm_volatility=0.3,
        hmm_liquidity=0.5,
        hmm_confidence=0.8,
        vix_level=18.5,
        vix_percentile=0.45,
        previous_regime={'trend': 0.72, 'vol': 0.28}
    )

    if decision.can_trade:
        # Proceed with entry
        position_scale = decision.position_scale
    else:
        # Block entry
        print(f"Blocked: {decision.veto_reason}")
"""

import os
import logging
import numpy as np
from dataclasses import dataclass
from typing import Optional, Dict, Tuple
from collections import deque

logger = logging.getLogger(__name__)


@dataclass
class RegimeDecision:
    """Result of regime filter evaluation"""
    can_trade: bool
    quality_score: float  # 0-1, higher = better regime for trading
    position_scale: float  # 0-1, suggested position size multiplier
    regime_state: str  # 'trending', 'choppy', 'transitioning', 'extreme'
    veto_reason: Optional[str] = None
    details: Optional[Dict] = None


class RegimeFilter:
    """
    Unified regime filter that gates all entry decisions.

    Combines:
    1. Regime Quality Score - Is this a good time to trade?
    2. Transition Detection - Is the regime stable or changing?
    3. VIX-HMM Reconciliation - Do volatility signals agree?
    """

    def __init__(
        self,
        # Quality score thresholds
        min_quality_to_trade: float = 0.35,
        quality_for_full_size: float = 0.65,

        # Transition detection
        transition_lookback: int = 10,
        transition_threshold: float = 0.15,  # Max regime change per period

        # VIX-HMM reconciliation
        vix_hmm_max_divergence: float = 2.0,  # Max std devs of disagreement

        # VIX regime boundaries
        vix_low: float = 15.0,
        vix_normal: float = 20.0,
        vix_elevated: float = 25.0,
        vix_high: float = 35.0,
    ):
        # Thresholds (configurable via env vars)
        self.min_quality_to_trade = float(os.environ.get(
            'REGIME_MIN_QUALITY', str(min_quality_to_trade)))
        self.quality_for_full_size = float(os.environ.get(
            'REGIME_FULL_SIZE_QUALITY', str(quality_for_full_size)))

        # Transition detection
        self.transition_lookback = transition_lookback
        self.transition_threshold = transition_threshold
        self.regime_history = deque(maxlen=transition_lookback)

        # VIX-HMM reconciliation
        self.vix_hmm_max_divergence = vix_hmm_max_divergence
        self.vix_low = vix_low
        self.vix_normal = vix_normal
        self.vix_elevated = vix_elevated
        self.vix_high = vix_high

        # Tracking
        self.total_checks = 0
        self.vetoes = {'quality': 0, 'transition': 0, 'vix_hmm': 0}
        self.approvals = 0

        logger.info(f"RegimeFilter initialized:")
        logger.info(f"  Min quality to trade: {self.min_quality_to_trade}")
        logger.info(f"  Quality for full size: {self.quality_for_full_size}")
        logger.info(f"  Transition threshold: {self.transition_threshold}")
        logger.info(f"  VIX-HMM max divergence: {self.vix_hmm_max_divergence}")

    def calculate_regime_quality(
        self,
        hmm_trend: float,
        hmm_volatility: float,
        hmm_liquidity: float,
        hmm_confidence: float,
        vix_level: float,
        vix_percentile: float = 0.5,
    ) -> Tuple[float, Dict]:
        """
        Calculate composite regime quality score (0-1).

        Higher score = better trading conditions.

        Components:
        1. Trend Clarity (30%): Strong trends > neutral
        2. Volatility Sweet Spot (25%): Moderate vol > low or extreme
        3. HMM Confidence (20%): Higher confidence in regime detection
        4. VIX Stability (15%): Mid-range VIX > extremes
        5. Liquidity (10%): Higher liquidity better

        Returns:
            (quality_score, component_breakdown)
        """
        components = {}

        # 1. Trend Clarity (30%) - Favor strong directional trends
        # Best: trend near 0 or 1, Worst: trend near 0.5
        trend_clarity = abs(hmm_trend - 0.5) * 2  # 0-1 scale
        components['trend_clarity'] = trend_clarity

        # 2. Volatility Sweet Spot (25%) - Moderate vol is best
        # Too low = no movement, too high = unpredictable
        # Optimal: 0.3-0.5 range
        if hmm_volatility < 0.3:
            vol_score = hmm_volatility / 0.3 * 0.7  # Low vol: 0-0.7
        elif hmm_volatility <= 0.5:
            vol_score = 1.0  # Sweet spot: 1.0
        elif hmm_volatility <= 0.7:
            vol_score = 1.0 - (hmm_volatility - 0.5) / 0.2 * 0.5  # 1.0-0.5
        else:
            vol_score = 0.5 - (hmm_volatility - 0.7) / 0.3 * 0.5  # 0.5-0
        components['volatility_score'] = vol_score

        # 3. HMM Confidence (20%) - Trust the regime detection
        components['hmm_confidence'] = hmm_confidence

        # 4. VIX Stability (15%) - Mid-range VIX is best
        if vix_level < self.vix_low:
            vix_score = 0.6  # Too quiet, hard to profit
        elif vix_level <= self.vix_normal:
            vix_score = 1.0  # Sweet spot
        elif vix_level <= self.vix_elevated:
            vix_score = 0.8  # Elevated but tradable
        elif vix_level <= self.vix_high:
            vix_score = 0.5  # High vol, reduce size
        else:
            vix_score = 0.2  # Extreme, avoid
        components['vix_score'] = vix_score

        # 5. Liquidity (10%)
        components['liquidity'] = hmm_liquidity

        # Weighted combination
        quality = (
            0.30 * trend_clarity +
            0.25 * vol_score +
            0.20 * hmm_confidence +
            0.15 * vix_score +
            0.10 * hmm_liquidity
        )

        return quality, components

    def detect_regime_transition(
        self,
        hmm_trend: float,
        hmm_volatility: float,
        previous_regime: Optional[Dict] = None,
    ) -> Tuple[bool, float, str]:
        """
        Detect if market is transitioning between regimes.

        Returns:
            (is_transitioning, stability_score, transition_type)
        """
        # Store current regime
        current = {'trend': hmm_trend, 'vol': hmm_volatility}
        self.regime_history.append(current)

        if len(self.regime_history) < 3:
            return False, 1.0, 'insufficient_data'

        # Calculate regime changes over recent history
        changes = []
        history_list = list(self.regime_history)
        for i in range(1, len(history_list)):
            prev = history_list[i-1]
            curr = history_list[i]
            trend_change = abs(curr['trend'] - prev['trend'])
            vol_change = abs(curr['vol'] - prev['vol'])
            changes.append(max(trend_change, vol_change))

        avg_change = np.mean(changes)
        max_change = max(changes)

        # Determine transition state
        is_transitioning = avg_change > self.transition_threshold

        # Stability score: 1.0 = perfectly stable, 0.0 = rapidly changing
        stability = max(0.0, 1.0 - avg_change / self.transition_threshold)

        # Identify transition type
        if not is_transitioning:
            transition_type = 'stable'
        elif max_change > self.transition_threshold * 2:
            transition_type = 'rapid_transition'
        else:
            transition_type = 'gradual_transition'

        return is_transitioning, stability, transition_type

    def check_vix_hmm_alignment(
        self,
        hmm_volatility: float,
        vix_level: float,
    ) -> Tuple[bool, float, str]:
        """
        Check if VIX and HMM volatility signals agree.

        VIX is the "ground truth" for implied volatility.
        HMM detects realized volatility patterns.

        When they diverge significantly, trading is risky.

        Returns:
            (is_aligned, alignment_score, divergence_type)
        """
        # Map VIX to expected HMM volatility range
        if vix_level < self.vix_low:
            expected_hmm_vol = (0.0, 0.35)
        elif vix_level < self.vix_normal:
            expected_hmm_vol = (0.25, 0.50)
        elif vix_level < self.vix_elevated:
            expected_hmm_vol = (0.40, 0.65)
        elif vix_level < self.vix_high:
            expected_hmm_vol = (0.55, 0.80)
        else:
            expected_hmm_vol = (0.70, 1.00)

        # Check if HMM is within expected range
        low, high = expected_hmm_vol
        if low <= hmm_volatility <= high:
            alignment_score = 1.0
            divergence_type = 'aligned'
        elif hmm_volatility < low:
            # HMM says low vol, VIX says higher
            divergence = (low - hmm_volatility) / 0.3  # Normalize
            alignment_score = max(0.0, 1.0 - divergence)
            divergence_type = 'hmm_under_vix'
        else:
            # HMM says high vol, VIX says lower
            divergence = (hmm_volatility - high) / 0.3  # Normalize
            alignment_score = max(0.0, 1.0 - divergence)
            divergence_type = 'hmm_over_vix'

        # Check if divergence is too large
        is_aligned = alignment_score > (1.0 - 1.0 / self.vix_hmm_max_divergence)

        return is_aligned, alignment_score, divergence_type

    def should_trade(
        self,
        hmm_trend: float,
        hmm_volatility: float,
        hmm_liquidity: float,
        hmm_confidence: float,
        vix_level: float,
        vix_percentile: float = 0.5,
        previous_regime: Optional[Dict] = None,
    ) -> RegimeDecision:
        """
        Master function: Should we trade in this regime?

        Combines all three checks:
        1. Quality score must be above threshold
        2. Regime must not be transitioning rapidly
        3. VIX and HMM must agree on volatility

        Returns RegimeDecision with full details.
        """
        self.total_checks += 1
        details = {}

        # 1. Calculate regime quality
        quality, quality_components = self.calculate_regime_quality(
            hmm_trend, hmm_volatility, hmm_liquidity,
            hmm_confidence, vix_level, vix_percentile
        )
        details['quality'] = quality
        details['quality_components'] = quality_components

        # 2. Check for regime transition
        is_transitioning, stability, transition_type = self.detect_regime_transition(
            hmm_trend, hmm_volatility, previous_regime
        )
        details['is_transitioning'] = is_transitioning
        details['stability'] = stability
        details['transition_type'] = transition_type

        # 3. Check VIX-HMM alignment
        is_aligned, alignment_score, divergence_type = self.check_vix_hmm_alignment(
            hmm_volatility, vix_level
        )
        details['vix_hmm_aligned'] = is_aligned
        details['alignment_score'] = alignment_score
        details['divergence_type'] = divergence_type

        # Determine regime state
        if is_transitioning:
            regime_state = 'transitioning'
        elif quality < 0.3:
            regime_state = 'choppy'
        elif vix_level > self.vix_high:
            regime_state = 'extreme'
        elif abs(hmm_trend - 0.5) > 0.25:
            regime_state = 'trending'
        else:
            regime_state = 'neutral'

        details['regime_state'] = regime_state

        # Apply veto checks
        veto_reason = None

        # Veto 1: Quality too low
        if quality < self.min_quality_to_trade:
            veto_reason = f'low_quality ({quality:.2f} < {self.min_quality_to_trade})'
            self.vetoes['quality'] += 1

        # Veto 2: Rapid regime transition
        elif is_transitioning and transition_type == 'rapid_transition':
            veto_reason = f'rapid_transition (stability={stability:.2f})'
            self.vetoes['transition'] += 1

        # Veto 3: VIX-HMM divergence
        elif not is_aligned:
            veto_reason = f'vix_hmm_divergence ({divergence_type}, alignment={alignment_score:.2f})'
            self.vetoes['vix_hmm'] += 1

        # Calculate position scale
        if veto_reason:
            can_trade = False
            position_scale = 0.0
        else:
            can_trade = True
            self.approvals += 1

            # Scale position based on quality
            if quality >= self.quality_for_full_size:
                position_scale = 1.0
            else:
                # Linear scale from min_quality to full_size_quality
                range_size = self.quality_for_full_size - self.min_quality_to_trade
                position_scale = (quality - self.min_quality_to_trade) / range_size
                position_scale = max(0.3, min(1.0, position_scale))  # Clamp to 0.3-1.0

            # Reduce for transitions
            if is_transitioning:
                position_scale *= 0.7

            # Reduce for VIX-HMM divergence (even if not vetoed)
            if alignment_score < 0.8:
                position_scale *= alignment_score

        return RegimeDecision(
            can_trade=can_trade,
            quality_score=quality,
            position_scale=position_scale,
            regime_state=regime_state,
            veto_reason=veto_reason,
            details=details
        )

    def get_stats(self) -> Dict:
        """Get filter statistics"""
        total_vetoes = sum(self.vetoes.values())
        return {
            'total_checks': self.total_checks,
            'approvals': self.approvals,
            'total_vetoes': total_vetoes,
            'veto_breakdown': self.vetoes.copy(),
            'approval_rate': self.approvals / max(1, self.total_checks),
            'veto_rate': total_vetoes / max(1, self.total_checks),
        }

    def reset_stats(self):
        """Reset statistics"""
        self.total_checks = 0
        self.vetoes = {'quality': 0, 'transition': 0, 'vix_hmm': 0}
        self.approvals = 0
        self.regime_history.clear()


# Singleton instance for global access
_regime_filter: Optional[RegimeFilter] = None


def get_regime_filter(force_new: bool = False) -> RegimeFilter:
    """Get the global regime filter singleton"""
    global _regime_filter
    if _regime_filter is None or force_new:
        _regime_filter = RegimeFilter()
    return _regime_filter
