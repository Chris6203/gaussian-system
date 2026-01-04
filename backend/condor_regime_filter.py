#!/usr/bin/env python3
"""
Condor Regime Filter - Use Iron Condor logic as a filter for directional trades

CONCEPT:
- Iron Condors profit from range-bound, neutral markets
- Directional trades profit from trending markets
- These are OPPOSITES!

FILTER LOGIC:
- When condor would ENTER (neutral market) → BLOCK directional trades
- When condor would REJECT (trending market) → ALLOW directional trades

This filters out the choppy, sideways markets where directional bets
randomly win/lose, and only trades when there's a clear trend.
"""

import logging
from typing import Dict, Tuple, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class CondorRegimeSignal:
    """Result of condor regime filter check"""
    allow_directional: bool
    regime_type: str  # "trending", "neutral", "high_vol"
    confidence: float
    reasoning: str


class CondorRegimeFilter:
    """
    Uses Iron Condor entry logic INVERSELY as a filter for directional trades.

    When conditions are good for Iron Condors:
        → Market is range-bound, neutral
        → Directional trades will randomly win/lose
        → BLOCK directional entries

    When conditions are BAD for Iron Condors:
        → Market is trending or volatile
        → Directional trades can capture the move
        → ALLOW directional entries
    """

    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}

        # Thresholds for condor entry (when these are met, block directional)
        self.vix_max_for_condor = self.config.get('vix_max_for_condor', 25.0)
        self.vix_min_for_condor = self.config.get('vix_min_for_condor', 14.0)
        self.mtf_neutral_min = self.config.get('mtf_neutral_min', 0.40)
        self.mtf_neutral_max = self.config.get('mtf_neutral_max', 0.60)
        self.iv_rank_min = self.config.get('iv_rank_min', 20.0)

        # Thresholds for trending market (when these are met, allow directional)
        self.trend_threshold = self.config.get('trend_threshold', 0.70)  # HMM trend
        self.momentum_threshold = self.config.get('momentum_threshold', 0.003)  # 0.3%

        # Stats
        self.checks = 0
        self.blocks = 0
        self.allows = 0

        logger.info(f"[CONDOR FILTER] Initialized - blocks when MTF={self.mtf_neutral_min}-{self.mtf_neutral_max}, VIX={self.vix_min_for_condor}-{self.vix_max_for_condor}")

    def _estimate_iv_rank(self, vix: float) -> float:
        """Estimate IV Rank from VIX"""
        if vix < 15:
            return 10 + (vix - 12) * 5
        elif vix < 20:
            return 25 + (vix - 15) * 5
        elif vix < 30:
            return 50 + (vix - 20) * 3
        else:
            return 80 + min(20, (vix - 30))

    def _calculate_mtf_consensus(self, features: Dict) -> float:
        """Calculate MTF consensus (0=bearish, 0.5=neutral, 1=bullish)"""
        hmm_trend = features.get('hmm_trend', 0.5)
        momentum_5m = features.get('momentum_5m', 0.0)

        # Convert momentum to 0-1 scale
        mom_signal = 0.5 + min(0.3, max(-0.3, momentum_5m * 10))

        # Weighted average
        return 0.7 * hmm_trend + 0.3 * mom_signal

    def check_regime(self, features: Dict) -> CondorRegimeSignal:
        """
        Check if current market regime is good for directional trades.

        Returns:
            CondorRegimeSignal with allow_directional=True if trending market
        """
        self.checks += 1

        vix = features.get('vix_level', 18.0)
        hmm_trend = features.get('hmm_trend', 0.5)
        hmm_confidence = features.get('hmm_confidence', 0.5)
        momentum_5m = abs(features.get('momentum_5m', 0.0))
        model_confidence = features.get('confidence', 0.5)

        iv_rank = self._estimate_iv_rank(vix)
        mtf_consensus = self._calculate_mtf_consensus(features)

        reasons = []

        # === Check for TRENDING market (allow directional) ===

        # 1. Strong HMM trend
        is_trending = hmm_trend > self.trend_threshold or hmm_trend < (1 - self.trend_threshold)
        if is_trending:
            trend_dir = "BULLISH" if hmm_trend > 0.5 else "BEARISH"
            reasons.append(f"Strong {trend_dir} trend (HMM={hmm_trend:.2f})")

        # 2. High momentum
        has_momentum = momentum_5m > self.momentum_threshold
        if has_momentum:
            reasons.append(f"High momentum ({momentum_5m:.4f})")

        # 3. Extreme VIX (too high for condors)
        high_vol = vix > self.vix_max_for_condor
        if high_vol:
            reasons.append(f"High VIX ({vix:.1f})")

        # 4. Low VIX (not enough premium for condors)
        low_vol = vix < self.vix_min_for_condor
        if low_vol:
            reasons.append(f"Low VIX ({vix:.1f})")

        # 5. MTF NOT neutral (trending)
        mtf_trending = mtf_consensus < self.mtf_neutral_min or mtf_consensus > self.mtf_neutral_max
        if mtf_trending:
            reasons.append(f"MTF trending ({mtf_consensus:.2f})")

        # === Decision logic ===

        # Count trending signals
        trending_signals = sum([is_trending, has_momentum, high_vol, mtf_trending])
        neutral_signals = sum([not is_trending, not has_momentum, not high_vol and not low_vol, not mtf_trending])

        # If 2+ trending signals, allow directional trades
        if trending_signals >= 2:
            self.allows += 1
            confidence = min(1.0, trending_signals / 4.0 + hmm_confidence * 0.2)
            return CondorRegimeSignal(
                allow_directional=True,
                regime_type="trending",
                confidence=confidence,
                reasoning=f"TRENDING ({trending_signals}/4): " + " | ".join(reasons)
            )

        # If 3+ neutral signals, block directional (condor territory)
        elif neutral_signals >= 3:
            self.blocks += 1
            confidence = min(1.0, neutral_signals / 4.0)
            return CondorRegimeSignal(
                allow_directional=False,
                regime_type="neutral",
                confidence=confidence,
                reasoning=f"NEUTRAL (condor territory): VIX={vix:.1f}, MTF={mtf_consensus:.2f}, HMM={hmm_trend:.2f}"
            )

        # Mixed signals - default to allowing (err on side of trading)
        else:
            self.allows += 1
            return CondorRegimeSignal(
                allow_directional=True,
                regime_type="mixed",
                confidence=0.5,
                reasoning=f"MIXED: {trending_signals} trending, {neutral_signals} neutral signals"
            )

    def should_allow_trade(self, features: Dict) -> Tuple[bool, str]:
        """
        Simple interface: should we allow this directional trade?

        Returns:
            (allow, reasoning)
        """
        signal = self.check_regime(features)
        return signal.allow_directional, signal.reasoning

    def get_stats(self) -> Dict:
        """Return filter statistics"""
        return {
            'total_checks': self.checks,
            'blocks': self.blocks,
            'allows': self.allows,
            'block_rate': self.blocks / max(1, self.checks),
            'config': {
                'vix_range': f"{self.vix_min_for_condor}-{self.vix_max_for_condor}",
                'mtf_neutral': f"{self.mtf_neutral_min}-{self.mtf_neutral_max}",
                'trend_threshold': self.trend_threshold
            }
        }


# Global instance
_condor_filter = None

def get_condor_regime_filter(config: Optional[Dict] = None) -> CondorRegimeFilter:
    """Get or create the global condor regime filter"""
    global _condor_filter
    if _condor_filter is None:
        _condor_filter = CondorRegimeFilter(config)
    return _condor_filter


def should_allow_directional_trade(features: Dict) -> Tuple[bool, str]:
    """
    Quick check: should we allow a directional trade given current conditions?

    Uses Iron Condor logic inversely:
    - Condor-favorable conditions → block directional
    - Trending conditions → allow directional
    """
    filter = get_condor_regime_filter()
    return filter.should_allow_trade(features)


# === Test ===
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    filter = CondorRegimeFilter()

    test_cases = [
        # Trending market - should ALLOW directional
        {'vix_level': 22.0, 'hmm_trend': 0.82, 'momentum_5m': 0.005, 'confidence': 0.7, 'hmm_confidence': 0.8},
        # Neutral market - should BLOCK directional (condor territory)
        {'vix_level': 18.0, 'hmm_trend': 0.52, 'momentum_5m': 0.001, 'confidence': 0.4, 'hmm_confidence': 0.6},
        # High VIX - should ALLOW directional (too risky for condors)
        {'vix_level': 32.0, 'hmm_trend': 0.45, 'momentum_5m': 0.002, 'confidence': 0.5, 'hmm_confidence': 0.5},
        # Low VIX, neutral - should BLOCK (no premium but range-bound)
        {'vix_level': 12.0, 'hmm_trend': 0.50, 'momentum_5m': 0.0005, 'confidence': 0.3, 'hmm_confidence': 0.7},
        # Strong bearish - should ALLOW directional
        {'vix_level': 25.0, 'hmm_trend': 0.18, 'momentum_5m': -0.008, 'confidence': 0.65, 'hmm_confidence': 0.85},
    ]

    print("=" * 70)
    print("CONDOR REGIME FILTER TEST")
    print("=" * 70)

    for i, features in enumerate(test_cases, 1):
        signal = filter.check_regime(features)
        action = "ALLOW ✓" if signal.allow_directional else "BLOCK ✗"
        print(f"\nTest {i}: VIX={features['vix_level']}, HMM={features['hmm_trend']:.2f}")
        print(f"  {action} | Regime: {signal.regime_type} | Conf: {signal.confidence:.2f}")
        print(f"  {signal.reasoning}")

    print("\n" + "=" * 70)
    print("STATS:", filter.get_stats())
