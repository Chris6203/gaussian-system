#!/usr/bin/env python3
"""
SPY-VIX Correlation Gate

Based on Simons Analysis (Phase 58): SPY-VIX correlation is the STRONGEST
predictive signal found (r=-0.34 with forward returns).

Key Insight:
- Normal market: SPY and VIX are NEGATIVELY correlated (SPY up = VIX down)
- Stress/regime change: Correlation becomes POSITIVE (unusual)
- When correlation is positive → expect market DOWN → be cautious or short

Trading Rules:
1. Correlation < -0.3 (normal): Trade freely, slight bullish bias
2. Correlation -0.3 to +0.1: Neutral, trade with caution
3. Correlation > +0.1 (abnormal): Regime stress, avoid calls, consider puts
4. Correlation > +0.3 (extreme): Full crash protection mode

Environment Variables:
- SPY_VIX_GATE=1: Enable SPY-VIX correlation gating
- SPY_VIX_LOOKBACK=20: Correlation lookback period (minutes)
- SPY_VIX_NORMAL_THRESH=-0.3: Threshold for "normal" correlation
- SPY_VIX_STRESS_THRESH=0.1: Threshold for "stress" correlation
- SPY_VIX_CRASH_THRESH=0.3: Threshold for "crash protection" mode
"""

import os
import logging
import numpy as np
from dataclasses import dataclass
from typing import Optional, List, Tuple
from collections import deque

logger = logging.getLogger(__name__)

# Configuration
SPY_VIX_GATE = os.environ.get('SPY_VIX_GATE', '0') == '1'
SPY_VIX_LOOKBACK = int(os.environ.get('SPY_VIX_LOOKBACK', '20'))
SPY_VIX_NORMAL_THRESH = float(os.environ.get('SPY_VIX_NORMAL_THRESH', '-0.3'))
SPY_VIX_STRESS_THRESH = float(os.environ.get('SPY_VIX_STRESS_THRESH', '0.1'))
SPY_VIX_CRASH_THRESH = float(os.environ.get('SPY_VIX_CRASH_THRESH', '0.3'))


@dataclass
class CorrelationRegime:
    """Current market regime based on SPY-VIX correlation."""
    regime: str  # 'NORMAL', 'CAUTION', 'STRESS', 'CRASH'
    correlation: float
    spy_momentum: float
    vix_momentum: float
    trade_bias: str  # 'BULLISH', 'NEUTRAL', 'BEARISH', 'NO_TRADE'
    confidence_multiplier: float  # Scale confidence by this
    reason: str


class SpyVixCorrelationGate:
    """
    Gate trades based on SPY-VIX correlation regime.

    Simons insight: Many weak signals combined = strong edge.
    SPY-VIX correlation is our STRONGEST single signal (r=-0.34).
    """

    def __init__(self):
        self.enabled = SPY_VIX_GATE
        self.lookback = SPY_VIX_LOOKBACK
        self.normal_thresh = SPY_VIX_NORMAL_THRESH
        self.stress_thresh = SPY_VIX_STRESS_THRESH
        self.crash_thresh = SPY_VIX_CRASH_THRESH

        # Price histories
        self.spy_prices: deque = deque(maxlen=200)
        self.vix_prices: deque = deque(maxlen=200)

        # Correlation history for trend detection
        self.correlation_history: deque = deque(maxlen=50)

        if self.enabled:
            logger.info(f"📊 SPY-VIX Correlation Gate ENABLED")
            logger.info(f"   Lookback: {self.lookback} periods")
            logger.info(f"   Normal threshold: {self.normal_thresh}")
            logger.info(f"   Stress threshold: {self.stress_thresh}")
            logger.info(f"   Crash threshold: {self.crash_thresh}")

    def update(self, spy_price: float, vix_price: Optional[float] = None):
        """Update price histories."""
        self.spy_prices.append(spy_price)
        if vix_price is not None:
            self.vix_prices.append(vix_price)

    def calculate_correlation(self) -> Optional[float]:
        """
        Calculate rolling SPY-VIX return correlation.

        Returns None if insufficient data.
        """
        if len(self.spy_prices) < self.lookback or len(self.vix_prices) < self.lookback:
            return None

        # Get recent prices
        spy = np.array(list(self.spy_prices)[-self.lookback:])
        vix = np.array(list(self.vix_prices)[-self.lookback:])

        if len(spy) != len(vix):
            return None

        # Calculate returns
        spy_ret = np.diff(spy) / spy[:-1]
        vix_ret = np.diff(vix) / vix[:-1]

        if len(spy_ret) < 5:
            return None

        # Correlation
        corr = np.corrcoef(spy_ret, vix_ret)[0, 1]

        if np.isnan(corr):
            return None

        # Track history
        self.correlation_history.append(corr)

        return corr

    def _calculate_momentum(self, prices: deque, period: int = 5) -> float:
        """Calculate price momentum."""
        if len(prices) < period + 1:
            return 0.0

        prices_list = list(prices)
        return (prices_list[-1] - prices_list[-period-1]) / prices_list[-period-1]

    def _detect_correlation_trend(self) -> str:
        """Detect if correlation is trending up (worsening) or down (improving)."""
        if len(self.correlation_history) < 10:
            return 'STABLE'

        recent = list(self.correlation_history)[-10:]
        older = list(self.correlation_history)[-20:-10] if len(self.correlation_history) >= 20 else recent

        recent_avg = np.mean(recent)
        older_avg = np.mean(older)

        if recent_avg > older_avg + 0.1:
            return 'WORSENING'  # Correlation becoming more positive (bad)
        elif recent_avg < older_avg - 0.1:
            return 'IMPROVING'  # Correlation becoming more negative (good)
        else:
            return 'STABLE'

    def get_regime(self) -> CorrelationRegime:
        """
        Get current market regime based on SPY-VIX correlation.

        Returns regime classification and trading bias.
        """
        if not self.enabled:
            return CorrelationRegime(
                regime='DISABLED',
                correlation=0.0,
                spy_momentum=0.0,
                vix_momentum=0.0,
                trade_bias='NEUTRAL',
                confidence_multiplier=1.0,
                reason='gate_disabled'
            )

        correlation = self.calculate_correlation()

        if correlation is None:
            return CorrelationRegime(
                regime='WARMUP',
                correlation=0.0,
                spy_momentum=0.0,
                vix_momentum=0.0,
                trade_bias='NEUTRAL',
                confidence_multiplier=0.5,  # Reduce confidence during warmup
                reason='insufficient_data'
            )

        spy_momentum = self._calculate_momentum(self.spy_prices)
        vix_momentum = self._calculate_momentum(self.vix_prices)
        corr_trend = self._detect_correlation_trend()

        # Determine regime
        if correlation >= self.crash_thresh:
            # CRASH MODE: Very high positive correlation = extreme stress
            return CorrelationRegime(
                regime='CRASH',
                correlation=correlation,
                spy_momentum=spy_momentum,
                vix_momentum=vix_momentum,
                trade_bias='NO_TRADE',
                confidence_multiplier=0.0,  # No trading
                reason=f'CRASH: corr={correlation:.2f} >= {self.crash_thresh} (extreme stress)'
            )

        elif correlation >= self.stress_thresh:
            # STRESS MODE: Positive correlation = market stress
            # Favor puts, avoid calls
            bias = 'BEARISH'
            multiplier = 0.5  # Reduce position sizes

            return CorrelationRegime(
                regime='STRESS',
                correlation=correlation,
                spy_momentum=spy_momentum,
                vix_momentum=vix_momentum,
                trade_bias=bias,
                confidence_multiplier=multiplier,
                reason=f'STRESS: corr={correlation:.2f} > {self.stress_thresh} (unusual, bearish bias)'
            )

        elif correlation <= self.normal_thresh:
            # NORMAL MODE: Strong negative correlation = healthy market
            # Good conditions for trading, slight bullish bias
            if corr_trend == 'WORSENING':
                multiplier = 0.8  # Slightly cautious if trend worsening
                bias = 'NEUTRAL'
            else:
                multiplier = 1.0
                bias = 'BULLISH'

            return CorrelationRegime(
                regime='NORMAL',
                correlation=correlation,
                spy_momentum=spy_momentum,
                vix_momentum=vix_momentum,
                trade_bias=bias,
                confidence_multiplier=multiplier,
                reason=f'NORMAL: corr={correlation:.2f} < {self.normal_thresh} (healthy market)'
            )

        else:
            # CAUTION MODE: Correlation in transition zone
            multiplier = 0.7

            if corr_trend == 'WORSENING':
                bias = 'BEARISH'
                multiplier = 0.5
            elif corr_trend == 'IMPROVING':
                bias = 'NEUTRAL'
                multiplier = 0.8
            else:
                bias = 'NEUTRAL'

            return CorrelationRegime(
                regime='CAUTION',
                correlation=correlation,
                spy_momentum=spy_momentum,
                vix_momentum=vix_momentum,
                trade_bias=bias,
                confidence_multiplier=multiplier,
                reason=f'CAUTION: corr={correlation:.2f} in transition (trend: {corr_trend})'
            )

    def should_trade(self, proposed_action: str) -> Tuple[bool, float, str]:
        """
        Check if a trade should proceed based on correlation regime.

        Args:
            proposed_action: 'BUY_CALLS' or 'BUY_PUTS'

        Returns:
            (should_trade, confidence_multiplier, reason)
        """
        regime = self.get_regime()

        # Crash mode - no trading
        if regime.regime == 'CRASH':
            return False, 0.0, regime.reason

        # Warmup - allow with reduced confidence
        if regime.regime == 'WARMUP':
            return True, 0.5, regime.reason

        # Check if action aligns with regime bias
        if regime.trade_bias == 'NO_TRADE':
            return False, 0.0, regime.reason

        if proposed_action == 'BUY_CALLS':
            if regime.trade_bias == 'BEARISH':
                # Calls in bearish regime - discourage
                return True, regime.confidence_multiplier * 0.5, f"{regime.reason} (CALLS discouraged)"
            elif regime.trade_bias == 'BULLISH':
                # Calls in bullish regime - encourage
                return True, min(1.2, regime.confidence_multiplier * 1.1), f"{regime.reason} (CALLS favored)"
            else:
                return True, regime.confidence_multiplier, regime.reason

        elif proposed_action == 'BUY_PUTS':
            if regime.trade_bias == 'BULLISH':
                # Puts in bullish regime - discourage
                return True, regime.confidence_multiplier * 0.5, f"{regime.reason} (PUTS discouraged)"
            elif regime.trade_bias == 'BEARISH':
                # Puts in bearish regime - encourage
                return True, min(1.2, regime.confidence_multiplier * 1.1), f"{regime.reason} (PUTS favored)"
            else:
                return True, regime.confidence_multiplier, regime.reason

        return True, regime.confidence_multiplier, regime.reason

    def get_stats(self) -> dict:
        """Get gate statistics for debugging."""
        correlation = self.calculate_correlation()
        regime = self.get_regime()

        return {
            'enabled': self.enabled,
            'current_correlation': correlation,
            'regime': regime.regime,
            'trade_bias': regime.trade_bias,
            'confidence_multiplier': regime.confidence_multiplier,
            'spy_price': self.spy_prices[-1] if self.spy_prices else None,
            'vix_price': self.vix_prices[-1] if self.vix_prices else None,
            'correlation_history_len': len(self.correlation_history)
        }


# Global instance
_spy_vix_gate = None


def get_spy_vix_gate() -> SpyVixCorrelationGate:
    """Get or create the global SPY-VIX correlation gate."""
    global _spy_vix_gate
    if _spy_vix_gate is None:
        _spy_vix_gate = SpyVixCorrelationGate()
    return _spy_vix_gate


def check_spy_vix_correlation(
    proposed_action: str,
    spy_price: float,
    vix_price: Optional[float] = None
) -> Tuple[bool, float, str]:
    """
    Convenience function to check SPY-VIX correlation gate.

    Usage:
        should_trade, conf_mult, reason = check_spy_vix_correlation('BUY_CALLS', 610.0, 16.5)
        if should_trade:
            adjusted_confidence = original_confidence * conf_mult
            execute_trade(confidence=adjusted_confidence)
        else:
            print(f"Trade blocked: {reason}")
    """
    gate = get_spy_vix_gate()
    gate.update(spy_price, vix_price)
    return gate.should_trade(proposed_action)
