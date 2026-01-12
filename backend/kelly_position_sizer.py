#!/usr/bin/env python3
"""
Kelly Criterion Position Sizing

Simons Philosophy: Optimal position sizing is as important as signal quality.
Carmack Philosophy: Simple formula, measure results, adapt.

The Kelly Criterion maximizes long-term growth while controlling drawdowns.

Formula: f* = (p * b - q) / b
Where:
- p = probability of winning
- q = probability of losing (1 - p)
- b = win/loss ratio (avg win / avg loss)
- f* = fraction of capital to risk

We use FRACTIONAL Kelly (f*/2 or f*/4) because:
1. We don't know exact probabilities
2. Reduces variance/drawdowns
3. More conservative in practice

Environment Variables:
- KELLY_POSITION_SIZING=1: Enable Kelly position sizing
- KELLY_FRACTION=0.25: Fraction of Kelly to use (0.25 = quarter Kelly)
- KELLY_MIN_TRADES=20: Minimum trades before using Kelly
- KELLY_MAX_POSITION=0.10: Maximum position size (10% of capital)
- KELLY_MIN_POSITION=0.01: Minimum position size (1% of capital)
"""

import os
import logging
import numpy as np
from dataclasses import dataclass
from typing import Optional, List, Tuple
from collections import deque

logger = logging.getLogger(__name__)

# Configuration
KELLY_POSITION_SIZING = os.environ.get('KELLY_POSITION_SIZING', '0') == '1'
KELLY_FRACTION = float(os.environ.get('KELLY_FRACTION', '0.25'))
KELLY_MIN_TRADES = int(os.environ.get('KELLY_MIN_TRADES', '20'))
KELLY_MAX_POSITION = float(os.environ.get('KELLY_MAX_POSITION', '0.10'))
KELLY_MIN_POSITION = float(os.environ.get('KELLY_MIN_POSITION', '0.01'))


@dataclass
class PositionSize:
    """Recommended position size."""
    fraction: float  # Fraction of capital (0.0 to 1.0)
    amount: float  # Dollar amount
    kelly_raw: float  # Raw Kelly criterion value
    kelly_adjusted: float  # After fractional adjustment
    reason: str
    win_rate: float
    win_loss_ratio: float


class KellyPositionSizer:
    """
    Position sizing using Kelly Criterion.

    Uses historical trade data to estimate optimal position size.
    """

    def __init__(self):
        self.enabled = KELLY_POSITION_SIZING
        self.kelly_fraction = KELLY_FRACTION
        self.min_trades = KELLY_MIN_TRADES
        self.max_position = KELLY_MAX_POSITION
        self.min_position = KELLY_MIN_POSITION

        # Trade history for Kelly calculation
        self.trade_results: deque = deque(maxlen=100)  # Last 100 trades
        self.call_results: deque = deque(maxlen=50)
        self.put_results: deque = deque(maxlen=50)

        # Current estimates
        self.estimated_win_rate = 0.5
        self.estimated_win_loss_ratio = 1.0

        if self.enabled:
            logger.info(f"📊 Kelly Position Sizer ENABLED")
            logger.info(f"   Fraction: {self.kelly_fraction} (quarter-Kelly)")
            logger.info(f"   Min trades: {self.min_trades}")
            logger.info(f"   Position range: {self.min_position:.1%} - {self.max_position:.1%}")

    def record_trade(self, pnl_pct: float, direction: str = 'UNKNOWN'):
        """
        Record a completed trade for Kelly calculation.

        Args:
            pnl_pct: P&L as decimal (0.05 = 5% profit)
            direction: 'CALL' or 'PUT'
        """
        self.trade_results.append(pnl_pct)

        if direction == 'CALL':
            self.call_results.append(pnl_pct)
        elif direction == 'PUT':
            self.put_results.append(pnl_pct)

        # Update estimates
        self._update_estimates()

    def _update_estimates(self):
        """Update win rate and win/loss ratio estimates."""
        if len(self.trade_results) < 5:
            return

        results = list(self.trade_results)
        wins = [r for r in results if r > 0]
        losses = [r for r in results if r <= 0]

        if len(wins) == 0:
            self.estimated_win_rate = 0.0
            self.estimated_win_loss_ratio = 0.0
            return

        if len(losses) == 0:
            self.estimated_win_rate = 1.0
            self.estimated_win_loss_ratio = float('inf')
            return

        self.estimated_win_rate = len(wins) / len(results)

        avg_win = np.mean(wins)
        avg_loss = abs(np.mean(losses))

        if avg_loss > 0:
            self.estimated_win_loss_ratio = avg_win / avg_loss
        else:
            self.estimated_win_loss_ratio = float('inf')

    def calculate_kelly(self, win_rate: Optional[float] = None,
                        win_loss_ratio: Optional[float] = None) -> float:
        """
        Calculate raw Kelly criterion.

        Returns fraction of capital to bet (can be > 1 or < 0).
        """
        p = win_rate if win_rate is not None else self.estimated_win_rate
        b = win_loss_ratio if win_loss_ratio is not None else self.estimated_win_loss_ratio

        if b <= 0 or p <= 0:
            return 0.0

        if b == float('inf'):
            return 1.0  # All wins, max bet

        q = 1 - p
        kelly = (p * b - q) / b

        return kelly

    def get_position_size(
        self,
        capital: float,
        confidence: float = 0.5,
        direction: str = 'UNKNOWN'
    ) -> PositionSize:
        """
        Get recommended position size.

        Args:
            capital: Total available capital
            confidence: Model confidence (0.0 to 1.0)
            direction: 'CALL' or 'PUT' for direction-specific sizing

        Returns:
            PositionSize with recommended fraction and amount
        """
        if not self.enabled:
            # Return default position size
            return PositionSize(
                fraction=0.05,
                amount=capital * 0.05,
                kelly_raw=0.0,
                kelly_adjusted=0.0,
                reason='kelly_disabled',
                win_rate=0.5,
                win_loss_ratio=1.0
            )

        # Check if we have enough data
        if len(self.trade_results) < self.min_trades:
            # Use conservative default until we have data
            default_frac = 0.02  # 2% of capital
            return PositionSize(
                fraction=default_frac,
                amount=capital * default_frac,
                kelly_raw=0.0,
                kelly_adjusted=0.0,
                reason=f'insufficient_data ({len(self.trade_results)}/{self.min_trades} trades)',
                win_rate=self.estimated_win_rate,
                win_loss_ratio=self.estimated_win_loss_ratio
            )

        # Get direction-specific stats if available
        if direction == 'CALL' and len(self.call_results) >= 10:
            results = list(self.call_results)
            wins = [r for r in results if r > 0]
            losses = [r for r in results if r <= 0]
            if wins and losses:
                win_rate = len(wins) / len(results)
                win_loss_ratio = np.mean(wins) / abs(np.mean(losses))
            else:
                win_rate = self.estimated_win_rate
                win_loss_ratio = self.estimated_win_loss_ratio
        elif direction == 'PUT' and len(self.put_results) >= 10:
            results = list(self.put_results)
            wins = [r for r in results if r > 0]
            losses = [r for r in results if r <= 0]
            if wins and losses:
                win_rate = len(wins) / len(results)
                win_loss_ratio = np.mean(wins) / abs(np.mean(losses))
            else:
                win_rate = self.estimated_win_rate
                win_loss_ratio = self.estimated_win_loss_ratio
        else:
            win_rate = self.estimated_win_rate
            win_loss_ratio = self.estimated_win_loss_ratio

        # Calculate raw Kelly
        kelly_raw = self.calculate_kelly(win_rate, win_loss_ratio)

        # Apply fractional Kelly
        kelly_adjusted = kelly_raw * self.kelly_fraction

        # Scale by confidence (higher confidence = closer to full Kelly fraction)
        # This prevents over-betting on low-confidence signals
        confidence_scale = 0.5 + 0.5 * confidence  # Range: 0.5 to 1.0
        kelly_adjusted *= confidence_scale

        # Clamp to bounds
        if kelly_adjusted < 0:
            # Negative Kelly = edge is negative, don't trade
            return PositionSize(
                fraction=0.0,
                amount=0.0,
                kelly_raw=kelly_raw,
                kelly_adjusted=kelly_adjusted,
                reason=f'negative_edge (kelly={kelly_raw:.2f})',
                win_rate=win_rate,
                win_loss_ratio=win_loss_ratio
            )

        final_fraction = np.clip(kelly_adjusted, self.min_position, self.max_position)
        final_amount = capital * final_fraction

        reason = f'kelly={kelly_raw:.2f}, adj={kelly_adjusted:.2f}, wr={win_rate:.1%}, ratio={win_loss_ratio:.2f}'

        return PositionSize(
            fraction=final_fraction,
            amount=final_amount,
            kelly_raw=kelly_raw,
            kelly_adjusted=kelly_adjusted,
            reason=reason,
            win_rate=win_rate,
            win_loss_ratio=win_loss_ratio
        )

    def get_stats(self) -> dict:
        """Get sizer statistics."""
        return {
            'enabled': self.enabled,
            'total_trades': len(self.trade_results),
            'call_trades': len(self.call_results),
            'put_trades': len(self.put_results),
            'estimated_win_rate': self.estimated_win_rate,
            'estimated_win_loss_ratio': self.estimated_win_loss_ratio,
            'current_kelly': self.calculate_kelly(),
            'adjusted_kelly': self.calculate_kelly() * self.kelly_fraction
        }


# Global instance
_kelly_sizer = None


def get_kelly_sizer() -> KellyPositionSizer:
    """Get or create the global Kelly position sizer."""
    global _kelly_sizer
    if _kelly_sizer is None:
        _kelly_sizer = KellyPositionSizer()
    return _kelly_sizer


def get_position_size(
    capital: float,
    confidence: float = 0.5,
    direction: str = 'UNKNOWN'
) -> PositionSize:
    """
    Convenience function to get position size.

    Usage:
        size = get_position_size(capital=5000, confidence=0.6, direction='CALL')
        if size.fraction > 0:
            position_amount = size.amount
            print(f"Bet ${position_amount:.2f} ({size.fraction:.1%} of capital)")
            print(f"Reason: {size.reason}")
    """
    sizer = get_kelly_sizer()
    return sizer.get_position_size(capital, confidence, direction)


def record_trade_result(pnl_pct: float, direction: str = 'UNKNOWN'):
    """Record a trade result for Kelly calculation."""
    sizer = get_kelly_sizer()
    sizer.record_trade(pnl_pct, direction)
