#!/usr/bin/env python3
"""
Win Probability Estimator

Simons Philosophy: Don't trade on hunches - calculate expected value.
Carmack Philosophy: Measure actual outcomes, calibrate predictions.

This module transforms raw model confidence into calibrated P(win).

Key Insight:
- Raw confidence (e.g., 60%) doesn't mean 60% win rate
- We need to track actual outcomes at each confidence level
- Then use Bayesian updating to estimate true P(win)

Expected Value Calculation:
  EV = P(win) * avg_win - P(loss) * avg_loss

Only trade when EV > threshold (e.g., > $0.50 per trade)

Environment Variables:
- WIN_PROB_ESTIMATOR=1: Enable win probability estimation
- WIN_PROB_MIN_EV=0.5: Minimum expected value to trade ($)
- WIN_PROB_MIN_SAMPLES=10: Min samples per bucket for calibration
- WIN_PROB_CONFIDENCE_BUCKETS=5: Number of confidence buckets
"""

import os
import logging
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Tuple
from collections import defaultdict

logger = logging.getLogger(__name__)

# Configuration
WIN_PROB_ESTIMATOR = os.environ.get('WIN_PROB_ESTIMATOR', '0') == '1'
WIN_PROB_MIN_EV = float(os.environ.get('WIN_PROB_MIN_EV', '0.0'))  # Allow all positive EV trades
WIN_PROB_MIN_SAMPLES = int(os.environ.get('WIN_PROB_MIN_SAMPLES', '10'))
WIN_PROB_CONFIDENCE_BUCKETS = int(os.environ.get('WIN_PROB_CONFIDENCE_BUCKETS', '5'))


@dataclass
class TradeRecord:
    """Record of a trade for probability estimation."""
    confidence: float
    direction: str  # 'CALL' or 'PUT'
    predicted_return: float
    actual_pnl: float
    is_win: bool
    spy_vix_regime: str = 'UNKNOWN'
    rsi: float = 50.0


@dataclass
class ProbabilityEstimate:
    """Estimated probabilities and expected value."""
    p_win: float  # Calibrated probability of winning
    p_loss: float  # 1 - p_win
    avg_win: float  # Average win amount ($)
    avg_loss: float  # Average loss amount ($)
    expected_value: float  # EV = p_win * avg_win - p_loss * avg_loss
    should_trade: bool  # True if EV > threshold
    confidence_bucket: int
    samples_in_bucket: int
    reason: str


class WinProbabilityEstimator:
    """
    Estimates calibrated win probability from historical data.

    Uses bucketed confidence levels to track actual outcomes.
    """

    def __init__(self):
        self.enabled = WIN_PROB_ESTIMATOR
        self.min_ev = WIN_PROB_MIN_EV
        self.min_samples = WIN_PROB_MIN_SAMPLES
        self.n_buckets = WIN_PROB_CONFIDENCE_BUCKETS

        # Track outcomes by confidence bucket
        # Bucket 0: 0-20%, Bucket 1: 20-40%, etc.
        self.call_outcomes: Dict[int, List[TradeRecord]] = defaultdict(list)
        self.put_outcomes: Dict[int, List[TradeRecord]] = defaultdict(list)

        # Track by regime too
        self.regime_outcomes: Dict[str, List[TradeRecord]] = defaultdict(list)

        # Overall stats
        self.total_trades = 0
        self.total_wins = 0

        # Prior estimates (before we have data)
        # Based on our actual data: 54% WR with SPY-VIX gate
        self.prior_win_rate = 0.54
        self.prior_avg_win = 2.5  # $2.50 average win
        self.prior_avg_loss = 2.5  # $2.50 average loss (balanced)

        if self.enabled:
            logger.info(f"📊 Win Probability Estimator ENABLED")
            logger.info(f"   Min EV: ${self.min_ev}")
            logger.info(f"   Min samples: {self.min_samples}")
            logger.info(f"   Buckets: {self.n_buckets}")

    def _get_bucket(self, confidence: float) -> int:
        """Get confidence bucket (0 to n_buckets-1)."""
        bucket = int(confidence * self.n_buckets)
        return min(bucket, self.n_buckets - 1)

    def record_trade(self, record: TradeRecord):
        """Record a completed trade for probability estimation."""
        bucket = self._get_bucket(record.confidence)

        if record.direction == 'CALL':
            self.call_outcomes[bucket].append(record)
        else:
            self.put_outcomes[bucket].append(record)

        self.regime_outcomes[record.spy_vix_regime].append(record)

        self.total_trades += 1
        if record.is_win:
            self.total_wins += 1

    def _calculate_bucket_stats(self, outcomes: List[TradeRecord]) -> Tuple[float, float, float]:
        """Calculate win rate, avg win, avg loss for a list of outcomes."""
        if not outcomes:
            return self.prior_win_rate, self.prior_avg_win, self.prior_avg_loss

        wins = [r for r in outcomes if r.is_win]
        losses = [r for r in outcomes if not r.is_win]

        # Win rate with Bayesian smoothing (add 1 win and 1 loss as prior)
        win_rate = (len(wins) + 1) / (len(outcomes) + 2)

        # Average win/loss
        avg_win = np.mean([r.actual_pnl for r in wins]) if wins else self.prior_avg_win
        avg_loss = abs(np.mean([r.actual_pnl for r in losses])) if losses else self.prior_avg_loss

        return win_rate, avg_win, avg_loss

    def get_probability(
        self,
        confidence: float,
        direction: str,
        predicted_return: float = 0.0,
        spy_vix_regime: str = 'UNKNOWN',
        trade_amount: float = 50.0  # Default option cost
    ) -> ProbabilityEstimate:
        """
        Get calibrated win probability and expected value.

        Args:
            confidence: Model confidence (0.0 to 1.0)
            direction: 'CALL' or 'PUT'
            predicted_return: Model's predicted return
            spy_vix_regime: Current market regime
            trade_amount: Dollar amount of trade

        Returns:
            ProbabilityEstimate with P(win), EV, and trade recommendation
        """
        if not self.enabled:
            return ProbabilityEstimate(
                p_win=0.5,
                p_loss=0.5,
                avg_win=self.prior_avg_win,
                avg_loss=self.prior_avg_loss,
                expected_value=0.0,
                should_trade=True,
                confidence_bucket=0,
                samples_in_bucket=0,
                reason='estimator_disabled'
            )

        bucket = self._get_bucket(confidence)

        # Get relevant outcomes
        outcomes = self.call_outcomes[bucket] if direction == 'CALL' else self.put_outcomes[bucket]
        n_samples = len(outcomes)

        # Calculate calibrated probability
        if n_samples >= self.min_samples:
            # Enough data - use empirical estimates
            p_win, avg_win, avg_loss = self._calculate_bucket_stats(outcomes)
            calibration_source = 'empirical'
        else:
            # Not enough data - use prior with partial update
            if n_samples > 0:
                emp_win, emp_avg_win, emp_avg_loss = self._calculate_bucket_stats(outcomes)
                # Blend prior and empirical based on sample size
                alpha = n_samples / self.min_samples
                p_win = (1 - alpha) * self.prior_win_rate + alpha * emp_win
                avg_win = (1 - alpha) * self.prior_avg_win + alpha * emp_avg_win
                avg_loss = (1 - alpha) * self.prior_avg_loss + alpha * emp_avg_loss
                calibration_source = f'blended ({n_samples}/{self.min_samples})'
            else:
                p_win = self.prior_win_rate
                avg_win = self.prior_avg_win
                avg_loss = self.prior_avg_loss
                calibration_source = 'prior'

        # Adjust for regime if we have data
        regime_outcomes = self.regime_outcomes.get(spy_vix_regime, [])
        if len(regime_outcomes) >= 5:
            regime_wr, _, _ = self._calculate_bucket_stats(regime_outcomes)
            # Blend with regime-specific adjustment
            p_win = 0.7 * p_win + 0.3 * regime_wr

        # Calculate expected value
        p_loss = 1 - p_win
        expected_value = p_win * avg_win - p_loss * avg_loss

        # Should we trade?
        should_trade = expected_value >= self.min_ev

        reason = (f"{calibration_source}: P(win)={p_win:.1%}, "
                  f"EV=${expected_value:.2f}, "
                  f"bucket={bucket}, n={n_samples}")

        if not should_trade:
            reason = f"REJECT: EV=${expected_value:.2f} < ${self.min_ev} | " + reason

        return ProbabilityEstimate(
            p_win=p_win,
            p_loss=p_loss,
            avg_win=avg_win,
            avg_loss=avg_loss,
            expected_value=expected_value,
            should_trade=should_trade,
            confidence_bucket=bucket,
            samples_in_bucket=n_samples,
            reason=reason
        )

    def get_all_bucket_stats(self) -> Dict:
        """Get statistics for all buckets (for debugging/visualization)."""
        stats = {
            'total_trades': self.total_trades,
            'total_wins': self.total_wins,
            'overall_win_rate': self.total_wins / self.total_trades if self.total_trades > 0 else 0,
            'call_buckets': {},
            'put_buckets': {}
        }

        for bucket in range(self.n_buckets):
            conf_range = f"{bucket * (100 // self.n_buckets)}-{(bucket + 1) * (100 // self.n_buckets)}%"

            # CALL stats
            call_outcomes = self.call_outcomes[bucket]
            if call_outcomes:
                call_wr, call_avg_win, call_avg_loss = self._calculate_bucket_stats(call_outcomes)
                stats['call_buckets'][conf_range] = {
                    'n': len(call_outcomes),
                    'win_rate': call_wr,
                    'avg_win': call_avg_win,
                    'avg_loss': call_avg_loss,
                    'ev': call_wr * call_avg_win - (1 - call_wr) * call_avg_loss
                }

            # PUT stats
            put_outcomes = self.put_outcomes[bucket]
            if put_outcomes:
                put_wr, put_avg_win, put_avg_loss = self._calculate_bucket_stats(put_outcomes)
                stats['put_buckets'][conf_range] = {
                    'n': len(put_outcomes),
                    'win_rate': put_wr,
                    'avg_win': put_avg_win,
                    'avg_loss': put_avg_loss,
                    'ev': put_wr * put_avg_win - (1 - put_wr) * put_avg_loss
                }

        return stats

    def get_summary(self) -> str:
        """Get human-readable summary."""
        stats = self.get_all_bucket_stats()
        lines = [
            f"Win Probability Estimator Summary",
            f"================================",
            f"Total trades: {stats['total_trades']}",
            f"Overall win rate: {stats['overall_win_rate']:.1%}",
            "",
            "CALL buckets:"
        ]

        for conf_range, bucket_stats in stats['call_buckets'].items():
            lines.append(f"  {conf_range}: n={bucket_stats['n']}, "
                        f"WR={bucket_stats['win_rate']:.1%}, "
                        f"EV=${bucket_stats['ev']:.2f}")

        lines.append("")
        lines.append("PUT buckets:")

        for conf_range, bucket_stats in stats['put_buckets'].items():
            lines.append(f"  {conf_range}: n={bucket_stats['n']}, "
                        f"WR={bucket_stats['win_rate']:.1%}, "
                        f"EV=${bucket_stats['ev']:.2f}")

        return "\n".join(lines)


# Global instance
_win_prob_estimator = None


def get_win_prob_estimator() -> WinProbabilityEstimator:
    """Get or create the global win probability estimator."""
    global _win_prob_estimator
    if _win_prob_estimator is None:
        _win_prob_estimator = WinProbabilityEstimator()
    return _win_prob_estimator


def estimate_win_probability(
    confidence: float,
    direction: str,
    predicted_return: float = 0.0,
    spy_vix_regime: str = 'UNKNOWN',
    trade_amount: float = 50.0
) -> ProbabilityEstimate:
    """
    Convenience function to estimate win probability.

    Usage:
        est = estimate_win_probability(
            confidence=0.6,
            direction='CALL',
            spy_vix_regime='NORMAL'
        )

        if est.should_trade:
            print(f"Trade! P(win)={est.p_win:.1%}, EV=${est.expected_value:.2f}")
        else:
            print(f"Skip: {est.reason}")
    """
    estimator = get_win_prob_estimator()
    return estimator.get_probability(
        confidence, direction, predicted_return, spy_vix_regime, trade_amount
    )


def record_trade_outcome(
    confidence: float,
    direction: str,
    predicted_return: float,
    actual_pnl: float,
    spy_vix_regime: str = 'UNKNOWN',
    rsi: float = 50.0
):
    """
    Record a trade outcome for probability calibration.

    Usage:
        record_trade_outcome(
            confidence=0.6,
            direction='CALL',
            predicted_return=0.001,
            actual_pnl=3.50,  # Won $3.50
            spy_vix_regime='NORMAL'
        )
    """
    estimator = get_win_prob_estimator()
    record = TradeRecord(
        confidence=confidence,
        direction=direction,
        predicted_return=predicted_return,
        actual_pnl=actual_pnl,
        is_win=actual_pnl > 0,
        spy_vix_regime=spy_vix_regime,
        rsi=rsi
    )
    estimator.record_trade(record)
