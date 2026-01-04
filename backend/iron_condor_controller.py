#!/usr/bin/env python3
"""
Iron Condor Entry Controller - Adapter for Jerry's Quantor-MTFuzz strategy

This controller adapts Jerry's Iron Condor strategy to work within
the gaussian-system's time-travel testing framework.

Key differences from directional strategies:
1. Iron Condors are delta-neutral (profit from range-bound markets)
2. We WANT low volatility and sideways movement
3. Entry signals are inverted from directional strategies
   - High confidence in any direction = BAD for condors
   - Low confidence / neutral = GOOD for condors
"""

import os
import sys
import logging
import numpy as np
from typing import Dict, Tuple, Optional, List
from dataclasses import dataclass
from datetime import datetime

# Add jerry-bot to path
jerry_bot_path = os.path.join(os.path.dirname(__file__), '..', 'jerry-bot')
sys.path.insert(0, jerry_bot_path)

logger = logging.getLogger(__name__)


@dataclass
class IronCondorSignal:
    """Signal for Iron Condor entry"""
    should_enter: bool
    wing_width: float
    expected_credit: float
    iv_rank: float
    vix_level: float
    mtf_consensus: float
    reasoning: str


class IronCondorEntryController:
    """
    Entry controller that uses Jerry's Iron Condor logic
    adapted for the gaussian-system testing framework.

    Instead of predicting direction, this controller:
    1. Looks for range-bound market conditions (neutral MTF)
    2. Requires adequate IV for premium collection
    3. Filters out high-volatility environments
    """

    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}

        # Iron Condor specific thresholds
        self.iv_rank_min = self.config.get('iv_rank_min', 20.0)
        self.vix_max = self.config.get('vix_max', 25.0)
        self.mtf_consensus_min = self.config.get('mtf_consensus_min', 0.40)
        self.mtf_consensus_max = self.config.get('mtf_consensus_max', 0.60)

        # Wing width parameters
        self.base_wing_width = self.config.get('base_wing_width', 5.0)
        self.max_wing_width = self.config.get('max_wing_width', 10.0)
        self.vix_widen_threshold = self.config.get('vix_widen_threshold', 22.0)

        # Position management
        self.profit_take_pct = self.config.get('profit_take_pct', 0.50)  # 50% of credit
        self.loss_multiple = self.config.get('loss_multiple', 2.0)  # 2x credit max loss
        self.max_hold_days = self.config.get('max_hold_days', 14)

        # Tracking
        self.decisions = 0
        self.entries = 0
        self.rejections_by_reason = {}

        logger.info(f"[IRON CONDOR] Initialized with IVR>{self.iv_rank_min}, VIX<{self.vix_max}, MTF={self.mtf_consensus_min}-{self.mtf_consensus_max}")

    def _calculate_mtf_consensus(self, features: Dict) -> float:
        """
        Calculate MTF consensus from available features.
        For Iron Condors, we want NEUTRAL consensus (0.4-0.6).
        """
        # Use HMM trend as proxy for MTF consensus
        hmm_trend = features.get('hmm_trend', 0.5)

        # Also incorporate momentum if available
        momentum_5m = features.get('momentum_5m', 0.0)
        momentum_15m = features.get('momentum_15m', 0.0)

        # Convert momentum to 0-1 scale (0.5 = neutral)
        mom_signal = 0.5 + np.clip(momentum_5m * 10, -0.3, 0.3)

        # Weighted average: HMM is primary, momentum is secondary
        consensus = 0.7 * hmm_trend + 0.3 * mom_signal

        return float(np.clip(consensus, 0.0, 1.0))

    def _estimate_iv_rank(self, features: Dict) -> float:
        """
        Estimate IV Rank from available features.
        IV Rank = (current_iv - 52w_low) / (52w_high - 52w_low) * 100
        """
        vix = features.get('vix_level', 18.0)

        # Approximate IV rank from VIX (rough mapping)
        # VIX 12-15 = low IVR (10-25), VIX 15-20 = mid (25-50), VIX 20-30 = high (50-80)
        if vix < 15:
            iv_rank = 10 + (vix - 12) * 5
        elif vix < 20:
            iv_rank = 25 + (vix - 15) * 5
        elif vix < 30:
            iv_rank = 50 + (vix - 20) * 3
        else:
            iv_rank = 80 + min(20, (vix - 30))

        return float(np.clip(iv_rank, 0, 100))

    def _calculate_wing_width(self, vix: float, iv_rank: float) -> float:
        """
        Dynamic wing width based on volatility regime.
        Wider wings in high vol = more premium but more risk.
        """
        width = self.base_wing_width

        if vix >= self.vix_widen_threshold or iv_rank >= 40:
            width += 2.5
        if vix >= 30 or iv_rank >= 60:
            width += 2.5

        return min(width, self.max_wing_width)

    def _estimate_credit(self, wing_width: float, iv_rank: float, vix: float) -> float:
        """
        Estimate credit received for Iron Condor.
        This is a simplified model - real credit depends on exact strikes.
        """
        # Base credit as percentage of wing width
        # Higher IV = higher premium
        base_credit_pct = 0.20 + (iv_rank / 100) * 0.15  # 20-35% of width

        # VIX adjustment
        vix_adj = 1.0 + (vix - 18) * 0.02  # Higher VIX = more premium

        credit = wing_width * base_credit_pct * vix_adj

        return round(credit, 2)

    def should_enter_condor(self, features: Dict) -> IronCondorSignal:
        """
        Evaluate whether to enter an Iron Condor position.

        Args:
            features: Dict with market features including:
                - vix_level: Current VIX
                - hmm_trend: HMM trend state (0-1)
                - momentum_5m, momentum_15m: Momentum indicators
                - confidence: Model confidence (we want LOW for condors!)

        Returns:
            IronCondorSignal with entry decision and parameters
        """
        self.decisions += 1

        # Extract features
        vix = features.get('vix_level', 18.0)
        iv_rank = self._estimate_iv_rank(features)
        mtf_consensus = self._calculate_mtf_consensus(features)
        model_confidence = features.get('confidence', 0.5)

        reasoning_parts = []

        # === Filter 1: VIX Threshold ===
        if vix > self.vix_max:
            reason = f"VIX {vix:.1f} > {self.vix_max}"
            self.rejections_by_reason[reason] = self.rejections_by_reason.get(reason, 0) + 1
            return IronCondorSignal(
                should_enter=False,
                wing_width=0, expected_credit=0,
                iv_rank=iv_rank, vix_level=vix, mtf_consensus=mtf_consensus,
                reasoning=f"REJECT: {reason}"
            )
        reasoning_parts.append(f"VIX={vix:.1f} OK")

        # === Filter 2: IV Rank Minimum ===
        if iv_rank < self.iv_rank_min:
            reason = f"IVR {iv_rank:.1f} < {self.iv_rank_min}"
            self.rejections_by_reason[reason] = self.rejections_by_reason.get(reason, 0) + 1
            return IronCondorSignal(
                should_enter=False,
                wing_width=0, expected_credit=0,
                iv_rank=iv_rank, vix_level=vix, mtf_consensus=mtf_consensus,
                reasoning=f"REJECT: {reason}"
            )
        reasoning_parts.append(f"IVR={iv_rank:.1f} OK")

        # === Filter 3: MTF Consensus (want NEUTRAL) ===
        if mtf_consensus < self.mtf_consensus_min:
            reason = f"MTF {mtf_consensus:.2f} < {self.mtf_consensus_min} (too bearish)"
            self.rejections_by_reason[reason] = self.rejections_by_reason.get(reason, 0) + 1
            return IronCondorSignal(
                should_enter=False,
                wing_width=0, expected_credit=0,
                iv_rank=iv_rank, vix_level=vix, mtf_consensus=mtf_consensus,
                reasoning=f"REJECT: {reason}"
            )
        if mtf_consensus > self.mtf_consensus_max:
            reason = f"MTF {mtf_consensus:.2f} > {self.mtf_consensus_max} (too bullish)"
            self.rejections_by_reason[reason] = self.rejections_by_reason.get(reason, 0) + 1
            return IronCondorSignal(
                should_enter=False,
                wing_width=0, expected_credit=0,
                iv_rank=iv_rank, vix_level=vix, mtf_consensus=mtf_consensus,
                reasoning=f"REJECT: {reason}"
            )
        reasoning_parts.append(f"MTF={mtf_consensus:.2f} NEUTRAL")

        # === Filter 4: Model Confidence (want LOW for condors!) ===
        # High confidence in direction = market moving = bad for condors
        if model_confidence > 0.75:
            reason = f"Confidence {model_confidence:.2f} too high (directional market)"
            self.rejections_by_reason[reason] = self.rejections_by_reason.get(reason, 0) + 1
            return IronCondorSignal(
                should_enter=False,
                wing_width=0, expected_credit=0,
                iv_rank=iv_rank, vix_level=vix, mtf_consensus=mtf_consensus,
                reasoning=f"REJECT: {reason}"
            )
        reasoning_parts.append(f"Conf={model_confidence:.2f} OK (low=good)")

        # === All filters passed - Calculate position ===
        wing_width = self._calculate_wing_width(vix, iv_rank)
        expected_credit = self._estimate_credit(wing_width, iv_rank, vix)

        self.entries += 1
        reasoning_parts.append(f"Width=${wing_width} Credit=${expected_credit:.2f}")

        return IronCondorSignal(
            should_enter=True,
            wing_width=wing_width,
            expected_credit=expected_credit,
            iv_rank=iv_rank,
            vix_level=vix,
            mtf_consensus=mtf_consensus,
            reasoning="ENTER: " + " | ".join(reasoning_parts)
        )

    def get_direction_for_gaussian(self, features: Dict) -> Tuple[bool, str, str]:
        """
        Adapter method to match gaussian-system entry controller interface.

        Returns:
            (should_trade, direction, reasoning)

        For Iron Condors:
        - direction is always "CONDOR" (delta-neutral)
        - should_trade based on condor entry logic
        """
        signal = self.should_enter_condor(features)

        if signal.should_enter:
            return True, "CONDOR", signal.reasoning
        else:
            return False, "HOLD", signal.reasoning

    def simulate_condor_pnl(self, entry_credit: float, wing_width: float,
                           spot_move_pct: float, days_held: int,
                           entry_vix: float, exit_vix: float) -> Tuple[float, str]:
        """
        Simulate Iron Condor P&L based on spot movement.

        Iron Condors profit when price stays within the wings.
        Loss occurs when price moves beyond short strikes.

        Returns:
            (pnl_dollars, exit_reason)
        """
        # Estimate short strike distance (typically 1-2 standard deviations)
        # Short strikes are typically at 15-20 delta = ~1.5 sigma
        short_strike_distance_pct = 0.03  # ~3% from spot for 15-delta

        # Theta decay (time value erosion benefits condors)
        theta_per_day = entry_credit * 0.03  # ~3% of credit per day
        theta_gain = theta_per_day * days_held

        # Vega impact (IV change affects condor value)
        vega_impact = (entry_vix - exit_vix) * entry_credit * 0.02

        # Check if breached
        abs_move = abs(spot_move_pct)

        if abs_move < short_strike_distance_pct * 0.5:
            # Well within range - collect most of premium
            pnl = entry_credit * self.profit_take_pct + theta_gain + vega_impact
            return pnl * 100, "PROFIT_TAKE"

        elif abs_move < short_strike_distance_pct:
            # Near short strike - reduced profit
            proximity_factor = 1 - (abs_move / short_strike_distance_pct)
            pnl = entry_credit * proximity_factor * 0.3 + theta_gain * 0.5
            return pnl * 100, "PARTIAL_PROFIT"

        elif abs_move < short_strike_distance_pct + (wing_width / 100):
            # Between short and long strike - losing
            breach_depth = (abs_move - short_strike_distance_pct) / (wing_width / 100)
            max_loss = wing_width - entry_credit
            pnl = -max_loss * breach_depth

            if pnl < -entry_credit * self.loss_multiple:
                return -entry_credit * self.loss_multiple * 100, "STOP_LOSS"
            return pnl * 100, "PARTIAL_LOSS"

        else:
            # Beyond long strike - max loss
            max_loss = wing_width - entry_credit
            return -max_loss * 100, "MAX_LOSS"

    def get_stats(self) -> Dict:
        """Return controller statistics"""
        return {
            'controller': 'iron_condor',
            'decisions': self.decisions,
            'entries': self.entries,
            'entry_rate': self.entries / max(1, self.decisions),
            'rejections': self.rejections_by_reason,
            'config': {
                'iv_rank_min': self.iv_rank_min,
                'vix_max': self.vix_max,
                'mtf_range': f"{self.mtf_consensus_min}-{self.mtf_consensus_max}",
                'base_width': self.base_wing_width,
                'profit_take': self.profit_take_pct,
                'loss_multiple': self.loss_multiple
            }
        }


def create_iron_condor_controller(config: Optional[Dict] = None) -> IronCondorEntryController:
    """Factory function to create Iron Condor controller"""
    return IronCondorEntryController(config)


# === Test ===
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    controller = IronCondorEntryController()

    # Test scenarios
    test_cases = [
        # Good for condors: low VIX, adequate IV, neutral market
        {'vix_level': 16.0, 'hmm_trend': 0.5, 'momentum_5m': 0.001, 'confidence': 0.4},
        # Bad: VIX too high
        {'vix_level': 28.0, 'hmm_trend': 0.5, 'momentum_5m': 0.0, 'confidence': 0.3},
        # Bad: Market trending (high MTF)
        {'vix_level': 18.0, 'hmm_trend': 0.85, 'momentum_5m': 0.05, 'confidence': 0.6},
        # Bad: High directional confidence
        {'vix_level': 18.0, 'hmm_trend': 0.5, 'momentum_5m': 0.0, 'confidence': 0.85},
        # Good: Perfect condor conditions
        {'vix_level': 20.0, 'hmm_trend': 0.52, 'momentum_5m': -0.002, 'confidence': 0.35},
    ]

    print("=" * 70)
    print("IRON CONDOR ENTRY CONTROLLER TEST")
    print("=" * 70)

    for i, features in enumerate(test_cases, 1):
        signal = controller.should_enter_condor(features)
        print(f"\nTest {i}: {features}")
        print(f"  Result: {'ENTER' if signal.should_enter else 'REJECT'}")
        print(f"  {signal.reasoning}")
        if signal.should_enter:
            print(f"  Wing Width: ${signal.wing_width}, Expected Credit: ${signal.expected_credit}")

    print("\n" + "=" * 70)
    print("CONTROLLER STATS:")
    print(controller.get_stats())
