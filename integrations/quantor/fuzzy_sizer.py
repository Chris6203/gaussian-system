"""
Fuzzy Position Sizing for Gaussian System
=========================================

Adapted from Quantor-MTFuzz fuzzy_engine.py
Original: 9-factor membership functions for Iron Condor sizing
Modified: Directional options (CALL/PUT) support for gaussian-system

Key Concepts:
- Stage 1: Hard ceiling based on risk fraction
- Stage 2: Fuzzy confidence from 9 technical factors
- Stage 3: Volatility penalty
- Stage 4: Final scaled position size
"""

import numpy as np
from typing import Dict, Optional, Tuple
from dataclasses import dataclass


@dataclass
class FuzzyMemberships:
    """Container for all membership scores (0.0 to 1.0)"""
    mtf_alignment: float = 0.5      # Multi-timeframe consensus
    iv_rank: float = 0.5            # IV percentile favorability
    regime: float = 0.5             # VIX-based regime stability
    rsi: float = 0.5                # RSI neutral zone
    adx: float = 0.5                # Trend strength (low = good for range)
    bbands: float = 0.5             # Bollinger position + squeeze
    stochastic: float = 0.5         # Stochastic neutral zone
    volume: float = 0.5             # Volume confirmation
    sma_distance: float = 0.5       # Price equilibrium


# Default weights for directional trading (different from Iron Condor)
DEFAULT_WEIGHTS_DIRECTIONAL = {
    'mtf_alignment': 0.20,    # Multi-timeframe agreement important
    'iv_rank': 0.10,          # IV matters for premium
    'regime': 0.15,           # Avoid crash mode
    'rsi': 0.10,              # Momentum confirmation
    'adx': 0.15,              # Trend strength (higher = better for directional)
    'bbands': 0.10,           # Volatility regime
    'stochastic': 0.05,       # Secondary momentum
    'volume': 0.10,           # Liquidity
    'sma_distance': 0.05      # Equilibrium
}

# Original Iron Condor weights (prefer neutral/range-bound)
DEFAULT_WEIGHTS_NEUTRAL = {
    'mtf_alignment': 0.15,
    'iv_rank': 0.15,
    'regime': 0.15,
    'rsi': 0.10,
    'adx': 0.15,              # Low ADX = good for neutral
    'bbands': 0.10,
    'stochastic': 0.05,
    'volume': 0.10,
    'sma_distance': 0.05
}


class FuzzyPositionSizer:
    """
    Fuzzy logic position sizing adapted for gaussian-system.

    Computes position size using 9-factor fuzzy membership functions
    that evaluate market conditions and scale position accordingly.
    """

    def __init__(
        self,
        risk_fraction: float = 0.02,
        min_contracts: int = 1,
        max_contracts: int = 10,
        weights: Optional[Dict[str, float]] = None,
        mode: str = 'directional'  # 'directional' or 'neutral'
    ):
        """
        Initialize fuzzy position sizer.

        Args:
            risk_fraction: Max fraction of equity to risk per trade (default 2%)
            min_contracts: Minimum position size
            max_contracts: Maximum position size
            weights: Custom membership weights (default based on mode)
            mode: 'directional' for CALL/PUT, 'neutral' for spreads
        """
        self.risk_fraction = risk_fraction
        self.min_contracts = min_contracts
        self.max_contracts = max_contracts
        self.mode = mode

        if weights:
            self.weights = weights
        else:
            self.weights = (DEFAULT_WEIGHTS_DIRECTIONAL if mode == 'directional'
                          else DEFAULT_WEIGHTS_NEUTRAL)

    # =========================================================================
    # MEMBERSHIP FUNCTIONS
    # =========================================================================

    def calculate_rsi_membership(
        self,
        rsi: float,
        direction: str = 'CALL',
        neutral_min: float = 40.0,
        neutral_max: float = 60.0
    ) -> float:
        """
        RSI membership adapted for directional trading.

        For directional trades:
        - CALL: Higher RSI (momentum) can be good, but not overbought
        - PUT: Lower RSI can be good, but not oversold

        For neutral: Middle range (40-60) is ideal.
        """
        if rsi is None or np.isnan(rsi):
            return 0.3

        if self.mode == 'neutral':
            # Original Iron Condor logic: prefer neutral
            if neutral_min <= rsi <= neutral_max:
                return 1.0
            elif rsi < neutral_min:
                return max(0.0, rsi / neutral_min)
            else:
                return max(0.0, (100 - rsi) / (100 - neutral_max))
        else:
            # Directional: momentum confirmation
            if direction == 'CALL':
                # CALL: RSI 50-70 is ideal (bullish momentum, not overbought)
                if 50 <= rsi <= 70:
                    return 1.0
                elif rsi < 50:
                    return max(0.3, rsi / 50)
                else:  # > 70 (overbought)
                    return max(0.2, (100 - rsi) / 30)
            else:
                # PUT: RSI 30-50 is ideal (bearish momentum, not oversold)
                if 30 <= rsi <= 50:
                    return 1.0
                elif rsi > 50:
                    return max(0.3, (100 - rsi) / 50)
                else:  # < 30 (oversold)
                    return max(0.2, rsi / 30)

    def calculate_adx_membership(
        self,
        adx: float,
        threshold_low: float = 25.0,
        threshold_high: float = 40.0
    ) -> float:
        """
        ADX membership - trend strength indicator.

        For directional: Higher ADX (strong trend) is GOOD
        For neutral: Lower ADX (weak trend) is GOOD
        """
        if adx is None or np.isnan(adx):
            return 0.5

        if self.mode == 'neutral':
            # Original: weak trend ideal for Iron Condor
            if adx <= threshold_low:
                return 1.0
            elif adx <= threshold_high:
                return 1.0 - ((adx - threshold_low) / (threshold_high - threshold_low))
            else:
                return 0.0
        else:
            # Directional: strong trend is good
            if adx >= threshold_high:
                return 1.0
            elif adx >= threshold_low:
                return (adx - threshold_low) / (threshold_high - threshold_low)
            else:
                return max(0.2, adx / threshold_low)

    def calculate_bbands_membership(
        self,
        bb_position: float,
        bb_width: float = None,
        direction: str = 'CALL'
    ) -> float:
        """
        Bollinger Bands membership.

        bb_position: 0.0 = lower band, 0.5 = middle, 1.0 = upper band

        For directional:
        - CALL: Price near lower band (bounce) or breaking upper (momentum)
        - PUT: Price near upper band (rejection) or breaking lower

        For neutral: Middle is ideal.
        """
        if bb_position is None or np.isnan(bb_position):
            return 0.5

        if self.mode == 'neutral':
            # Prefer middle (0.5 = perfect)
            return max(0.0, 1.0 - abs(bb_position - 0.5) * 2)
        else:
            if direction == 'CALL':
                # CALL: lower half of bands is opportunity
                if bb_position < 0.3:
                    return 1.0  # Near lower band - bounce opportunity
                elif bb_position < 0.5:
                    return 0.8
                elif bb_position > 0.9:
                    return 0.4  # Overbought caution
                else:
                    return 0.6
            else:
                # PUT: upper half of bands is opportunity
                if bb_position > 0.7:
                    return 1.0  # Near upper band - rejection opportunity
                elif bb_position > 0.5:
                    return 0.8
                elif bb_position < 0.1:
                    return 0.4  # Oversold caution
                else:
                    return 0.6

    def calculate_stoch_membership(
        self,
        stoch_k: float,
        direction: str = 'CALL',
        neutral_min: float = 30.0,
        neutral_max: float = 70.0
    ) -> float:
        """Stochastic membership (similar logic to RSI)."""
        if stoch_k is None or np.isnan(stoch_k):
            return 0.5

        if self.mode == 'neutral':
            if neutral_min <= stoch_k <= neutral_max:
                return 1.0
            elif stoch_k < neutral_min:
                return max(0.0, stoch_k / neutral_min)
            else:
                return max(0.0, (100 - stoch_k) / (100 - neutral_max))
        else:
            # Directional: use for momentum confirmation
            if direction == 'CALL':
                if stoch_k < 30:
                    return 1.0  # Oversold - good for calls
                elif stoch_k < 50:
                    return 0.8
                elif stoch_k > 80:
                    return 0.3  # Overbought
                else:
                    return 0.5
            else:
                if stoch_k > 70:
                    return 1.0  # Overbought - good for puts
                elif stoch_k > 50:
                    return 0.8
                elif stoch_k < 20:
                    return 0.3  # Oversold
                else:
                    return 0.5

    def calculate_volume_membership(
        self,
        volume_ratio: float,
        min_ratio: float = 0.8
    ) -> float:
        """Volume confirmation - higher volume = better fills."""
        if volume_ratio is None or np.isnan(volume_ratio):
            return 0.5
        return min(1.0, max(0.0, volume_ratio / min_ratio))

    def calculate_sma_distance_membership(
        self,
        sma_distance: float,
        max_distance: float = 0.02
    ) -> float:
        """SMA distance - price equilibrium check."""
        if sma_distance is None or np.isnan(sma_distance):
            return 0.5
        abs_dist = abs(sma_distance)
        if abs_dist <= max_distance:
            return 1.0 - (abs_dist / max_distance)
        return 0.0

    def calculate_mtf_membership(
        self,
        mtf_consensus: float
    ) -> float:
        """
        Multi-timeframe alignment.

        mtf_consensus: -1.0 (all bearish) to +1.0 (all bullish)
        For directional, we want alignment with direction.
        """
        if mtf_consensus is None or np.isnan(mtf_consensus):
            return 0.5

        # Convert from [-1, 1] to [0, 1]
        return (mtf_consensus + 1.0) / 2.0

    def calculate_iv_membership(self, ivr: float) -> float:
        """IV Rank membership - higher IV = more premium."""
        if ivr is None or np.isnan(ivr):
            return 0.5
        return min(1.0, ivr / 60.0)

    def calculate_regime_membership(
        self,
        vix: float,
        threshold: float = 20.0
    ) -> float:
        """VIX-based regime stability."""
        if vix is None or np.isnan(vix):
            return 0.5
        if vix <= 12.0:
            return 1.0
        penalty = (vix - 12.0) / 18.0
        return max(0.0, 1.0 - penalty)

    # =========================================================================
    # MAIN SIZING LOGIC
    # =========================================================================

    def compute_memberships(
        self,
        rsi: float = None,
        adx: float = None,
        bb_position: float = None,
        bb_width: float = None,
        stoch_k: float = None,
        volume_ratio: float = None,
        sma_distance: float = None,
        mtf_consensus: float = None,
        ivr: float = None,
        vix: float = None,
        direction: str = 'CALL'
    ) -> FuzzyMemberships:
        """Compute all membership values from market data."""
        return FuzzyMemberships(
            mtf_alignment=self.calculate_mtf_membership(mtf_consensus),
            iv_rank=self.calculate_iv_membership(ivr),
            regime=self.calculate_regime_membership(vix),
            rsi=self.calculate_rsi_membership(rsi, direction),
            adx=self.calculate_adx_membership(adx),
            bbands=self.calculate_bbands_membership(bb_position, bb_width, direction),
            stochastic=self.calculate_stoch_membership(stoch_k, direction),
            volume=self.calculate_volume_membership(volume_ratio),
            sma_distance=self.calculate_sma_distance_membership(sma_distance)
        )

    def compute_fuzzy_confidence(self, memberships: FuzzyMemberships) -> float:
        """Compute weighted fuzzy confidence score."""
        m_dict = {
            'mtf_alignment': memberships.mtf_alignment,
            'iv_rank': memberships.iv_rank,
            'regime': memberships.regime,
            'rsi': memberships.rsi,
            'adx': memberships.adx,
            'bbands': memberships.bbands,
            'stochastic': memberships.stochastic,
            'volume': memberships.volume,
            'sma_distance': memberships.sma_distance
        }

        confidence = sum(self.weights.get(k, 0) * v for k, v in m_dict.items())
        return max(0.0, min(1.0, confidence))

    def compute_volatility_penalty(
        self,
        realized_vol: float,
        low_vol: float = 0.10,
        high_vol: float = 0.30
    ) -> float:
        """
        Compute volatility penalty (sigma_star).
        Higher volatility = higher penalty = smaller position.
        """
        if realized_vol is None or np.isnan(realized_vol):
            return 0.5

        if high_vol <= low_vol:
            return 1.0

        sigma_star = (realized_vol - low_vol) / (high_vol - low_vol)
        return max(0.0, min(1.0, sigma_star))

    def compute_position_size(
        self,
        equity: float,
        max_loss_per_contract: float,
        memberships: FuzzyMemberships,
        realized_vol: float = None,
        debug: bool = False
    ) -> Tuple[int, Dict]:
        """
        Compute final position size using fuzzy logic pipeline.

        Args:
            equity: Account equity
            max_loss_per_contract: Maximum loss per contract
            memberships: Pre-computed membership scores
            realized_vol: Realized volatility for penalty
            debug: Return debug info

        Returns:
            Tuple of (position_size, debug_info)
        """
        # Stage 1: Hard ceiling
        if equity <= 0 or max_loss_per_contract <= 0:
            return (0, {}) if debug else 0

        max_risk = self.risk_fraction * equity
        q0 = int(max_risk // max_loss_per_contract)
        q0 = min(q0, self.max_contracts)

        if q0 < self.min_contracts:
            return (0, {'reason': 'below_minimum'}) if debug else 0

        # Stage 2: Fuzzy confidence
        confidence = self.compute_fuzzy_confidence(memberships)

        # Stage 3: Volatility penalty
        vol_penalty = self.compute_volatility_penalty(realized_vol) if realized_vol else 0.0

        # Stage 4: Scaling factor g = confidence * (1 - vol_penalty)
        scaling = confidence * (1.0 - vol_penalty)
        scaling = max(0.0, min(1.0, scaling))

        # Stage 5: Final size
        q_final = int(q0 * scaling)
        q_final = max(self.min_contracts, min(q_final, self.max_contracts))

        # Handle edge case: if scaling makes it 0, use minimum
        if q_final < self.min_contracts and q0 >= self.min_contracts:
            q_final = self.min_contracts

        debug_info = {
            'base_qty': q0,
            'confidence': confidence,
            'vol_penalty': vol_penalty,
            'scaling': scaling,
            'final_qty': q_final,
            'memberships': memberships
        }

        if debug:
            return (q_final, debug_info)
        return q_final
