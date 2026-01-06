"""
Quantor-MTFuzz Facade for Gaussian System
==========================================

Unified interface for all Quantor-MTFuzz features:
- Fuzzy position sizing (9-factor)
- Market regime classification
- Volatility analytics (RV, skew, VRP)

Usage:
    from integrations.quantor import QuantorFacade

    quantor = QuantorFacade()

    # Get position size recommendation
    result = quantor.compute_position_size(
        equity=10000,
        max_loss=500,
        market_data=current_data
    )

    # Check if trading is allowed
    if quantor.should_trade(market_data):
        # Execute trade
        pass
"""

from typing import Dict, Optional, Tuple, Any
from dataclasses import dataclass
import numpy as np

from .fuzzy_sizer import FuzzyPositionSizer, FuzzyMemberships
from .regime_filter import RegimeFilter, MarketRegime
from .volatility import VolatilityAnalyzer, VolatilityMetrics


@dataclass
class QuantorResult:
    """Complete result from Quantor analysis."""
    # Position sizing
    position_size: int
    fuzzy_confidence: float
    memberships: FuzzyMemberships

    # Regime
    regime: MarketRegime
    regime_adjustments: Dict[str, float]
    trading_allowed: bool
    trading_reason: str

    # Volatility
    vol_metrics: VolatilityMetrics
    vol_adjustments: Dict[str, float]

    # Liquidity
    liquidity_ok: bool
    liquidity_reason: str

    # Final recommendation
    recommended_size: int
    recommended_stop_mult: float
    recommended_profit_mult: float
    overall_confidence: float


class QuantorFacade:
    """
    Unified facade for Quantor-MTFuzz features in gaussian-system.

    Combines fuzzy position sizing, regime detection, and volatility
    analytics into a single easy-to-use interface.
    """

    def __init__(
        self,
        # Fuzzy sizer params
        risk_fraction: float = 0.02,
        min_contracts: int = 1,
        max_contracts: int = 10,
        fuzzy_mode: str = 'directional',

        # Regime filter params
        vix_crash_threshold: float = 35.0,
        vix_high_vol_threshold: float = 20.0,
        adx_trend_threshold: float = 25.0,

        # Volatility params
        low_vol_threshold: float = 0.10,
        high_vol_threshold: float = 0.25,
        extreme_vol_threshold: float = 0.40,
        steep_skew_threshold: float = 0.15
    ):
        """
        Initialize Quantor facade with all components.

        Args:
            risk_fraction: Max fraction of equity to risk per trade
            min_contracts: Minimum position size
            max_contracts: Maximum position size
            fuzzy_mode: 'directional' or 'neutral'
            vix_crash_threshold: VIX above which trading blocked
            vix_high_vol_threshold: VIX above which regime is high vol
            adx_trend_threshold: ADX above which market is trending
            low_vol_threshold: RV below this is low vol regime
            high_vol_threshold: RV above this is high vol regime
            extreme_vol_threshold: RV above this is extreme
            steep_skew_threshold: Skew above this indicates crash risk
        """
        self.fuzzy_sizer = FuzzyPositionSizer(
            risk_fraction=risk_fraction,
            min_contracts=min_contracts,
            max_contracts=max_contracts,
            mode=fuzzy_mode
        )

        self.regime_filter = RegimeFilter(
            vix_crash_threshold=vix_crash_threshold,
            vix_high_vol_threshold=vix_high_vol_threshold,
            adx_trend_threshold=adx_trend_threshold
        )

        self.vol_analyzer = VolatilityAnalyzer(
            low_vol_threshold=low_vol_threshold,
            high_vol_threshold=high_vol_threshold,
            extreme_vol_threshold=extreme_vol_threshold,
            steep_skew_threshold=steep_skew_threshold
        )

    def analyze(
        self,
        equity: float,
        max_loss_per_contract: float,
        direction: str = 'CALL',
        # Price data
        close_prices: np.ndarray = None,
        highs: np.ndarray = None,
        lows: np.ndarray = None,
        volumes: list = None,
        current_close: float = None,
        # Technical indicators
        rsi: float = None,
        adx: float = None,
        bb_position: float = None,
        bb_width: float = None,
        stoch_k: float = None,
        sma_200: float = None,
        mtf_consensus: float = None,
        # Volatility data
        vix: float = None,
        implied_vol: float = None,
        iv_put: float = None,
        iv_call: float = None,
        iv_atm: float = None,
        ivr: float = None,  # IV Rank
    ) -> QuantorResult:
        """
        Run complete Quantor analysis and return comprehensive result.

        Args:
            equity: Account equity
            max_loss_per_contract: Maximum loss per contract
            direction: 'CALL' or 'PUT'
            close_prices: Historical close prices (for RV calculation)
            highs: Historical highs (for ATR)
            lows: Historical lows (for ATR)
            volumes: Recent volume values
            current_close: Current price
            rsi: RSI value
            adx: ADX value
            bb_position: Bollinger position (0-1)
            bb_width: Bollinger width
            stoch_k: Stochastic %K
            sma_200: 200-period SMA
            mtf_consensus: Multi-timeframe consensus (-1 to 1)
            vix: Current VIX level
            implied_vol: Current ATM IV
            iv_put: OTM put IV
            iv_call: OTM call IV
            iv_atm: ATM IV
            ivr: IV Rank (percentile)

        Returns:
            QuantorResult with all analysis
        """
        # =====================================================================
        # 1. VOLATILITY ANALYSIS
        # =====================================================================
        if close_prices is not None and len(close_prices) > 20:
            vol_metrics = self.vol_analyzer.analyze(
                close_prices=close_prices,
                highs=highs,
                lows=lows,
                implied_vol=implied_vol,
                iv_put=iv_put,
                iv_call=iv_call,
                iv_atm=iv_atm
            )
        else:
            vol_metrics = VolatilityMetrics()

        vol_adjustments = self.vol_analyzer.get_position_adjustments(vol_metrics)

        # =====================================================================
        # 2. REGIME CLASSIFICATION
        # =====================================================================
        close = current_close or (close_prices[-1] if close_prices is not None and len(close_prices) > 0 else None)

        regime = self.regime_filter.classify_regime(
            vix=vix,
            adx=adx,
            close=close,
            sma_200=sma_200
        )

        regime_adjustments = self.regime_filter.get_regime_adjustments(regime)
        trading_allowed, trading_reason = self.regime_filter.is_trading_allowed(regime, direction)

        # =====================================================================
        # 3. LIQUIDITY CHECK
        # =====================================================================
        if volumes is not None and len(volumes) >= 3:
            liquidity_ok, liquidity_reason = self.regime_filter.check_liquidity(
                volumes=volumes,
                highs=list(highs[-5:]) if highs is not None else None,
                lows=list(lows[-5:]) if lows is not None else None,
                current_close=close
            )
        else:
            liquidity_ok = True
            liquidity_reason = "No volume data - assuming OK"

        # =====================================================================
        # 4. FUZZY MEMBERSHIPS
        # =====================================================================
        volume_ratio = sum(volumes) / len(volumes) / 1000 if volumes else None  # Normalize
        sma_distance = (close - sma_200) / sma_200 if close and sma_200 else None

        memberships = self.fuzzy_sizer.compute_memberships(
            rsi=rsi,
            adx=adx,
            bb_position=bb_position,
            bb_width=bb_width,
            stoch_k=stoch_k,
            volume_ratio=volume_ratio,
            sma_distance=sma_distance,
            mtf_consensus=mtf_consensus,
            ivr=ivr,
            vix=vix,
            direction=direction
        )

        fuzzy_confidence = self.fuzzy_sizer.compute_fuzzy_confidence(memberships)

        # =====================================================================
        # 5. POSITION SIZING
        # =====================================================================
        position_size, debug_info = self.fuzzy_sizer.compute_position_size(
            equity=equity,
            max_loss_per_contract=max_loss_per_contract,
            memberships=memberships,
            realized_vol=vol_metrics.realized_vol,
            debug=True
        )

        # =====================================================================
        # 6. COMBINE ADJUSTMENTS
        # =====================================================================
        # Apply all adjustment factors
        combined_size_mult = (
            regime_adjustments['position_size'] *
            vol_adjustments['size_multiplier']
        )

        recommended_size = max(1, int(position_size * combined_size_mult))

        # Stop/profit multipliers
        recommended_stop_mult = (
            regime_adjustments.get('stop_loss', 1.0) *
            vol_adjustments.get('stop_multiplier', 1.0)
        )

        recommended_profit_mult = (
            regime_adjustments.get('take_profit', 1.0) *
            vol_adjustments.get('profit_multiplier', 1.0)
        )

        # Overall confidence
        overall_confidence = fuzzy_confidence
        if not trading_allowed:
            overall_confidence = 0.0
        if not liquidity_ok:
            overall_confidence *= 0.5
        if vol_metrics.vol_regime == "extreme":
            overall_confidence *= 0.3

        # =====================================================================
        # 7. BUILD RESULT
        # =====================================================================
        return QuantorResult(
            # Position sizing
            position_size=position_size,
            fuzzy_confidence=fuzzy_confidence,
            memberships=memberships,

            # Regime
            regime=regime,
            regime_adjustments=regime_adjustments,
            trading_allowed=trading_allowed,
            trading_reason=trading_reason,

            # Volatility
            vol_metrics=vol_metrics,
            vol_adjustments=vol_adjustments,

            # Liquidity
            liquidity_ok=liquidity_ok,
            liquidity_reason=liquidity_reason,

            # Final recommendations
            recommended_size=recommended_size,
            recommended_stop_mult=recommended_stop_mult,
            recommended_profit_mult=recommended_profit_mult,
            overall_confidence=overall_confidence
        )

    def should_trade(
        self,
        vix: float = None,
        adx: float = None,
        close: float = None,
        sma_200: float = None,
        direction: str = None,
        volumes: list = None,
        highs: list = None,
        lows: list = None
    ) -> Tuple[bool, str]:
        """
        Quick check if trading is allowed based on regime and liquidity.

        Returns:
            Tuple of (should_trade, reason)
        """
        regime = self.regime_filter.classify_regime(vix, adx, close, sma_200)
        trading_allowed, trading_reason = self.regime_filter.is_trading_allowed(regime, direction)

        if not trading_allowed:
            return False, trading_reason

        if volumes is not None:
            liquidity_ok, liquidity_reason = self.regime_filter.check_liquidity(
                volumes, highs, lows, close
            )
            if not liquidity_ok:
                return False, f"Liquidity: {liquidity_reason}"

        return True, f"Trading allowed: {regime.value}"

    def compute_position_size(
        self,
        equity: float,
        max_loss: float,
        market_data: Dict[str, Any]
    ) -> Tuple[int, Dict]:
        """
        Simplified position sizing interface.

        Args:
            equity: Account equity
            max_loss: Maximum loss per contract
            market_data: Dict with keys: rsi, adx, bb_position, stoch_k,
                        vix, mtf_consensus, ivr, direction, etc.

        Returns:
            Tuple of (position_size, debug_info)
        """
        direction = market_data.get('direction', 'CALL')

        memberships = self.fuzzy_sizer.compute_memberships(
            rsi=market_data.get('rsi'),
            adx=market_data.get('adx'),
            bb_position=market_data.get('bb_position'),
            bb_width=market_data.get('bb_width'),
            stoch_k=market_data.get('stoch_k'),
            volume_ratio=market_data.get('volume_ratio'),
            sma_distance=market_data.get('sma_distance'),
            mtf_consensus=market_data.get('mtf_consensus'),
            ivr=market_data.get('ivr'),
            vix=market_data.get('vix'),
            direction=direction
        )

        return self.fuzzy_sizer.compute_position_size(
            equity=equity,
            max_loss_per_contract=max_loss,
            memberships=memberships,
            realized_vol=market_data.get('realized_vol'),
            debug=True
        )

    def get_regime(
        self,
        vix: float,
        adx: float = None,
        close: float = None,
        sma_200: float = None
    ) -> MarketRegime:
        """Get current market regime."""
        return self.regime_filter.classify_regime(vix, adx, close, sma_200)

    def get_direction_bias(
        self,
        vix: float,
        adx: float = None,
        close: float = None,
        sma_200: float = None
    ) -> float:
        """Get directional bias (-1 bearish to +1 bullish) based on regime."""
        regime = self.regime_filter.classify_regime(vix, adx, close, sma_200)
        return self.regime_filter.get_direction_bias(regime)
