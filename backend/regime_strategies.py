#!/usr/bin/env python3
"""
Regime-Specific Trading Strategies
==================================

Adjusts trading thresholds and behavior based on market regime:
- Low volatility: Lower thresholds, more trades, tighter stops
- High volatility: Higher thresholds, fewer trades, wider stops
- Trending: Follow trend, looser trailing stops
- Mean-reverting/Choppy: Fade moves, quick exits

Integrates with:
- Multi-Dimensional HMM for regime detection
- VIX levels as volatility proxy
- Calibration system for confidence adjustment

Usage:
    from backend.regime_strategies import RegimeStrategyManager
    
    manager = RegimeStrategyManager()
    
    # Get regime-adjusted parameters
    params = manager.get_trading_params(
        vix_level=15.0,
        hmm_trend_state=0,
        hmm_vol_state=1
    )
    
    # Check if should trade in current regime
    if manager.should_trade(signal, params):
        execute_trade(signal)
"""

import logging
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class VolatilityRegime(Enum):
    """Volatility regime classification."""
    ULTRA_LOW = "ultra_low"      # VIX < 12
    LOW = "low"                   # VIX 12-15
    NORMAL = "normal"             # VIX 15-20
    ELEVATED = "elevated"         # VIX 20-25
    HIGH = "high"                 # VIX 25-35
    EXTREME = "extreme"           # VIX > 35


class TrendRegime(Enum):
    """Trend regime classification."""
    STRONG_UP = "strong_up"
    MODERATE_UP = "moderate_up"
    SIDEWAYS = "sideways"
    MODERATE_DOWN = "moderate_down"
    STRONG_DOWN = "strong_down"


class LiquidityRegime(Enum):
    """Liquidity regime classification."""
    HIGH = "high"
    NORMAL = "normal"
    LOW = "low"
    ILLIQUID = "illiquid"


@dataclass
class RegimeParams:
    """Trading parameters for a specific regime."""
    # Confidence thresholds
    min_confidence: float = 0.55
    entry_threshold: float = 0.55
    exit_threshold: float = 0.40
    
    # Position sizing
    position_scale: float = 1.0
    max_positions: int = 5
    
    # Risk management
    stop_loss_pct: float = 0.20
    take_profit_pct: float = 0.40
    trailing_stop_pct: float = 0.15
    max_hold_minutes: int = 120
    
    # Trade frequency
    max_trades_per_hour: int = 5
    min_time_between_trades: int = 5  # minutes
    
    # Calibration adjustments
    require_calibration: bool = True
    max_brier_score: float = 0.35
    max_ece: float = 0.15
    
    # Regime-specific flags
    allow_counter_trend: bool = True
    require_multi_tf_agreement: bool = True
    
    # Description
    regime_name: str = ""
    description: str = ""


# Pre-defined regime parameters
REGIME_PARAMS = {
    # ==========================================================================
    # VOLATILITY-BASED REGIMES
    # ==========================================================================
    
    VolatilityRegime.ULTRA_LOW: RegimeParams(
        min_confidence=0.45,          # LOWER threshold - need more trades
        entry_threshold=0.45,
        exit_threshold=0.35,
        position_scale=1.2,           # Slightly larger positions
        max_positions=6,
        stop_loss_pct=0.15,           # Tighter stops (moves are small)
        take_profit_pct=0.25,         # Smaller targets
        trailing_stop_pct=0.10,
        max_hold_minutes=90,          # Shorter holds
        max_trades_per_hour=8,        # More trades allowed
        min_time_between_trades=3,
        require_calibration=True,
        max_brier_score=0.40,         # Slightly relaxed
        require_multi_tf_agreement=False,  # Don't need agreement in calm markets
        regime_name="Ultra-Low Vol",
        description="VIX < 12: Aggressive trading, tight stops, quick profits"
    ),
    
    VolatilityRegime.LOW: RegimeParams(
        min_confidence=0.50,
        entry_threshold=0.50,
        exit_threshold=0.38,
        position_scale=1.1,
        max_positions=5,
        stop_loss_pct=0.18,
        take_profit_pct=0.30,
        trailing_stop_pct=0.12,
        max_hold_minutes=100,
        max_trades_per_hour=6,
        min_time_between_trades=4,
        require_multi_tf_agreement=False,
        regime_name="Low Vol",
        description="VIX 12-15: Active trading, moderate risk"
    ),
    
    VolatilityRegime.NORMAL: RegimeParams(
        min_confidence=0.55,
        entry_threshold=0.55,
        exit_threshold=0.40,
        position_scale=1.0,
        max_positions=5,
        stop_loss_pct=0.20,
        take_profit_pct=0.40,
        trailing_stop_pct=0.15,
        max_hold_minutes=120,
        max_trades_per_hour=5,
        min_time_between_trades=5,
        regime_name="Normal Vol",
        description="VIX 15-20: Standard trading parameters"
    ),
    
    VolatilityRegime.ELEVATED: RegimeParams(
        min_confidence=0.60,
        entry_threshold=0.60,
        exit_threshold=0.45,
        position_scale=0.8,
        max_positions=4,
        stop_loss_pct=0.25,
        take_profit_pct=0.50,
        trailing_stop_pct=0.18,
        max_hold_minutes=90,
        max_trades_per_hour=4,
        min_time_between_trades=8,
        require_multi_tf_agreement=True,
        regime_name="Elevated Vol",
        description="VIX 20-25: Cautious trading, wider stops"
    ),
    
    VolatilityRegime.HIGH: RegimeParams(
        min_confidence=0.65,
        entry_threshold=0.65,
        exit_threshold=0.50,
        position_scale=0.5,
        max_positions=3,
        stop_loss_pct=0.30,
        take_profit_pct=0.60,
        trailing_stop_pct=0.22,
        max_hold_minutes=60,
        max_trades_per_hour=3,
        min_time_between_trades=10,
        require_multi_tf_agreement=True,
        allow_counter_trend=False,
        regime_name="High Vol",
        description="VIX 25-35: Conservative trading, trend-only"
    ),
    
    VolatilityRegime.EXTREME: RegimeParams(
        min_confidence=0.75,
        entry_threshold=0.75,
        exit_threshold=0.60,
        position_scale=0.3,
        max_positions=2,
        stop_loss_pct=0.40,
        take_profit_pct=0.80,
        trailing_stop_pct=0.30,
        max_hold_minutes=45,
        max_trades_per_hour=2,
        min_time_between_trades=15,
        require_multi_tf_agreement=True,
        allow_counter_trend=False,
        require_calibration=True,
        max_brier_score=0.25,  # Stricter calibration requirement
        regime_name="Extreme Vol",
        description="VIX > 35: Capital preservation mode"
    ),
}


# Trend regime modifiers (applied on top of volatility params)
TREND_MODIFIERS = {
    TrendRegime.STRONG_UP: {
        'bullish_confidence_bonus': 0.05,
        'bearish_confidence_penalty': 0.10,
        'trailing_stop_multiplier': 1.3,  # Wider trailing for trend
        'allow_counter_trend': False,
    },
    TrendRegime.MODERATE_UP: {
        'bullish_confidence_bonus': 0.02,
        'bearish_confidence_penalty': 0.05,
        'trailing_stop_multiplier': 1.15,
    },
    TrendRegime.SIDEWAYS: {
        'max_hold_multiplier': 0.7,  # Shorter holds in chop
        'take_profit_multiplier': 0.8,  # Smaller targets
        'require_mean_reversion': True,
    },
    TrendRegime.MODERATE_DOWN: {
        'bullish_confidence_penalty': 0.05,
        'bearish_confidence_bonus': 0.02,
        'trailing_stop_multiplier': 1.15,
    },
    TrendRegime.STRONG_DOWN: {
        'bullish_confidence_penalty': 0.10,
        'bearish_confidence_bonus': 0.05,
        'trailing_stop_multiplier': 1.3,
        'allow_counter_trend': False,
    },
}


class RegimeDetector:
    """
    Detects current market regime from various inputs.
    """
    
    def __init__(
        self,
        vix_thresholds: Tuple[float, ...] = (12, 15, 20, 25, 35),
        trend_lookback: int = 20,
        use_hmm: bool = True
    ):
        """
        Args:
            vix_thresholds: VIX levels for regime boundaries
            trend_lookback: Bars to look back for trend detection
            use_hmm: Whether to use HMM states if available
        """
        self.vix_thresholds = vix_thresholds
        self.trend_lookback = trend_lookback
        self.use_hmm = use_hmm
        
        self._last_regime: Optional[Dict] = None
        self._regime_history: List[Dict] = []
    
    def detect_volatility_regime(self, vix_level: float) -> VolatilityRegime:
        """Classify volatility regime from VIX."""
        if vix_level < self.vix_thresholds[0]:
            return VolatilityRegime.ULTRA_LOW
        elif vix_level < self.vix_thresholds[1]:
            return VolatilityRegime.LOW
        elif vix_level < self.vix_thresholds[2]:
            return VolatilityRegime.NORMAL
        elif vix_level < self.vix_thresholds[3]:
            return VolatilityRegime.ELEVATED
        elif vix_level < self.vix_thresholds[4]:
            return VolatilityRegime.HIGH
        else:
            return VolatilityRegime.EXTREME
    
    def detect_trend_regime(
        self,
        prices: np.ndarray = None,
        hmm_trend_state: int = None
    ) -> TrendRegime:
        """Classify trend regime from prices or HMM state."""
        # Use HMM if available and enabled
        if self.use_hmm and hmm_trend_state is not None:
            # HMM states: 0=bullish, 1=neutral, 2=bearish (typical mapping)
            if hmm_trend_state == 0:
                return TrendRegime.STRONG_UP
            elif hmm_trend_state == 2:
                return TrendRegime.STRONG_DOWN
            else:
                return TrendRegime.SIDEWAYS
        
        # Fall back to price-based detection
        if prices is None or len(prices) < self.trend_lookback:
            return TrendRegime.SIDEWAYS
        
        recent = prices[-self.trend_lookback:]
        returns = (recent[-1] / recent[0] - 1) * 100
        
        if returns > 2.0:
            return TrendRegime.STRONG_UP
        elif returns > 0.5:
            return TrendRegime.MODERATE_UP
        elif returns < -2.0:
            return TrendRegime.STRONG_DOWN
        elif returns < -0.5:
            return TrendRegime.MODERATE_DOWN
        else:
            return TrendRegime.SIDEWAYS
    
    def detect_all(
        self,
        vix_level: float,
        prices: np.ndarray = None,
        hmm_states: Dict[str, int] = None
    ) -> Dict[str, Any]:
        """
        Detect all regime types.
        
        Args:
            vix_level: Current VIX level
            prices: Recent price array
            hmm_states: Dict with 'trend', 'vol', 'liq' HMM states
            
        Returns:
            Dict with regime classifications
        """
        vol_regime = self.detect_volatility_regime(vix_level)
        
        trend_state = hmm_states.get('trend') if hmm_states else None
        trend_regime = self.detect_trend_regime(prices, trend_state)
        
        regime = {
            'volatility': vol_regime,
            'trend': trend_regime,
            'vix_level': vix_level,
            'timestamp': datetime.now(),
            'hmm_states': hmm_states
        }
        
        # Track history
        self._last_regime = regime
        self._regime_history.append(regime)
        if len(self._regime_history) > 100:
            self._regime_history = self._regime_history[-100:]
        
        return regime


class RegimeStrategyManager:
    """
    Manages regime-specific trading strategies.
    
    Provides adjusted trading parameters based on current market regime.
    """
    
    def __init__(
        self,
        detector: RegimeDetector = None,
        base_params: RegimeParams = None
    ):
        """
        Args:
            detector: RegimeDetector instance
            base_params: Default parameters to use
        """
        self.detector = detector or RegimeDetector()
        self.base_params = base_params or REGIME_PARAMS[VolatilityRegime.NORMAL]
        
        self._current_params: Optional[RegimeParams] = None
        self._current_regime: Optional[Dict] = None
        self._trade_times: List[datetime] = []
        
        logger.info("✅ RegimeStrategyManager initialized")
    
    def get_trading_params(
        self,
        vix_level: float,
        prices: np.ndarray = None,
        hmm_states: Dict[str, int] = None
    ) -> RegimeParams:
        """
        Get trading parameters for current market regime.
        
        Args:
            vix_level: Current VIX level
            prices: Recent price array
            hmm_states: HMM state dict
            
        Returns:
            RegimeParams adjusted for current regime
        """
        # Detect regime
        regime = self.detector.detect_all(vix_level, prices, hmm_states)
        self._current_regime = regime
        
        # Get base params for volatility regime
        vol_regime = regime['volatility']
        params = REGIME_PARAMS.get(vol_regime, self.base_params)
        
        # Apply trend modifiers
        trend_regime = regime['trend']
        if trend_regime in TREND_MODIFIERS:
            params = self._apply_trend_modifiers(params, trend_regime)
        
        self._current_params = params
        
        logger.debug(
            f"[REGIME] {vol_regime.value}/{trend_regime.value}: "
            f"conf>={params.min_confidence:.0%}, scale={params.position_scale:.1f}"
        )
        
        return params
    
    def _apply_trend_modifiers(
        self,
        params: RegimeParams,
        trend: TrendRegime
    ) -> RegimeParams:
        """Apply trend-based modifiers to params."""
        mods = TREND_MODIFIERS.get(trend, {})
        
        # Create modified copy
        modified = RegimeParams(
            min_confidence=params.min_confidence,
            entry_threshold=params.entry_threshold,
            exit_threshold=params.exit_threshold,
            position_scale=params.position_scale,
            max_positions=params.max_positions,
            stop_loss_pct=params.stop_loss_pct,
            take_profit_pct=params.take_profit_pct * mods.get('take_profit_multiplier', 1.0),
            trailing_stop_pct=params.trailing_stop_pct * mods.get('trailing_stop_multiplier', 1.0),
            max_hold_minutes=int(params.max_hold_minutes * mods.get('max_hold_multiplier', 1.0)),
            max_trades_per_hour=params.max_trades_per_hour,
            min_time_between_trades=params.min_time_between_trades,
            require_calibration=params.require_calibration,
            max_brier_score=params.max_brier_score,
            max_ece=params.max_ece,
            allow_counter_trend=mods.get('allow_counter_trend', params.allow_counter_trend),
            require_multi_tf_agreement=params.require_multi_tf_agreement,
            regime_name=f"{params.regime_name} + {trend.value}",
            description=params.description
        )
        
        return modified
    
    def should_trade(
        self,
        signal: Dict,
        params: RegimeParams = None,
        calibration_metrics: Dict = None
    ) -> Tuple[bool, List[str]]:
        """
        Check if should trade given signal and current regime.
        
        Args:
            signal: Signal dict with 'action', 'confidence', etc.
            params: Trading params (uses current if None)
            calibration_metrics: Calibration metrics dict
            
        Returns:
            (should_trade, rejection_reasons)
        """
        params = params or self._current_params or self.base_params
        reasons = []
        
        action = signal.get('action', 'HOLD')
        confidence = signal.get('confidence', 0)
        
        # 1. Action check
        if action == 'HOLD':
            return False, ['HOLD signal']
        
        # 2. Confidence check
        if confidence < params.min_confidence:
            reasons.append(f"Low confidence: {confidence:.1%} < {params.min_confidence:.1%}")
        
        # 3. Counter-trend check
        if not params.allow_counter_trend and self._current_regime:
            trend = self._current_regime.get('trend')
            is_counter = (
                (trend in [TrendRegime.STRONG_UP, TrendRegime.MODERATE_UP] and 'PUT' in action) or
                (trend in [TrendRegime.STRONG_DOWN, TrendRegime.MODERATE_DOWN] and 'CALL' in action)
            )
            if is_counter:
                reasons.append(f"Counter-trend trade blocked in {trend.value}")
        
        # 4. Calibration check
        if params.require_calibration and calibration_metrics:
            brier = calibration_metrics.get('brier_score', 1.0)
            ece = calibration_metrics.get('ece', 1.0)
            samples = calibration_metrics.get('sample_count', 0)
            
            if samples < 30:
                reasons.append(f"Insufficient calibration samples: {samples}/30")
            elif brier > params.max_brier_score:
                reasons.append(f"Poor Brier: {brier:.3f} > {params.max_brier_score:.3f}")
            elif ece > params.max_ece:
                reasons.append(f"Poor ECE: {ece:.3f} > {params.max_ece:.3f}")
        
        # 5. Trade frequency check
        self._cleanup_trade_times()
        recent_trades = len([t for t in self._trade_times 
                            if (datetime.now() - t).total_seconds() < 3600])
        
        if recent_trades >= params.max_trades_per_hour:
            reasons.append(f"Max trades/hour: {recent_trades}/{params.max_trades_per_hour}")
        
        # 6. Multi-TF agreement
        if params.require_multi_tf_agreement:
            multi_tf = signal.get('multi_timeframe_predictions', {})
            if multi_tf:
                agreement = self._check_multi_tf_agreement(action, multi_tf)
                if not agreement:
                    reasons.append("Multi-timeframe disagreement")
        
        should_trade = len(reasons) == 0
        
        if should_trade:
            self._trade_times.append(datetime.now())
        
        return should_trade, reasons
    
    def _check_multi_tf_agreement(self, action: str, multi_tf: Dict) -> bool:
        """Check if multiple timeframes agree on direction."""
        is_bullish = 'CALL' in action
        agreements = 0
        total = 0
        
        for tf, pred in multi_tf.items():
            if isinstance(pred, dict):
                direction = pred.get('direction', 0)
                total += 1
                if (is_bullish and direction > 0) or (not is_bullish and direction < 0):
                    agreements += 1
        
        return total == 0 or agreements >= total * 0.6
    
    def _cleanup_trade_times(self):
        """Remove old trade times."""
        cutoff = datetime.now() - timedelta(hours=2)
        self._trade_times = [t for t in self._trade_times if t > cutoff]
    
    def get_regime_stats(self) -> Dict:
        """Get statistics about regime history."""
        if not self.detector._regime_history:
            return {}
        
        history = self.detector._regime_history
        
        vol_counts = {}
        trend_counts = {}
        
        for r in history:
            vol = r['volatility'].value
            trend = r['trend'].value
            vol_counts[vol] = vol_counts.get(vol, 0) + 1
            trend_counts[trend] = trend_counts.get(trend, 0) + 1
        
        return {
            'current_regime': {
                'volatility': self._current_regime['volatility'].value if self._current_regime else None,
                'trend': self._current_regime['trend'].value if self._current_regime else None,
                'vix': self._current_regime['vix_level'] if self._current_regime else None
            },
            'current_params': {
                'min_confidence': self._current_params.min_confidence if self._current_params else None,
                'position_scale': self._current_params.position_scale if self._current_params else None,
                'regime_name': self._current_params.regime_name if self._current_params else None
            },
            'history_count': len(history),
            'volatility_distribution': vol_counts,
            'trend_distribution': trend_counts,
            'trades_last_hour': len([t for t in self._trade_times 
                                    if (datetime.now() - t).total_seconds() < 3600])
        }


# =============================================================================
# LOW-VOL SCENARIO BACKTESTER
# =============================================================================

class LowVolBacktester:
    """
    Backtests strategies specifically in low-volatility scenarios.
    
    Helps tune hyperparameters for quiet markets where the bot might
    struggle to generate profits.
    """
    
    def __init__(
        self,
        vix_threshold: float = 15.0,
        min_samples: int = 100
    ):
        """
        Args:
            vix_threshold: VIX level below which is "low vol"
            min_samples: Minimum samples for valid backtest
        """
        self.vix_threshold = vix_threshold
        self.min_samples = min_samples
        
        self._results: List[Dict] = []
    
    def filter_low_vol_periods(
        self,
        data: Any,
        vix_data: Any
    ) -> Tuple[Any, List[Tuple[datetime, datetime]]]:
        """
        Filter data to only low-volatility periods.
        
        Returns:
            (filtered_data, list of (start, end) periods)
        """
        # Implementation depends on data format
        # This is a placeholder showing the concept
        pass
    
    def run_backtest(
        self,
        strategy_func,
        data: Any,
        params: Dict
    ) -> Dict:
        """
        Run backtest with given strategy and parameters.
        
        Args:
            strategy_func: Strategy function(data, params) -> trades
            data: Historical data
            params: Strategy parameters to test
            
        Returns:
            Performance metrics
        """
        # Run strategy
        trades = strategy_func(data, params)
        
        if len(trades) < self.min_samples:
            return {'valid': False, 'reason': 'Insufficient trades'}
        
        # Calculate metrics
        wins = sum(1 for t in trades if t.get('pnl', 0) > 0)
        losses = len(trades) - wins
        total_pnl = sum(t.get('pnl', 0) for t in trades)
        
        result = {
            'valid': True,
            'params': params,
            'trades': len(trades),
            'win_rate': wins / len(trades),
            'total_pnl': total_pnl,
            'avg_pnl': total_pnl / len(trades),
            'profit_factor': abs(sum(t['pnl'] for t in trades if t['pnl'] > 0) / 
                               (sum(t['pnl'] for t in trades if t['pnl'] < 0) or 1)),
        }
        
        self._results.append(result)
        return result
    
    def grid_search(
        self,
        strategy_func,
        data: Any,
        param_grid: Dict[str, List]
    ) -> List[Dict]:
        """
        Grid search over parameter combinations.
        
        Args:
            strategy_func: Strategy function
            data: Historical data
            param_grid: Dict mapping param name to list of values
            
        Returns:
            List of results sorted by performance
        """
        import itertools
        
        results = []
        
        # Generate all combinations
        keys = list(param_grid.keys())
        values = list(param_grid.values())
        
        for combo in itertools.product(*values):
            params = dict(zip(keys, combo))
            result = self.run_backtest(strategy_func, data, params)
            if result['valid']:
                results.append(result)
        
        # Sort by total PnL
        results.sort(key=lambda x: x['total_pnl'], reverse=True)
        
        return results
    
    def get_optimal_params(self) -> Optional[Dict]:
        """Get best parameters from all backtests."""
        valid = [r for r in self._results if r['valid']]
        if not valid:
            return None
        
        best = max(valid, key=lambda x: x['total_pnl'])
        return best['params']


# =============================================================================
# FACTORY FUNCTIONS
# =============================================================================

def create_regime_manager(config: Dict = None) -> RegimeStrategyManager:
    """Create RegimeStrategyManager from config."""
    detector = RegimeDetector(
        vix_thresholds=config.get('vix_thresholds', (12, 15, 20, 25, 35)) if config else (12, 15, 20, 25, 35),
        trend_lookback=config.get('trend_lookback', 20) if config else 20,
        use_hmm=config.get('use_hmm', True) if config else True
    )
    
    return RegimeStrategyManager(detector=detector)


def get_regime_params(vix_level: float) -> RegimeParams:
    """Quick helper to get params for a VIX level."""
    detector = RegimeDetector()
    regime = detector.detect_volatility_regime(vix_level)
    return REGIME_PARAMS.get(regime, REGIME_PARAMS[VolatilityRegime.NORMAL])


