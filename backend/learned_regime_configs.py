#!/usr/bin/env python3
"""
Learned Regime Configurations
=============================

Dynamic configuration selection based on market regime (VIX + trend).
Configs are derived from our best-performing experiments.

Last updated: 2026-01-05
Based on 389 experiments (post bug-fix)

Usage:
    from backend.learned_regime_configs import RegimeConfigSelector

    selector = RegimeConfigSelector()

    # Get config for current conditions
    config = selector.get_config(vix=18.5, hmm_trend=0.7)

    # Apply to trading
    stop_loss = config['stop_loss_pct']
    take_profit = config['take_profit_pct']
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, Optional, Any
from enum import Enum

logger = logging.getLogger(__name__)


class VIXRegime(Enum):
    """VIX-based volatility regime."""
    ULTRA_LOW = "ULTRA_LOW"      # VIX < 12
    LOW = "LOW"                   # VIX 12-15
    NORMAL = "NORMAL"             # VIX 15-20
    ELEVATED = "ELEVATED"         # VIX 20-25
    HIGH = "HIGH"                 # VIX 25-35
    EXTREME = "EXTREME"           # VIX > 35


class TrendRegime(Enum):
    """HMM trend-based regime."""
    BULLISH = "BULLISH"           # HMM trend > 0.65
    NEUTRAL = "NEUTRAL"           # HMM trend 0.35-0.65
    BEARISH = "BEARISH"           # HMM trend < 0.35


@dataclass
class RegimeConfig:
    """Trading configuration for a specific regime."""
    regime_name: str
    source_experiment: str = ""
    expected_win_rate: float = 40.0
    expected_pnl_pct: float = 0.0

    # Entry parameters
    min_confidence: float = 0.55
    entry_threshold: float = 0.55

    # Exit parameters
    stop_loss_pct: float = 8.0
    take_profit_pct: float = 12.0
    trailing_stop_activation: float = 4.0
    trailing_stop_pct: float = 2.0
    max_hold_minutes: int = 45

    # Position sizing
    position_scale: float = 1.0
    max_positions: int = 3

    # Trade frequency
    max_trades_per_hour: int = 5
    min_time_between_trades: int = 5

    # Flags
    allow_counter_trend: bool = True
    require_hmm_alignment: bool = True


# =============================================================================
# LEARNED CONFIGURATIONS FROM EXPERIMENTS
# =============================================================================

# Best overall config (IDEA-266: 68.1% WR, +201.88% P&L)
BEST_OVERALL = RegimeConfig(
    regime_name="BEST_OVERALL",
    source_experiment="EXP-0167_IDEA-266",
    expected_win_rate=68.1,
    expected_pnl_pct=201.88,
    min_confidence=0.55,
    stop_loss_pct=8.0,
    take_profit_pct=12.0,
    max_hold_minutes=45,
    position_scale=1.0
)

# Configs by VIX regime (adjusted from best performers)
REGIME_CONFIGS: Dict[VIXRegime, RegimeConfig] = {
    VIXRegime.ULTRA_LOW: RegimeConfig(
        regime_name="ULTRA_LOW_VOL",
        source_experiment="dec_validation",
        expected_win_rate=60.0,
        expected_pnl_pct=400.0,
        min_confidence=0.45,           # Lower threshold - more trades
        entry_threshold=0.45,
        stop_loss_pct=6.0,             # Tighter stops (small moves)
        take_profit_pct=10.0,          # Smaller targets
        trailing_stop_activation=3.0,
        trailing_stop_pct=1.5,
        max_hold_minutes=60,
        position_scale=1.2,            # Larger positions (less risk)
        max_positions=5,
        max_trades_per_hour=8,
        min_time_between_trades=3,
        require_hmm_alignment=False    # Don't need alignment in calm
    ),

    VIXRegime.LOW: RegimeConfig(
        regime_name="LOW_VOL",
        source_experiment="dec_validation_v2",
        expected_win_rate=55.0,
        expected_pnl_pct=300.0,
        min_confidence=0.50,
        entry_threshold=0.50,
        stop_loss_pct=7.0,
        take_profit_pct=11.0,
        trailing_stop_activation=3.5,
        trailing_stop_pct=1.8,
        max_hold_minutes=50,
        position_scale=1.1,
        max_positions=4,
        max_trades_per_hour=6,
        min_time_between_trades=4
    ),

    VIXRegime.NORMAL: RegimeConfig(
        regime_name="NORMAL_VOL",
        source_experiment="EXP-0167_IDEA-266",  # Our best!
        expected_win_rate=68.1,
        expected_pnl_pct=201.88,
        min_confidence=0.55,
        entry_threshold=0.55,
        stop_loss_pct=8.0,
        take_profit_pct=12.0,
        trailing_stop_activation=4.0,
        trailing_stop_pct=2.0,
        max_hold_minutes=45,
        position_scale=1.0,
        max_positions=3,
        max_trades_per_hour=5,
        min_time_between_trades=5
    ),

    VIXRegime.ELEVATED: RegimeConfig(
        regime_name="ELEVATED_VOL",
        source_experiment="skew_tighter_tp",
        expected_win_rate=56.8,
        expected_pnl_pct=130.0,
        min_confidence=0.60,           # Higher threshold
        entry_threshold=0.60,
        stop_loss_pct=10.0,            # Wider stops
        take_profit_pct=15.0,          # Larger targets
        trailing_stop_activation=5.0,
        trailing_stop_pct=2.5,
        max_hold_minutes=40,
        position_scale=0.8,            # Smaller positions
        max_positions=2,
        max_trades_per_hour=4,
        min_time_between_trades=7,
        require_hmm_alignment=True
    ),

    VIXRegime.HIGH: RegimeConfig(
        regime_name="HIGH_VOL",
        source_experiment="conf_max_25",
        expected_win_rate=52.8,
        expected_pnl_pct=104.0,
        min_confidence=0.65,           # Very selective
        entry_threshold=0.65,
        stop_loss_pct=15.0,            # Wide stops
        take_profit_pct=25.0,          # Large targets
        trailing_stop_activation=8.0,
        trailing_stop_pct=4.0,
        max_hold_minutes=30,           # Shorter holds
        position_scale=0.6,            # Small positions
        max_positions=2,
        max_trades_per_hour=3,
        min_time_between_trades=10,
        require_hmm_alignment=True,
        allow_counter_trend=False      # Only trade with trend
    ),

    VIXRegime.EXTREME: RegimeConfig(
        regime_name="EXTREME_VOL",
        source_experiment="conservative",
        expected_win_rate=45.0,
        expected_pnl_pct=50.0,
        min_confidence=0.70,           # Very high threshold
        entry_threshold=0.70,
        stop_loss_pct=20.0,            # Very wide stops
        take_profit_pct=35.0,          # Large targets
        trailing_stop_activation=10.0,
        trailing_stop_pct=5.0,
        max_hold_minutes=20,           # Very short holds
        position_scale=0.4,            # Tiny positions
        max_positions=1,
        max_trades_per_hour=2,
        min_time_between_trades=15,
        require_hmm_alignment=True,
        allow_counter_trend=False
    )
}


class RegimeConfigSelector:
    """
    Selects the optimal trading configuration based on current market regime.

    Usage:
        selector = RegimeConfigSelector()
        config = selector.get_config(vix=18.5, hmm_trend=0.7)
    """

    def __init__(self):
        self.current_regime: Optional[VIXRegime] = None
        self.current_config: RegimeConfig = BEST_OVERALL
        self.regime_change_count = 0

    def classify_vix_regime(self, vix: float) -> VIXRegime:
        """Classify VIX level into regime bucket."""
        if vix < 12:
            return VIXRegime.ULTRA_LOW
        elif vix < 15:
            return VIXRegime.LOW
        elif vix < 20:
            return VIXRegime.NORMAL
        elif vix < 25:
            return VIXRegime.ELEVATED
        elif vix < 35:
            return VIXRegime.HIGH
        else:
            return VIXRegime.EXTREME

    def classify_trend_regime(self, hmm_trend: float) -> TrendRegime:
        """Classify HMM trend into regime bucket."""
        if hmm_trend > 0.65:
            return TrendRegime.BULLISH
        elif hmm_trend < 0.35:
            return TrendRegime.BEARISH
        else:
            return TrendRegime.NEUTRAL

    def get_config(self, vix: float, hmm_trend: float = 0.5) -> RegimeConfig:
        """
        Get the optimal configuration for current market conditions.

        Args:
            vix: Current VIX level
            hmm_trend: HMM trend state (0=bearish, 1=bullish)

        Returns:
            RegimeConfig with parameters for this regime
        """
        new_regime = self.classify_vix_regime(vix)

        # Check for regime change
        if new_regime != self.current_regime:
            old_regime = self.current_regime
            self.current_regime = new_regime
            self.current_config = REGIME_CONFIGS.get(new_regime, BEST_OVERALL)
            self.regime_change_count += 1

            logger.info(
                f"[REGIME] Changed: {old_regime} -> {new_regime} "
                f"(VIX={vix:.1f}, config={self.current_config.source_experiment})"
            )

        return self.current_config

    def get_config_dict(self, vix: float, hmm_trend: float = 0.5) -> Dict[str, Any]:
        """Get config as dictionary for easy integration."""
        config = self.get_config(vix, hmm_trend)
        return {
            'regime_name': config.regime_name,
            'source_experiment': config.source_experiment,
            'min_confidence': config.min_confidence,
            'stop_loss_pct': config.stop_loss_pct,
            'take_profit_pct': config.take_profit_pct,
            'trailing_stop_activation': config.trailing_stop_activation,
            'trailing_stop_pct': config.trailing_stop_pct,
            'max_hold_minutes': config.max_hold_minutes,
            'position_scale': config.position_scale,
            'max_positions': config.max_positions,
            'require_hmm_alignment': config.require_hmm_alignment,
            'allow_counter_trend': config.allow_counter_trend
        }

    def get_status(self) -> Dict[str, Any]:
        """Get current regime status."""
        return {
            'current_regime': self.current_regime.value if self.current_regime else None,
            'config_source': self.current_config.source_experiment,
            'expected_win_rate': self.current_config.expected_win_rate,
            'regime_changes': self.regime_change_count
        }


# Global instance for easy access
_selector: Optional[RegimeConfigSelector] = None


def get_regime_config(vix: float, hmm_trend: float = 0.5) -> RegimeConfig:
    """Get config for current regime (singleton pattern)."""
    global _selector
    if _selector is None:
        _selector = RegimeConfigSelector()
    return _selector.get_config(vix, hmm_trend)


def get_regime_config_dict(vix: float, hmm_trend: float = 0.5) -> Dict[str, Any]:
    """Get config as dict for current regime."""
    global _selector
    if _selector is None:
        _selector = RegimeConfigSelector()
    return _selector.get_config_dict(vix, hmm_trend)
