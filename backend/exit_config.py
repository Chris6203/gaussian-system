"""
Centralized Exit Configuration - SINGLE SOURCE OF TRUTH

All exit-related parameters should be loaded from this module.
This prevents conflicting thresholds across different components.
"""
from dataclasses import dataclass
from typing import Optional
import json
import os
import logging

logger = logging.getLogger(__name__)


@dataclass
class ExitConfig:
    """Single source of truth for exit parameters"""

    # Stop loss and take profit (as decimal, e.g., 0.08 = 8%)
    stop_loss_pct: float = 0.08        # 8% stop loss
    take_profit_pct: float = 0.15      # 15% profit target

    # Trailing stop (as decimal)
    trailing_activation_pct: float = 0.08   # Activate trailing at 8% profit
    trailing_distance_pct: float = 0.04     # Trail by 4%

    # Time-based exits
    max_hold_minutes: int = 240        # 4 hours max hold
    min_expiry_minutes: int = 30       # Exit if < 30min to expiry

    @classmethod
    def from_config(cls, config_path: str = "config.json") -> "ExitConfig":
        """Load from config.json architecture.exit_policy section

        Environment variable overrides (for testing):
        - TT_STOP_LOSS_PCT: Stop loss as percentage (e.g., 15 for 15%)
        - TT_TAKE_PROFIT_PCT: Take profit as percentage (e.g., 25 for 25%)
        - TT_MAX_HOLD_MINUTES: Max hold time in minutes
        - TT_TRAILING_ACT_PCT: Trailing stop activation %
        - TT_TRAILING_DIST_PCT: Trailing stop distance %
        """
        try:
            with open(config_path) as f:
                cfg = json.load(f)

            ep = cfg.get("architecture", {}).get("exit_policy", {})

            # Convert from percentage format to decimal
            # Config stores as -8.0 (meaning 8%), we need 0.08
            stop_loss = abs(ep.get("hard_stop_loss_pct", -8.0)) / 100.0
            take_profit = ep.get("hard_take_profit_pct", 15.0) / 100.0
            trailing_act = ep.get("trailing_stop_activation_pct", 8.0) / 100.0
            trailing_dist = ep.get("trailing_stop_distance_pct", 4.0) / 100.0
            max_hold = ep.get("hard_max_hold_minutes", 240)
            min_expiry = ep.get("hard_min_expiry_minutes", 30)

            # Environment variable overrides for testing
            if os.environ.get("TT_STOP_LOSS_PCT"):
                stop_loss = float(os.environ["TT_STOP_LOSS_PCT"]) / 100.0
                logger.info(f"[ENV] Stop loss override: {stop_loss*100:.1f}%")
            if os.environ.get("TT_TAKE_PROFIT_PCT"):
                take_profit = float(os.environ["TT_TAKE_PROFIT_PCT"]) / 100.0
                logger.info(f"[ENV] Take profit override: {take_profit*100:.1f}%")
            if os.environ.get("TT_MAX_HOLD_MINUTES"):
                max_hold = int(os.environ["TT_MAX_HOLD_MINUTES"])
                logger.info(f"[ENV] Max hold override: {max_hold} min")
            if os.environ.get("TT_TRAILING_ACT_PCT"):
                trailing_act = float(os.environ["TT_TRAILING_ACT_PCT"]) / 100.0
                logger.info(f"[ENV] Trailing activation override: {trailing_act*100:.1f}%")
            if os.environ.get("TT_TRAILING_DIST_PCT"):
                trailing_dist = float(os.environ["TT_TRAILING_DIST_PCT"]) / 100.0
                logger.info(f"[ENV] Trailing distance override: {trailing_dist*100:.1f}%")

            result = cls(
                stop_loss_pct=stop_loss,
                take_profit_pct=take_profit,
                trailing_activation_pct=trailing_act,
                trailing_distance_pct=trailing_dist,
                max_hold_minutes=max_hold,
                min_expiry_minutes=min_expiry,
            )

            logger.info(f"📋 ExitConfig loaded: stop={result.stop_loss_pct*100:.1f}%, "
                       f"take={result.take_profit_pct*100:.1f}%, "
                       f"trail_act={result.trailing_activation_pct*100:.1f}%, "
                       f"trail_dist={result.trailing_distance_pct*100:.1f}%, "
                       f"max_hold={result.max_hold_minutes}min")
            return result

        except FileNotFoundError:
            logger.warning(f"Config file {config_path} not found, using defaults")
            return cls()
        except Exception as e:
            logger.error(f"Error loading config: {e}, using defaults")
            return cls()


# Singleton instance for global access
_exit_config: Optional[ExitConfig] = None


def get_exit_config(force_reload: bool = False) -> ExitConfig:
    """Get the global exit configuration singleton"""
    global _exit_config
    if _exit_config is None or force_reload:
        _exit_config = ExitConfig.from_config()
    return _exit_config


def reset_exit_config():
    """Reset the singleton (useful for testing)"""
    global _exit_config
    _exit_config = None
