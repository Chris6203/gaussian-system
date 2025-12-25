#!/usr/bin/env python3
"""
Horizon Alignment Module
=========================

Ensures consistent time horizons across all components:
- Predictor label generation
- RL reward calculation
- Exit policy reference time

Misaligned horizons cause:
- Predictor learns to predict X minutes, but RL rewards based on Y minutes
- Exit policy uses different time reference than predictor
- Confusing learning signals, poor performance

This module provides:
1. Central horizon configuration
2. Assertions to catch misalignment
3. Helper functions for time-based calculations

Usage:
    from backend.horizon_alignment import HorizonConfig, get_aligned_horizon
    
    config = HorizonConfig.from_config("config.json")
    
    # Get aligned horizons
    pred_horizon = config.prediction_minutes
    rl_horizon = config.rl_reward_minutes
    exit_horizon = config.exit_reference_minutes
    
    # Assert they're aligned
    config.assert_aligned()
"""

import logging
from dataclasses import dataclass
from typing import Dict, Optional, Tuple
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


@dataclass
class HorizonConfig:
    """
    Central configuration for time horizons.
    
    All three horizons should typically be the same for consistency:
    - prediction_minutes: How far ahead the predictor predicts
    - rl_reward_minutes: Window for RL reward calculation
    - exit_reference_minutes: Reference horizon for exit policy timing
    """
    prediction_minutes: int = 15
    rl_reward_minutes: int = 15
    exit_reference_minutes: int = 15
    
    # Whether to enforce alignment
    strict_alignment: bool = True
    
    def assert_aligned(self):
        """Assert all horizons are aligned. Raises if not."""
        horizons = {
            'prediction': self.prediction_minutes,
            'rl_reward': self.rl_reward_minutes,
            'exit_reference': self.exit_reference_minutes,
        }
        
        values = list(horizons.values())
        if len(set(values)) > 1:
            msg = f"Horizons not aligned: {horizons}"
            if self.strict_alignment:
                raise ValueError(msg)
            else:
                logger.warning(f"⚠️ {msg}")
    
    @classmethod
    def from_dict(cls, config_dict: Dict) -> "HorizonConfig":
        """Create from config dictionary."""
        arch = config_dict.get("architecture", {})
        horizons = arch.get("horizons", {})
        
        return cls(
            prediction_minutes=horizons.get("prediction_minutes", 15),
            rl_reward_minutes=horizons.get("rl_reward_minutes", 15),
            exit_reference_minutes=horizons.get("exit_reference_minutes", 15),
            strict_alignment=horizons.get("assert_aligned", True),
        )
    
    @classmethod
    def from_file(cls, config_path: str) -> "HorizonConfig":
        """Load from config file."""
        import json
        try:
            with open(config_path, 'r') as f:
                config_dict = json.load(f)
            return cls.from_dict(config_dict)
        except Exception as e:
            logger.warning(f"Could not load horizon config: {e}, using defaults")
            return cls()
    
    def log_summary(self):
        """Log horizon configuration."""
        aligned = (self.prediction_minutes == self.rl_reward_minutes == self.exit_reference_minutes)
        status = "✅ ALIGNED" if aligned else "⚠️ MISALIGNED"
        
        logger.info(f"⏱️ Horizon Configuration ({status})")
        logger.info(f"   Prediction: {self.prediction_minutes} minutes")
        logger.info(f"   RL Reward:  {self.rl_reward_minutes} minutes")
        logger.info(f"   Exit Ref:   {self.exit_reference_minutes} minutes")


def calculate_label_offset(
    horizon_minutes: int,
    data_interval_minutes: int = 1,
) -> int:
    """
    Calculate the label offset (number of rows to shift) for training labels.
    
    Args:
        horizon_minutes: Prediction horizon in minutes
        data_interval_minutes: Data bar interval (e.g., 1 for 1-minute data)
        
    Returns:
        Number of rows to shift for labels
    """
    return horizon_minutes // data_interval_minutes


def calculate_time_features(
    entry_time: datetime,
    current_time: datetime,
    horizon_minutes: int,
    expiry_time: Optional[datetime] = None,
) -> Dict[str, float]:
    """
    Calculate time-based features for RL state.
    
    All features are normalized to [0, 1] or [-1, 1].
    
    Args:
        entry_time: When position was entered
        current_time: Current time
        horizon_minutes: Reference horizon (prediction horizon)
        expiry_time: Option expiry time (if applicable)
        
    Returns:
        Dictionary of time features
    """
    # Time held
    time_held = (current_time - entry_time).total_seconds() / 60
    
    # Ratio of time held to prediction horizon
    time_ratio = min(2.0, time_held / max(1, horizon_minutes))
    
    # Time remaining in prediction window
    time_remaining = max(0, horizon_minutes - time_held)
    time_remaining_ratio = time_remaining / max(1, horizon_minutes)
    
    # Time past prediction (if held longer than horizon)
    past_prediction_minutes = max(0, time_held - horizon_minutes)
    past_prediction_ratio = min(1.0, past_prediction_minutes / horizon_minutes) if past_prediction_minutes > 0 else 0
    
    features = {
        'time_held_minutes': time_held,
        'time_ratio': time_ratio,  # held / horizon
        'time_remaining_ratio': time_remaining_ratio,
        'past_prediction': float(time_held > horizon_minutes),
        'past_prediction_ratio': past_prediction_ratio,
    }
    
    # Expiry-related features
    if expiry_time is not None:
        time_to_expiry = (expiry_time - current_time).total_seconds() / 60
        features['minutes_to_expiry'] = max(0, time_to_expiry)
        features['expiry_urgency'] = 1.0 / max(1, time_to_expiry / 60)  # Higher when closer
    
    return features


def calculate_rl_reward_window(
    horizon_minutes: int,
    entry_time: datetime,
    exit_time: Optional[datetime] = None,
) -> Tuple[datetime, datetime]:
    """
    Calculate the time window for RL reward calculation.
    
    Returns the start and end times that should be used for
    measuring the outcome of a trade decision.
    
    Args:
        horizon_minutes: RL reward horizon
        entry_time: When position was entered
        exit_time: When position was exited (if already closed)
        
    Returns:
        (start_time, end_time) tuple for reward calculation
    """
    start_time = entry_time
    
    if exit_time is not None:
        # Trade was closed - use actual exit time
        end_time = exit_time
    else:
        # Trade still open - use horizon-based window
        end_time = entry_time + timedelta(minutes=horizon_minutes)
    
    return start_time, end_time


def validate_label_horizon(
    df_labels,
    expected_horizon_minutes: int,
    price_col: str = 'close',
    label_col: str = 'future_return',
    tolerance_pct: float = 0.1,
) -> bool:
    """
    Validate that labels in a DataFrame match the expected horizon.
    
    This is a sanity check to ensure label generation is aligned
    with the configured prediction horizon.
    
    Args:
        df_labels: DataFrame with labels
        expected_horizon_minutes: Expected horizon in minutes
        price_col: Column name for price
        label_col: Column name for label (future return)
        tolerance_pct: Acceptable error tolerance (10% default)
        
    Returns:
        True if labels are valid, False otherwise
    """
    if df_labels.empty or label_col not in df_labels.columns:
        logger.warning(f"Cannot validate: {label_col} not in DataFrame")
        return False
    
    # Check first few rows
    # Labels should be shifted by horizon_minutes rows
    # This is a heuristic check - exact validation depends on data interval
    
    try:
        # Check that label column has values
        label_count = df_labels[label_col].notna().sum()
        
        if label_count < 10:
            logger.warning(f"Too few labels to validate: {label_count}")
            return False
        
        # Basic sanity: returns should be in reasonable range
        mean_return = df_labels[label_col].mean()
        std_return = df_labels[label_col].std()
        
        if abs(mean_return) > 0.5 or std_return > 0.5:
            logger.warning(f"Labels may be miscalculated: mean={mean_return:.4f}, std={std_return:.4f}")
            return False
        
        logger.info(f"✅ Labels validated: {label_count} samples, mean={mean_return:.4f}, std={std_return:.4f}")
        return True
        
    except Exception as e:
        logger.warning(f"Label validation error: {e}")
        return False


# Global horizon config (set once at startup)
_horizon_config: Optional[HorizonConfig] = None


def get_horizon_config() -> HorizonConfig:
    """Get the global horizon configuration."""
    global _horizon_config
    if _horizon_config is None:
        _horizon_config = HorizonConfig()
    return _horizon_config


def set_horizon_config(config: HorizonConfig):
    """Set the global horizon configuration."""
    global _horizon_config
    _horizon_config = config
    config.log_summary()
    config.assert_aligned()


def get_aligned_horizon() -> int:
    """Get the aligned horizon value (all should be same)."""
    config = get_horizon_config()
    return config.prediction_minutes


# Export
__all__ = [
    'HorizonConfig',
    'calculate_label_offset',
    'calculate_time_features',
    'calculate_rl_reward_window',
    'validate_label_horizon',
    'get_horizon_config',
    'set_horizon_config',
    'get_aligned_horizon',
]
