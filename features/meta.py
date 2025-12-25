#!/usr/bin/env python3
"""
Meta / Regime / Time Features

Computes meta-features for trading:
- Time of day features
- Options expiry features
- Volatility regime classification
"""

import numpy as np
import pandas as pd
from datetime import datetime, date, timedelta
from typing import Dict, List, Optional, Tuple
import calendar
import logging

logger = logging.getLogger(__name__)


# =============================================================================
# TIME OF DAY FEATURES
# =============================================================================

def compute_time_features(
    timestamp: Optional[datetime] = None,
    market_open_hour: int = 9,
    market_open_minute: int = 30,
    market_close_hour: int = 16,
    market_close_minute: int = 0
) -> Dict[str, float]:
    """
    Compute time-of-day features.
    
    Args:
        timestamp: Current timestamp (default: now)
        market_open_hour/minute: Market open time (9:30 AM ET default)
        market_close_hour/minute: Market close time (4:00 PM ET default)
        
    Returns:
        Dict of time features:
        
        Continuous:
        - minutes_since_open: Minutes since market open
        - minutes_to_close: Minutes until market close
        - session_progress: 0 at open, 1 at close
        
        Session Buckets (one-hot):
        - is_first_30min: First 30 minutes of session
        - is_last_hour: Last hour of session
        - is_midday: Middle of session (11AM-2PM)
        - is_power_hour: Last hour (3-4 PM)
        
        Cyclic Encoding:
        - time_sin: Sine of time (for cyclic representation)
        - time_cos: Cosine of time
    """
    timestamp = timestamp or datetime.now()
    
    features = {
        # Continuous
        'minutes_since_open': 0.0,
        'minutes_to_close': 0.0,
        'session_progress': 0.5,
        
        # Session buckets
        'is_first_30min': 0.0,
        'is_last_hour': 0.0,
        'is_midday': 0.0,
        'is_power_hour': 0.0,
        'is_pre_market': 0.0,
        'is_post_market': 0.0,
        
        # Cyclic
        'time_sin': 0.0,
        'time_cos': 1.0,
    }
    
    # Market open/close in minutes from midnight
    market_open = market_open_hour * 60 + market_open_minute  # 570 for 9:30
    market_close = market_close_hour * 60 + market_close_minute  # 960 for 16:00
    session_length = market_close - market_open  # 390 minutes
    
    # Current time in minutes from midnight
    current_minutes = timestamp.hour * 60 + timestamp.minute
    
    # Minutes since open
    minutes_since_open = current_minutes - market_open
    features['minutes_since_open'] = float(max(0, minutes_since_open))
    
    # Minutes to close
    minutes_to_close = market_close - current_minutes
    features['minutes_to_close'] = float(max(0, minutes_to_close))
    
    # Session progress
    if minutes_since_open >= 0 and minutes_since_open <= session_length:
        features['session_progress'] = minutes_since_open / session_length
    elif minutes_since_open < 0:
        features['session_progress'] = 0.0
    else:
        features['session_progress'] = 1.0
    
    # Session buckets
    if minutes_since_open < 0:
        features['is_pre_market'] = 1.0
    elif minutes_since_open > session_length:
        features['is_post_market'] = 1.0
    elif minutes_since_open <= 30:
        features['is_first_30min'] = 1.0
    elif minutes_to_close <= 60:
        features['is_last_hour'] = 1.0
        features['is_power_hour'] = 1.0
    elif 90 <= minutes_since_open <= 270:  # 11:00 - 2:00
        features['is_midday'] = 1.0
    
    # Cyclic encoding (full day cycle)
    day_progress = current_minutes / (24 * 60)
    features['time_sin'] = float(np.sin(2 * np.pi * day_progress))
    features['time_cos'] = float(np.cos(2 * np.pi * day_progress))
    
    return features


# =============================================================================
# OPTIONS EXPIRY FEATURES
# =============================================================================

def get_third_friday(year: int, month: int) -> date:
    """Get the third Friday of a month (monthly options expiration)."""
    c = calendar.Calendar(firstweekday=calendar.MONDAY)
    monthcal = c.monthdatescalendar(year, month)
    
    fridays = [
        day for week in monthcal for day in week
        if day.weekday() == calendar.FRIDAY and day.month == month
    ]
    
    return fridays[2] if len(fridays) >= 3 else fridays[-1]


def get_monthly_opex_dates(year: int) -> List[date]:
    """Get all monthly options expiration dates for a year."""
    return [get_third_friday(year, month) for month in range(1, 13)]


def compute_expiry_features(
    current_date: Optional[date] = None,
    expiry_date: Optional[date] = None
) -> Dict[str, float]:
    """
    Compute options expiry-related features.
    
    Args:
        current_date: Current date (default: today)
        expiry_date: Option expiry date (for the traded option)
        
    Returns:
        Dict of expiry features:
        
        Days to Expiry:
        - days_to_expiry: DTE for the option
        - is_0dte: Is same-day expiry
        - is_1dte: Is next-day expiry
        - is_weekly: Is within 7 days
        
        Monthly OPEX:
        - is_opex_day: Is monthly options expiration day
        - is_opex_week: Is monthly OPEX week
        - days_to_opex: Days until next monthly OPEX
        
        Day of Week:
        - is_monday: Monday effect
        - is_friday: Friday expiry effect
        - day_of_week: 0=Mon, 4=Fri
    """
    current_date = current_date or date.today()
    
    features = {
        # DTE
        'days_to_expiry': 30.0,
        'is_0dte': 0.0,
        'is_1dte': 0.0,
        'is_weekly': 0.0,
        
        # Monthly OPEX
        'is_opex_day': 0.0,
        'is_opex_week': 0.0,
        'days_to_opex': 0.0,
        
        # Day of week
        'is_monday': 0.0,
        'is_friday': 0.0,
        'day_of_week': float(current_date.weekday()) / 4.0,
    }
    
    # Day of week features
    dow = current_date.weekday()
    features['is_monday'] = 1.0 if dow == 0 else 0.0
    features['is_friday'] = 1.0 if dow == 4 else 0.0
    
    # Days to expiry
    if expiry_date:
        dte = (expiry_date - current_date).days
        features['days_to_expiry'] = float(max(0, dte))
        features['is_0dte'] = 1.0 if dte == 0 else 0.0
        features['is_1dte'] = 1.0 if dte == 1 else 0.0
        features['is_weekly'] = 1.0 if dte <= 7 else 0.0
    
    # Monthly OPEX
    try:
        # Get OPEX dates for current and next year
        opex_dates = (
            get_monthly_opex_dates(current_date.year) +
            get_monthly_opex_dates(current_date.year + 1)
        )
        
        # Find next OPEX
        future_opex = [d for d in opex_dates if d >= current_date]
        
        if future_opex:
            next_opex = future_opex[0]
            days_to_opex = (next_opex - current_date).days
            
            features['days_to_opex'] = float(days_to_opex)
            features['is_opex_day'] = 1.0 if days_to_opex == 0 else 0.0
            features['is_opex_week'] = 1.0 if days_to_opex <= 5 else 0.0
            
    except Exception as e:
        logger.debug(f"Error computing OPEX features: {e}")
    
    return features


# =============================================================================
# VOLATILITY REGIME FEATURES
# =============================================================================

def compute_vol_regime_features(
    realized_vol: float,
    vol_history: Optional[np.ndarray] = None,
    vix_level: Optional[float] = None
) -> Dict[str, float]:
    """
    Compute volatility regime classification features.
    
    Args:
        realized_vol: Current realized volatility
        vol_history: Historical volatility values for percentile calculation
        vix_level: Current VIX level (optional)
        
    Returns:
        Dict of vol regime features:
        
        One-Hot Regime:
        - vol_regime_low: Low volatility regime
        - vol_regime_normal: Normal volatility regime
        - vol_regime_high: High volatility regime
        
        Continuous:
        - vol_percentile: Volatility percentile in history
        - vol_zscore: Volatility z-score
        
        VIX-Based:
        - vix_regime_low: VIX < 15
        - vix_regime_elevated: VIX 15-25
        - vix_regime_high: VIX > 25
    """
    features = {
        # One-hot regime
        'vol_regime_low': 0.0,
        'vol_regime_normal': 1.0,
        'vol_regime_high': 0.0,
        
        # Continuous
        'vol_percentile': 0.5,
        'vol_zscore': 0.0,
        
        # VIX-based
        'vix_regime_low': 0.0,
        'vix_regime_elevated': 0.0,
        'vix_regime_high': 0.0,
    }
    
    # Percentile-based regime (if history available)
    if vol_history is not None and len(vol_history) >= 20:
        percentile = np.sum(vol_history < realized_vol) / len(vol_history)
        features['vol_percentile'] = float(percentile)
        
        # Z-score
        mean_vol = np.mean(vol_history)
        std_vol = np.std(vol_history)
        if std_vol > 0:
            features['vol_zscore'] = float((realized_vol - mean_vol) / std_vol)
        
        # One-hot regime based on percentiles
        if percentile < 0.25:
            features['vol_regime_low'] = 1.0
            features['vol_regime_normal'] = 0.0
        elif percentile > 0.75:
            features['vol_regime_high'] = 1.0
            features['vol_regime_normal'] = 0.0
    
    # VIX-based regime
    if vix_level is not None:
        if vix_level < 15:
            features['vix_regime_low'] = 1.0
        elif vix_level < 25:
            features['vix_regime_elevated'] = 1.0
        else:
            features['vix_regime_high'] = 1.0
    
    return features


# =============================================================================
# COMBINED META FEATURES
# =============================================================================

def compute_meta_features(
    timestamp: Optional[datetime] = None,
    expiry_date: Optional[date] = None,
    realized_vol: float = 0.0,
    vol_history: Optional[np.ndarray] = None,
    vix_level: Optional[float] = None
) -> Dict[str, float]:
    """
    Compute all meta features combined.
    
    Args:
        timestamp: Current timestamp
        expiry_date: Option expiry date
        realized_vol: Current realized volatility
        vol_history: Historical volatility values
        vix_level: Current VIX level
        
    Returns:
        Dict with all meta features combined
    """
    features = {}
    
    # Time features
    time_features = compute_time_features(timestamp)
    features.update(time_features)
    
    # Expiry features
    current_date = timestamp.date() if timestamp else date.today()
    expiry_features = compute_expiry_features(current_date, expiry_date)
    features.update(expiry_features)
    
    # Vol regime features
    vol_features = compute_vol_regime_features(realized_vol, vol_history, vix_level)
    features.update(vol_features)
    
    return features


if __name__ == "__main__":
    """Test meta features."""
    import logging
    
    logging.basicConfig(level=logging.INFO)
    
    print("=" * 70)
    print("META FEATURE TEST")
    print("=" * 70)
    
    # Test time features
    print("\n--- Time Features (10:00 AM) ---")
    test_time = datetime(2024, 1, 15, 10, 0)  # Monday 10:00 AM
    time_features = compute_time_features(test_time)
    for k, v in sorted(time_features.items()):
        print(f"  {k}: {v:.3f}")
    
    # Test time features at different times
    print("\n--- Time Features (3:30 PM) ---")
    test_time = datetime(2024, 1, 15, 15, 30)  # Monday 3:30 PM
    time_features = compute_time_features(test_time)
    for k, v in sorted(time_features.items()):
        print(f"  {k}: {v:.3f}")
    
    # Test expiry features
    print("\n--- Expiry Features ---")
    current_date = date(2024, 1, 15)
    expiry_date = date(2024, 1, 19)  # Third Friday (OPEX)
    expiry_features = compute_expiry_features(current_date, expiry_date)
    for k, v in sorted(expiry_features.items()):
        print(f"  {k}: {v:.3f}")
    
    # Test vol regime features
    print("\n--- Vol Regime Features ---")
    np.random.seed(42)
    vol_history = np.abs(np.random.randn(100) * 0.02)  # Random vol history
    realized_vol = 0.035  # Current vol (high)
    
    vol_features = compute_vol_regime_features(
        realized_vol=realized_vol,
        vol_history=vol_history,
        vix_level=22.5
    )
    for k, v in sorted(vol_features.items()):
        print(f"  {k}: {v:.3f}")
    
    # Test combined
    print("\n--- Combined Meta Features ---")
    all_features = compute_meta_features(
        timestamp=datetime(2024, 1, 15, 10, 30),
        expiry_date=date(2024, 1, 17),
        realized_vol=0.025,
        vol_history=vol_history,
        vix_level=18.0
    )
    print(f"\nTotal features: {len(all_features)}")
    
    print("\n" + "=" * 70)
    print("TEST COMPLETE")
    print("=" * 70)




