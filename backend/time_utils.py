#!/usr/bin/env python3
"""
Time Utilities
==============

Centralized timestamp handling for consistent market time across the codebase.
All timestamps should be in US/Eastern (market time) for consistency.

Usage:
    from backend.time_utils import get_market_time, to_market_time, format_timestamp

    # Get current market time
    now = get_market_time()
    
    # Convert any datetime to market time
    market_dt = to_market_time(some_datetime)
    
    # Format for database/API (ISO format with T separator)
    ts_str = format_timestamp(now)
"""

from datetime import datetime, timedelta
from typing import Optional, Union

# Market timezone
try:
    import pytz
    MARKET_TZ = pytz.timezone('US/Eastern')
    UTC_TZ = pytz.UTC
    HAS_PYTZ = True
except ImportError:
    MARKET_TZ = None
    UTC_TZ = None
    HAS_PYTZ = False


def get_market_time() -> datetime:
    """
    Get current time in market timezone (US/Eastern) as naive datetime.
    
    Returns:
        datetime: Current time in Eastern timezone (naive, no tzinfo)
    """
    if HAS_PYTZ:
        market_dt = UTC_TZ.localize(datetime.utcnow()).astimezone(MARKET_TZ)
        return market_dt.replace(tzinfo=None)
    else:
        # Fallback: assume UTC-5 for Eastern (doesn't account for DST)
        return datetime.utcnow() - timedelta(hours=5)


def to_market_time(dt: Union[datetime, str, None]) -> Optional[datetime]:
    """
    Convert any datetime to market time (US/Eastern) as naive datetime.
    
    Args:
        dt: datetime object (naive or aware) or ISO format string
        
    Returns:
        datetime: Converted time in Eastern timezone (naive, no tzinfo)
                  None if input is None or invalid
    """
    if dt is None:
        return None
    
    # Parse string if needed
    if isinstance(dt, str):
        try:
            # Normalize format: handle space or T separator
            dt = datetime.fromisoformat(dt.replace(' ', 'T').replace('Z', '+00:00'))
        except Exception:
            return None
    
    if not isinstance(dt, datetime):
        return None
    
    if HAS_PYTZ:
        if dt.tzinfo is None:
            # Naive datetime - assume it's already in market time
            return dt
        else:
            # Aware datetime - convert to market time
            return dt.astimezone(MARKET_TZ).replace(tzinfo=None)
    else:
        # Fallback: return as-is (assume already in market time)
        if dt.tzinfo is not None:
            dt = dt.replace(tzinfo=None)
        return dt


def format_timestamp(dt: Union[datetime, None], include_tz: bool = False) -> str:
    """
    Format datetime for database/API storage.
    
    Uses ISO format with 'T' separator for consistent JavaScript parsing.
    
    Args:
        dt: datetime to format
        include_tz: If True, append timezone indicator
        
    Returns:
        str: Formatted timestamp string (e.g., "2025-06-11T17:22:00")
             Empty string if dt is None
    """
    if dt is None:
        return ""
    
    # Use ISO format with T separator
    formatted = dt.strftime('%Y-%m-%dT%H:%M:%S')
    
    if include_tz:
        formatted += " ET"
    
    return formatted


def parse_timestamp(ts_str: str) -> Optional[datetime]:
    """
    Parse a timestamp string into a naive datetime (market time).
    
    Handles various formats:
    - "2025-06-11T17:22:00" (ISO with T)
    - "2025-06-11 17:22:00" (ISO with space)
    - "2025-06-11T17:22:00Z" (UTC indicator)
    - "2025-06-11T17:22:00-05:00" (with timezone offset)
    
    Args:
        ts_str: Timestamp string to parse
        
    Returns:
        datetime: Parsed datetime in market time (naive)
                  None if parsing fails
    """
    if not ts_str:
        return None
    
    try:
        # Normalize format
        normalized = ts_str.replace(' ', 'T')
        
        # Handle Z suffix (UTC)
        if normalized.endswith('Z'):
            normalized = normalized[:-1] + '+00:00'
        
        dt = datetime.fromisoformat(normalized)
        return to_market_time(dt)
    except Exception:
        return None


def is_market_hours(dt: Optional[datetime] = None) -> bool:
    """
    Check if the given time (or current time) is during market hours.
    
    Market hours: 9:30 AM - 4:00 PM Eastern, Monday-Friday
    
    Args:
        dt: Time to check (defaults to current market time)
        
    Returns:
        bool: True if during market hours
    """
    if dt is None:
        dt = get_market_time()
    
    # Weekend check
    if dt.weekday() >= 5:
        return False
    
    # Time check
    market_open = dt.replace(hour=9, minute=30, second=0, microsecond=0)
    market_close = dt.replace(hour=16, minute=0, second=0, microsecond=0)
    
    return market_open <= dt <= market_close


def market_time_str() -> str:
    """Get current market time as formatted string for logging."""
    return format_timestamp(get_market_time())


# Convenience aliases
now = get_market_time


if __name__ == '__main__':
    # Test the module
    print("=" * 50)
    print("Time Utilities Test")
    print("=" * 50)
    print(f"pytz available: {HAS_PYTZ}")
    print(f"Market time: {format_timestamp(get_market_time())}")
    print(f"Is market hours: {is_market_hours()}")
    print("=" * 50)


