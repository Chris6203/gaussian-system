"""
Market Utilities
================

Utilities for market hours, device initialization, and time handling.

Usage:
    from bot_modules.market_utils import (
        MarketHours,
        DeviceManager,
        get_market_time
    )
"""

import logging
from datetime import datetime, time
from typing import Dict, Optional, Tuple

import pytz
import torch

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------------------
# Market Hours Configuration
# ------------------------------------------------------------------------------
class MarketHours:
    """
    Manages market hours for different symbols and exchanges.
    
    Handles:
    - Regular trading hours (9:30 AM - 4:00 PM ET)
    - Extended hours for VIX (8:30 AM - 4:15 PM ET)
    - Weekend/holiday detection
    - Pre/post market periods
    """
    
    # Default market hours (Eastern Time)
    DEFAULT_OPEN = time(9, 30)
    DEFAULT_CLOSE = time(16, 0)
    
    # Symbol-specific market hours
    SYMBOL_HOURS = {
        "BITX": {
            "open": time(9, 30),
            "close": time(16, 0)
        },
        "SPY": {
            "open": time(9, 30),
            "close": time(16, 0)
        },
        "QQQ": {
            "open": time(9, 30),
            "close": time(16, 0)
        },
        "VIX": {
            "open": time(8, 30),   # VIX futures/options start earlier
            "close": time(16, 15)  # VIX closes slightly later
        },
        "^VIX": {
            "open": time(8, 30),
            "close": time(16, 15)
        }
    }
    
    def __init__(self, timezone: str = "US/Eastern"):
        """
        Initialize market hours handler.
        
        Args:
            timezone: Timezone for market hours (default: US/Eastern)
        """
        self.timezone = pytz.timezone(timezone)
        self.symbol_hours = self.SYMBOL_HOURS.copy()
    
    def is_market_open(self, symbol: str = None, dt: datetime = None) -> bool:
        """
        Check if the market is open.
        
        Args:
            symbol: Symbol to check (uses default hours if not specified)
            dt: Datetime to check (uses current time if not specified)
            
        Returns:
            True if market is open
        """
        if dt is None:
            dt = self.get_current_market_time()
        elif dt.tzinfo is not None:
            dt = dt.astimezone(self.timezone).replace(tzinfo=None)
        
        # Weekend check
        if dt.weekday() >= 5:
            return False
        
        # Get symbol-specific hours
        if symbol and symbol in self.symbol_hours:
            open_time = self.symbol_hours[symbol]["open"]
            close_time = self.symbol_hours[symbol]["close"]
        else:
            open_time = self.DEFAULT_OPEN
            close_time = self.DEFAULT_CLOSE
        
        return open_time <= dt.time() <= close_time
    
    def get_current_market_time(self) -> datetime:
        """
        Get current time in market timezone.
        
        Returns:
            Current datetime in market timezone (naive)
        """
        market_dt = pytz.UTC.localize(datetime.utcnow()).astimezone(self.timezone)
        return market_dt.replace(tzinfo=None)
    
    def time_to_open(self, symbol: str = None, dt: datetime = None) -> Optional[int]:
        """
        Get seconds until market opens.
        
        Args:
            symbol: Symbol to check
            dt: Datetime to check (uses current time if not specified)
            
        Returns:
            Seconds until open, or None if already open or weekend
        """
        if dt is None:
            dt = self.get_current_market_time()
        
        if self.is_market_open(symbol, dt):
            return 0
        
        if dt.weekday() >= 5:  # Weekend
            return None
        
        # Get open time
        if symbol and symbol in self.symbol_hours:
            open_time = self.symbol_hours[symbol]["open"]
        else:
            open_time = self.DEFAULT_OPEN
        
        open_dt = dt.replace(hour=open_time.hour, minute=open_time.minute, second=0)
        
        if open_dt > dt:
            return int((open_dt - dt).total_seconds())
        return None
    
    def time_to_close(self, symbol: str = None, dt: datetime = None) -> Optional[int]:
        """
        Get seconds until market closes.
        
        Args:
            symbol: Symbol to check
            dt: Datetime to check (uses current time if not specified)
            
        Returns:
            Seconds until close, or None if market is closed
        """
        if dt is None:
            dt = self.get_current_market_time()
        
        if not self.is_market_open(symbol, dt):
            return None
        
        # Get close time
        if symbol and symbol in self.symbol_hours:
            close_time = self.symbol_hours[symbol]["close"]
        else:
            close_time = self.DEFAULT_CLOSE
        
        close_dt = dt.replace(hour=close_time.hour, minute=close_time.minute, second=0)
        
        if close_dt > dt:
            return int((close_dt - dt).total_seconds())
        return 0
    
    def get_session_progress(self, symbol: str = None, dt: datetime = None) -> float:
        """
        Get progress through the trading session (0.0 to 1.0).
        
        Args:
            symbol: Symbol to check
            dt: Datetime to check
            
        Returns:
            Progress as fraction (0 = just opened, 1 = closing)
        """
        if dt is None:
            dt = self.get_current_market_time()
        
        if not self.is_market_open(symbol, dt):
            return 0.0
        
        # Get hours
        if symbol and symbol in self.symbol_hours:
            open_time = self.symbol_hours[symbol]["open"]
            close_time = self.symbol_hours[symbol]["close"]
        else:
            open_time = self.DEFAULT_OPEN
            close_time = self.DEFAULT_CLOSE
        
        # Calculate progress
        open_minutes = open_time.hour * 60 + open_time.minute
        close_minutes = close_time.hour * 60 + close_time.minute
        current_minutes = dt.hour * 60 + dt.minute
        
        total_minutes = close_minutes - open_minutes
        elapsed = current_minutes - open_minutes
        
        return min(1.0, max(0.0, elapsed / total_minutes))
    
    def add_symbol_hours(self, symbol: str, open_time: time, close_time: time):
        """
        Add or update symbol-specific market hours.
        
        Args:
            symbol: Symbol name
            open_time: Market open time
            close_time: Market close time
        """
        self.symbol_hours[symbol] = {"open": open_time, "close": close_time}


# ------------------------------------------------------------------------------
# Device Management
# ------------------------------------------------------------------------------
class DeviceManager:
    """
    Manages PyTorch device selection with robust fallback to CPU.
    
    Handles:
    - GPU detection and initialization
    - Graceful fallback on CUDA errors
    - Memory management
    - Safe tensor/model movement
    """
    
    def __init__(self, prefer_gpu: bool = True):
        """
        Initialize device manager.
        
        Args:
            prefer_gpu: Whether to prefer GPU if available
        """
        self.prefer_gpu = prefer_gpu
        self._device: Optional[torch.device] = None
        self._gpu_name: Optional[str] = None
    
    @property
    def device(self) -> torch.device:
        """Get the current device (lazily initialized)."""
        if self._device is None:
            self._device = self._initialize_device()
        return self._device
    
    def _initialize_device(self) -> torch.device:
        """
        Initialize PyTorch device with robust fallback to CPU.
        
        Returns:
            torch.device: Either 'cuda' or 'cpu'
        """
        if not self.prefer_gpu:
            logger.info("ℹ️  GPU preference disabled, using CPU")
            return torch.device("cpu")
        
        try:
            if torch.cuda.is_available():
                # Verify we can actually use the GPU
                try:
                    test_tensor = torch.zeros(1, device='cuda')
                    self._gpu_name = torch.cuda.get_device_name(0)
                    logger.info(f"✅ GPU detected: {self._gpu_name}")
                    logger.info(f"✅ CUDA version: {torch.version.cuda}")
                    logger.info(f"✅ PyTorch will use GPU acceleration")
                    del test_tensor
                    torch.cuda.empty_cache()
                    return torch.device("cuda")
                except RuntimeError as e:
                    logger.warning(f"⚠️  GPU detected but initialization failed: {e}")
                    logger.warning(f"⚠️  Falling back to CPU")
                    return torch.device("cpu")
            else:
                logger.info(f"ℹ️  No GPU detected, using CPU")
                logger.info(f"ℹ️  To enable GPU: Install CUDA-enabled PyTorch")
                return torch.device("cpu")
        except Exception as e:
            logger.warning(f"⚠️  Error during device initialization: {e}")
            logger.warning(f"⚠️  Falling back to CPU")
            return torch.device("cpu")
    
    def to_device(self, tensor_or_model, fallback_to_cpu: bool = True):
        """
        Safely move tensor or model to device with automatic fallback.
        
        Args:
            tensor_or_model: PyTorch tensor or model to move
            fallback_to_cpu: If True, falls back to CPU on OOM
            
        Returns:
            Tensor or model on the appropriate device
        """
        try:
            return tensor_or_model.to(self.device)
        except RuntimeError as e:
            if "out of memory" in str(e).lower() and fallback_to_cpu:
                logger.warning(f"⚠️  GPU out of memory, falling back to CPU")
                if self.device.type == 'cuda':
                    torch.cuda.empty_cache()
                self._device = torch.device("cpu")
                logger.warning(f"⚠️  Permanently switched to CPU mode")
                return tensor_or_model.to(self.device)
            else:
                raise
    
    def empty_cache(self):
        """Empty GPU cache if using CUDA."""
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()
    
    def get_memory_info(self) -> Dict[str, float]:
        """
        Get GPU memory information.
        
        Returns:
            Dict with memory stats in GB (empty if on CPU)
        """
        if self.device.type != 'cuda':
            return {}
        
        try:
            allocated = torch.cuda.memory_allocated() / 1e9
            reserved = torch.cuda.memory_reserved() / 1e9
            total = torch.cuda.get_device_properties(0).total_memory / 1e9
            return {
                'allocated_gb': allocated,
                'reserved_gb': reserved,
                'total_gb': total,
                'free_gb': total - reserved,
            }
        except Exception:
            return {}
    
    @property
    def is_gpu(self) -> bool:
        """Check if using GPU."""
        return self.device.type == 'cuda'
    
    @property
    def gpu_name(self) -> Optional[str]:
        """Get GPU name if available."""
        return self._gpu_name


# ------------------------------------------------------------------------------
# Convenience Functions
# ------------------------------------------------------------------------------
def get_market_time(timezone: str = "US/Eastern") -> datetime:
    """
    Get current time in market timezone.
    
    Args:
        timezone: Timezone name
        
    Returns:
        Current datetime in market timezone (naive)
    """
    tz = pytz.timezone(timezone)
    market_dt = pytz.UTC.localize(datetime.utcnow()).astimezone(tz)
    return market_dt.replace(tzinfo=None)


def is_market_open(symbol: str = None, timezone: str = "US/Eastern") -> bool:
    """
    Check if market is currently open.
    
    Args:
        symbol: Symbol to check
        timezone: Market timezone
        
    Returns:
        True if market is open
    """
    return MarketHours(timezone).is_market_open(symbol)


def annualization_scale(interval: str) -> float:
    """
    Get annualization scale factor for volatility.
    
    Args:
        interval: Time interval ("1m", "5m", "15m", etc.)
        
    Returns:
        Scale factor for annualizing volatility
    """
    import numpy as np
    # 252 trading days * 390 minutes/day = steps per year (in minutes)
    minutes_per_year = 252 * 390
    interval_minutes = {
        "1m": 1, 
        "5m": 5, 
        "15m": 15,
        "30m": 30,
        "1h": 60,
        "1d": 390,
    }
    step = interval_minutes.get(interval, 1)
    return np.sqrt(minutes_per_year / step)


__all__ = [
    'MarketHours',
    'DeviceManager',
    'get_market_time',
    'is_market_open',
    'annualization_scale',
]








