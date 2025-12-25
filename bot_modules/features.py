"""
Feature Computation Module
==========================

Provides clean interfaces for computing trading features:

- Technical indicators (RSI, MACD, Bollinger Bands)
- Volatility metrics (ATR, Parkinson, Garman-Klass)
- Momentum features (price changes, volume spikes)
- Cross-asset features (correlation, relative strength)
- Market regime features (HMM states, VIX levels)

Usage:
    from bot_modules.features import FeatureComputer, compute_technical_features
    
    computer = FeatureComputer(config)
    features = computer.compute(price_data, volume_data)
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class FeatureConfig:
    """Configuration for feature computation."""
    rsi_period: int = 14
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    bb_period: int = 20
    bb_std: float = 2.0
    atr_period: int = 14
    volume_ma_period: int = 20
    momentum_periods: List[int] = None
    
    def __post_init__(self):
        if self.momentum_periods is None:
            self.momentum_periods = [5, 10, 20]


class FeatureComputer:
    """
    Centralized feature computation with caching.
    
    Computes all features needed for the trading model:
    - Price-based: returns, momentum, trend
    - Volatility: ATR, realized vol, implied vol (if available)
    - Technical: RSI, MACD, Bollinger Bands
    - Volume: relative volume, volume spikes
    - Cross-asset: correlations, relative performance
    """
    
    def __init__(self, config: FeatureConfig = None):
        """
        Args:
            config: Feature configuration (uses defaults if None)
        """
        self.config = config or FeatureConfig()
        self._cache = {}
    
    def compute_all(
        self,
        data: pd.DataFrame,
        context_data: Dict[str, pd.DataFrame] = None
    ) -> np.ndarray:
        """
        Compute all features from price data.
        
        Args:
            data: OHLCV DataFrame for primary symbol
            context_data: Optional dict of DataFrames for context symbols
            
        Returns:
            Feature vector as numpy array
        """
        features = []
        
        # Price features
        features.extend(self._compute_price_features(data))
        
        # Volatility features
        features.extend(self._compute_volatility_features(data))
        
        # Technical features
        features.extend(self._compute_technical_features(data))
        
        # Volume features
        features.extend(self._compute_volume_features(data))
        
        # Cross-asset features (if context data available)
        if context_data:
            features.extend(self._compute_cross_asset_features(data, context_data))
        
        return np.array(features, dtype=np.float32)
    
    def _compute_price_features(self, data: pd.DataFrame) -> List[float]:
        """Compute price-based features."""
        close = data['close'].values if 'close' in data.columns else data['Close'].values
        
        features = []
        
        # Returns at multiple horizons
        for period in self.config.momentum_periods:
            if len(close) > period:
                ret = (close[-1] / close[-period - 1] - 1) * 100
                features.append(float(np.clip(ret, -10, 10)))
            else:
                features.append(0.0)
        
        # Current price normalized
        if len(close) > 20:
            mean_price = np.mean(close[-20:])
            norm_price = (close[-1] / mean_price - 1) * 100
            features.append(float(np.clip(norm_price, -10, 10)))
        else:
            features.append(0.0)
        
        return features
    
    def _compute_volatility_features(self, data: pd.DataFrame) -> List[float]:
        """Compute volatility features."""
        high = data['high'].values if 'high' in data.columns else data['High'].values
        low = data['low'].values if 'low' in data.columns else data['Low'].values
        close = data['close'].values if 'close' in data.columns else data['Close'].values
        
        features = []
        
        # ATR
        if len(close) > self.config.atr_period:
            tr = np.maximum(
                high[-self.config.atr_period:] - low[-self.config.atr_period:],
                np.maximum(
                    np.abs(high[-self.config.atr_period:] - np.roll(close, 1)[-self.config.atr_period:]),
                    np.abs(low[-self.config.atr_period:] - np.roll(close, 1)[-self.config.atr_period:])
                )
            )
            atr = np.mean(tr[1:])  # Skip first element (invalid due to roll)
            atr_pct = (atr / close[-1]) * 100
            features.append(float(np.clip(atr_pct, 0, 10)))
        else:
            features.append(1.0)  # Default 1% ATR
        
        # Parkinson volatility
        if len(close) > 20:
            hl_ratio = np.log(high[-20:] / low[-20:])
            parkinson = np.sqrt(np.mean(hl_ratio ** 2) / (4 * np.log(2)))
            features.append(float(np.clip(parkinson * 100, 0, 10)))
        else:
            features.append(1.0)
        
        # Realized volatility
        if len(close) > 20:
            returns = np.diff(np.log(close[-21:]))
            realized_vol = np.std(returns) * np.sqrt(252 * 390) * 100  # Annualized
            features.append(float(np.clip(realized_vol, 0, 100)))
        else:
            features.append(15.0)  # Default ~15% vol
        
        return features
    
    def _compute_technical_features(self, data: pd.DataFrame) -> List[float]:
        """Compute technical indicator features."""
        close = data['close'].values if 'close' in data.columns else data['Close'].values
        
        features = []
        
        # RSI
        rsi = self._compute_rsi(close, self.config.rsi_period)
        features.append(float((rsi - 50) / 50))  # Normalized to [-1, 1]
        
        # MACD
        macd, signal = self._compute_macd(close)
        macd_hist = macd - signal
        features.append(float(np.clip(macd_hist * 100, -5, 5)))
        
        # Bollinger Bands position
        bb_pos = self._compute_bb_position(close)
        features.append(float(np.clip(bb_pos, -2, 2)))
        
        return features
    
    def _compute_volume_features(self, data: pd.DataFrame) -> List[float]:
        """Compute volume-based features."""
        volume = data['volume'].values if 'volume' in data.columns else data['Volume'].values
        
        features = []
        
        # Relative volume
        if len(volume) > self.config.volume_ma_period:
            vol_ma = np.mean(volume[-self.config.volume_ma_period:])
            if vol_ma > 0:
                rel_vol = volume[-1] / vol_ma
                features.append(float(np.clip(rel_vol, 0, 5)))
            else:
                features.append(1.0)
        else:
            features.append(1.0)
        
        # Volume trend
        if len(volume) > 10:
            recent_vol = np.mean(volume[-5:])
            older_vol = np.mean(volume[-10:-5])
            if older_vol > 0:
                vol_trend = recent_vol / older_vol - 1
                features.append(float(np.clip(vol_trend, -1, 1)))
            else:
                features.append(0.0)
        else:
            features.append(0.0)
        
        return features
    
    def _compute_cross_asset_features(
        self,
        data: pd.DataFrame,
        context_data: Dict[str, pd.DataFrame]
    ) -> List[float]:
        """Compute cross-asset features."""
        close = data['close'].values if 'close' in data.columns else data['Close'].values
        
        features = []
        
        # Correlation with major indices
        for symbol in ['QQQ', 'IWM', 'DIA']:
            if symbol in context_data:
                ctx_close = context_data[symbol]['close'].values if 'close' in context_data[symbol].columns else context_data[symbol]['Close'].values
                if len(ctx_close) >= 20 and len(close) >= 20:
                    corr = np.corrcoef(close[-20:], ctx_close[-20:])[0, 1]
                    features.append(float(corr) if not np.isnan(corr) else 0.0)
                else:
                    features.append(0.0)
            else:
                features.append(0.0)
        
        return features
    
    def _compute_rsi(self, prices: np.ndarray, period: int = 14) -> float:
        """Compute RSI."""
        if len(prices) < period + 1:
            return 50.0
        
        deltas = np.diff(prices[-period-1:])
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        avg_gain = np.mean(gains)
        avg_loss = np.mean(losses)
        
        if avg_loss == 0:
            return 100.0 if avg_gain > 0 else 50.0
        
        rs = avg_gain / avg_loss
        return 100 - (100 / (1 + rs))
    
    def _compute_macd(self, prices: np.ndarray) -> Tuple[float, float]:
        """Compute MACD and signal line."""
        if len(prices) < self.config.macd_slow + self.config.macd_signal:
            return 0.0, 0.0
        
        # EMA calculation
        def ema(data, period):
            alpha = 2 / (period + 1)
            result = [data[0]]
            for i in range(1, len(data)):
                result.append(alpha * data[i] + (1 - alpha) * result[-1])
            return np.array(result)
        
        ema_fast = ema(prices, self.config.macd_fast)
        ema_slow = ema(prices, self.config.macd_slow)
        macd_line = ema_fast - ema_slow
        signal_line = ema(macd_line, self.config.macd_signal)
        
        return float(macd_line[-1]), float(signal_line[-1])
    
    def _compute_bb_position(self, prices: np.ndarray) -> float:
        """Compute position within Bollinger Bands."""
        if len(prices) < self.config.bb_period:
            return 0.0
        
        period_prices = prices[-self.config.bb_period:]
        middle = np.mean(period_prices)
        std = np.std(period_prices)
        
        if std == 0:
            return 0.0
        
        upper = middle + self.config.bb_std * std
        lower = middle - self.config.bb_std * std
        
        current = prices[-1]
        band_width = upper - lower
        
        if band_width == 0:
            return 0.0
        
        # Position: -1 at lower band, 0 at middle, +1 at upper band
        return (current - middle) / (band_width / 2)


# Convenience functions
def compute_technical_features(data: pd.DataFrame, config: FeatureConfig = None) -> np.ndarray:
    """Compute technical features from price data."""
    computer = FeatureComputer(config)
    return computer._compute_technical_features(data)


def compute_volatility_features(data: pd.DataFrame, config: FeatureConfig = None) -> np.ndarray:
    """Compute volatility features from price data."""
    computer = FeatureComputer(config)
    return computer._compute_volatility_features(data)


def compute_all_features(
    data: pd.DataFrame,
    context_data: Dict[str, pd.DataFrame] = None,
    config: FeatureConfig = None
) -> np.ndarray:
    """Compute all features from price data."""
    computer = FeatureComputer(config)
    return computer.compute_all(data, context_data)









