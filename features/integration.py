#!/usr/bin/env python3
"""
Feature Pipeline Integration

Integrates the new feature pipeline with the existing UnifiedOptionsBot.
Provides a drop-in enhancement that preserves existing functionality.
"""

import numpy as np
import pandas as pd
from datetime import datetime, date, timedelta
from typing import Dict, List, Optional, Any, Tuple
import logging
import json
from pathlib import Path

logger = logging.getLogger(__name__)

# Import the new feature pipeline
try:
    from .pipeline import FeaturePipeline, FeatureConfig
    from .equity_etf import EQUITY_ETF_SYMBOLS
    from .breadth import BREADTH_SYMBOLS
    from .macro import MACRO_SYMBOLS
    from .crypto import CRYPTO_SYMBOLS
    FEATURE_PIPELINE_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Feature pipeline not available: {e}")
    FEATURE_PIPELINE_AVAILABLE = False


class EnhancedFeatureManager:
    """
    Enhanced feature manager that combines original bot features with new pipeline.
    
    This class:
    1. Maintains backward compatibility with existing features
    2. Adds new feature families when enabled
    3. Provides unified interface for feature computation
    """
    
    def __init__(
        self,
        config: Optional[Dict] = None,
        config_path: str = 'config.json',
        enable_enhanced_features: bool = True
    ):
        """
        Initialize enhanced feature manager.
        
        Args:
            config: Configuration dict (loaded from file if not provided)
            config_path: Path to config.json
            enable_enhanced_features: Whether to enable new feature pipeline
        """
        self.logger = logging.getLogger(__name__)
        
        # Load config
        if config is None:
            try:
                with open(config_path, 'r') as f:
                    config = json.load(f)
            except FileNotFoundError:
                config = {}
        
        self.config = config
        self.enable_enhanced = enable_enhanced_features and FEATURE_PIPELINE_AVAILABLE
        
        # Initialize feature pipeline
        self._pipeline: Optional[FeaturePipeline] = None
        
        if self.enable_enhanced:
            try:
                feature_config = FeatureConfig.from_dict(config)
                self._pipeline = FeaturePipeline(feature_config)
                self.logger.info("[FEATURES] Enhanced feature pipeline initialized")
            except Exception as e:
                self.logger.warning(f"[FEATURES] Could not initialize pipeline: {e}")
                self._pipeline = None
        
        # Cache for market data
        self._data_cache: Dict[str, pd.DataFrame] = {}
        self._options_chain_cache: Optional[pd.DataFrame] = None
        self._spot_price: float = 0.0
        self._vix_level: float = 0.0
        
        # Feature history for consistency
        self._enhanced_feature_names: Optional[List[str]] = None
    
    @property
    def is_enhanced_enabled(self) -> bool:
        """Check if enhanced features are available."""
        return self._pipeline is not None
    
    @property
    def enhanced_feature_count(self) -> int:
        """Get number of enhanced features."""
        if self._pipeline and self._pipeline.feature_columns:
            return len(self._pipeline.feature_columns)
        return 0
    
    def get_required_symbols(self) -> List[str]:
        """Get list of symbols required for feature computation."""
        symbols = set()
        
        # Primary symbol
        symbols.add(self.config.get('trading', {}).get('symbol', 'SPY'))
        
        # From config
        symbols.update(self.config.get('data_fetching', {}).get('symbols', []))
        
        # From pipeline
        if self._pipeline:
            symbols.update(self._pipeline.get_required_symbols())
        
        return sorted(list(symbols))
    
    def update_data(self, symbol: str, data: pd.DataFrame):
        """
        Update cached data for a symbol.
        
        Args:
            symbol: Symbol name
            data: OHLCV DataFrame
        """
        if data is not None and not data.empty:
            self._data_cache[symbol] = data.copy()
            
            # Update spot and VIX if relevant
            primary = self.config.get('trading', {}).get('symbol', 'SPY')
            close_col = 'close' if 'close' in data.columns else 'Close'
            
            if symbol == primary and close_col in data.columns:
                self._spot_price = float(data[close_col].iloc[-1])
            
            if symbol in ['VIX', '^VIX'] and close_col in data.columns:
                self._vix_level = float(data[close_col].iloc[-1])
    
    def update_options_chain(self, chain: pd.DataFrame):
        """
        Update cached options chain.
        
        Args:
            chain: Options chain DataFrame
        """
        if chain is not None and not chain.empty:
            self._options_chain_cache = chain.copy()
    
    def compute_enhanced_features(
        self,
        timestamp: Optional[datetime] = None,
        expiry_date: Optional[date] = None
    ) -> Dict[str, float]:
        """
        Compute enhanced features using the new pipeline.
        
        Args:
            timestamp: Current timestamp
            expiry_date: Option expiry date
            
        Returns:
            Dict of feature name -> value
        """
        if not self._pipeline:
            return {}
        
        try:
            features = self._pipeline.compute_features(
                equity_data=self._data_cache,
                options_chain=self._options_chain_cache,
                spot_price=self._spot_price,
                expiry_date=expiry_date,
                timestamp=timestamp,
                vix_level=self._vix_level
            )
            
            # Cache feature names for consistency
            if self._enhanced_feature_names is None:
                self._enhanced_feature_names = sorted(features.keys())
            
            return features
            
        except Exception as e:
            self.logger.warning(f"[FEATURES] Error computing enhanced features: {e}")
            return {}
    
    def get_enhanced_feature_array(
        self,
        timestamp: Optional[datetime] = None,
        expiry_date: Optional[date] = None
    ) -> np.ndarray:
        """
        Get enhanced features as numpy array.
        
        Args:
            timestamp: Current timestamp
            expiry_date: Option expiry date
            
        Returns:
            1D numpy array of feature values
        """
        features = self.compute_enhanced_features(timestamp, expiry_date)
        
        if not features:
            return np.array([], dtype=np.float32)
        
        # Use consistent ordering
        if self._enhanced_feature_names:
            values = [features.get(name, 0.0) for name in self._enhanced_feature_names]
        else:
            values = list(features.values())
        
        return np.array(values, dtype=np.float32)
    
    def augment_existing_features(
        self,
        original_features: np.ndarray,
        timestamp: Optional[datetime] = None,
        expiry_date: Optional[date] = None,
        max_new_features: int = 50
    ) -> np.ndarray:
        """
        Augment existing feature array with enhanced features.
        
        This preserves backward compatibility - original features remain
        in their original positions, new features are appended.
        
        Args:
            original_features: Original feature array from existing pipeline
            timestamp: Current timestamp
            expiry_date: Option expiry date
            max_new_features: Maximum number of new features to add
            
        Returns:
            Augmented feature array
        """
        if not self._pipeline:
            return original_features
        
        try:
            enhanced = self.get_enhanced_feature_array(timestamp, expiry_date)
            
            if len(enhanced) == 0:
                return original_features
            
            # Limit number of new features
            if len(enhanced) > max_new_features:
                enhanced = enhanced[:max_new_features]
            
            # Concatenate
            return np.concatenate([original_features, enhanced])
            
        except Exception as e:
            self.logger.warning(f"[FEATURES] Error augmenting features: {e}")
            return original_features
    
    def get_cross_asset_features_enhanced(
        self,
        primary_data: pd.DataFrame
    ) -> Dict[str, float]:
        """
        Get enhanced cross-asset features.
        
        This can replace or augment the existing get_cross_asset_features() method.
        
        Args:
            primary_data: Primary symbol (SPY) DataFrame
            
        Returns:
            Dict of cross-asset features
        """
        features = {}
        
        # Compute enhanced features if available
        if self._pipeline:
            # Update primary data
            primary = self.config.get('trading', {}).get('symbol', 'SPY')
            self.update_data(primary, primary_data)
            
            # Get features
            enhanced = self.compute_enhanced_features()
            
            # Extract relevant cross-asset features
            cross_asset_prefixes = ['eq_', 'brd_', 'cry_', 'mac_']
            for key, value in enhanced.items():
                for prefix in cross_asset_prefixes:
                    if key.startswith(prefix):
                        features[key] = value
                        break
        
        return features
    
    def describe(self) -> str:
        """Get description of feature manager state."""
        lines = ["Enhanced Feature Manager:"]
        lines.append(f"  Enhanced Features: {'enabled' if self.is_enhanced_enabled else 'disabled'}")
        lines.append(f"  Cached Symbols: {len(self._data_cache)}")
        lines.append(f"  Options Chain: {'cached' if self._options_chain_cache is not None else 'not cached'}")
        lines.append(f"  Spot Price: ${self._spot_price:.2f}")
        lines.append(f"  VIX Level: {self._vix_level:.2f}")
        
        if self._pipeline:
            lines.append(f"  Enhanced Features: {self.enhanced_feature_count}")
        
        return '\n'.join(lines)


def integrate_with_bot(
    bot_instance: Any,
    config_path: str = 'config.json'
) -> EnhancedFeatureManager:
    """
    Integrate enhanced features with an existing UnifiedOptionsBot instance.
    
    This function sets up the enhanced feature manager and hooks it into
    the bot's data flow.
    
    Args:
        bot_instance: UnifiedOptionsBot instance
        config_path: Path to config file
        
    Returns:
        EnhancedFeatureManager instance
    """
    # Load config
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
    except FileNotFoundError:
        config = {}
    
    # Check if enhanced features are enabled
    feature_config = config.get('feature_pipeline', {})
    enabled = feature_config.get('enabled', True)
    
    # Create manager
    manager = EnhancedFeatureManager(
        config=config,
        enable_enhanced_features=enabled
    )
    
    # Attach to bot
    bot_instance._enhanced_feature_manager = manager
    
    logger.info(f"[INTEGRATION] Enhanced feature manager attached to bot")
    logger.info(f"[INTEGRATION] Required symbols: {manager.get_required_symbols()}")
    
    return manager


def patch_create_features(bot_instance: Any, manager: EnhancedFeatureManager):
    """
    Patch the bot's create_features method to include enhanced features.
    
    This preserves the original feature computation and appends enhanced features.
    
    Args:
        bot_instance: UnifiedOptionsBot instance
        manager: EnhancedFeatureManager instance
    """
    # Store original method
    original_create_features = bot_instance.create_features
    
    def enhanced_create_features(data: pd.DataFrame) -> Optional[np.ndarray]:
        """Enhanced create_features that includes new feature families."""
        # Call original
        original_features = original_create_features(data)
        
        if original_features is None:
            return None
        
        # Update manager with current data
        primary = bot_instance.config.get('trading', {}).get('symbol', 'SPY')
        manager.update_data(primary, data)
        
        # Augment with enhanced features
        try:
            augmented = manager.augment_existing_features(
                original_features,
                max_new_features=50  # Limit to avoid dimension explosion
            )
            return augmented
        except Exception as e:
            logger.warning(f"[INTEGRATION] Feature augmentation failed: {e}")
            return original_features
    
    # Replace method
    bot_instance.create_features = enhanced_create_features
    logger.info("[INTEGRATION] create_features method patched with enhanced features")


def patch_update_market_context(bot_instance: Any, manager: EnhancedFeatureManager):
    """
    Patch the bot's update_market_context method to update enhanced feature manager.
    
    Args:
        bot_instance: UnifiedOptionsBot instance
        manager: EnhancedFeatureManager instance
    """
    # Store original method
    original_update = bot_instance.update_market_context
    
    def enhanced_update_market_context(symbol: str, data: pd.DataFrame):
        """Enhanced update_market_context that also updates feature manager."""
        # Call original
        original_update(symbol, data)
        
        # Update feature manager
        manager.update_data(symbol, data)
    
    # Replace method
    bot_instance.update_market_context = enhanced_update_market_context
    logger.info("[INTEGRATION] update_market_context method patched")


# =============================================================================
# USAGE EXAMPLE
# =============================================================================

if __name__ == "__main__":
    """Example usage of enhanced feature integration."""
    import logging
    
    logging.basicConfig(level=logging.INFO)
    
    print("=" * 70)
    print("ENHANCED FEATURE INTEGRATION EXAMPLE")
    print("=" * 70)
    
    # Create mock data
    np.random.seed(42)
    n_bars = 100
    
    # Create feature manager
    manager = EnhancedFeatureManager(enable_enhanced_features=True)
    
    # Generate mock data for various symbols
    symbols = ['SPY', 'QQQ', 'IWM', 'TLT', 'UUP', 'BTC-USD']
    
    for symbol in symbols:
        prices = 100 * np.exp(np.cumsum(np.random.randn(n_bars) * 0.01))
        df = pd.DataFrame({
            'open': prices * 0.99,
            'high': prices * 1.01,
            'low': prices * 0.98,
            'close': prices,
            'volume': np.random.randint(1000000, 10000000, n_bars),
        }, index=pd.date_range('2024-01-01', periods=n_bars, freq='1h'))
        
        manager.update_data(symbol, df)
    
    print(f"\n{manager.describe()}")
    
    # Compute enhanced features
    print("\n--- Computing Enhanced Features ---")
    features = manager.compute_enhanced_features()
    print(f"Generated {len(features)} enhanced features")
    
    # Show sample features
    print("\nSample features:")
    for k, v in list(features.items())[:15]:
        print(f"  {k}: {v:.4f}")
    
    # Get as array
    arr = manager.get_enhanced_feature_array()
    print(f"\nFeature array shape: {arr.shape}")
    
    # Augment existing features
    print("\n--- Augmenting Existing Features ---")
    original = np.random.randn(50).astype(np.float32)  # Mock original features
    augmented = manager.augment_existing_features(original)
    print(f"Original: {original.shape[0]} features")
    print(f"Augmented: {augmented.shape[0]} features")
    
    print("\n" + "=" * 70)
    print("INTEGRATION EXAMPLE COMPLETE")
    print("=" * 70)




