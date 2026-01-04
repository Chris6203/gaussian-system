#!/usr/bin/env python3
"""
Unified Feature Pipeline

Combines all feature families into a single pipeline that can be used
for both training and inference.
"""

import numpy as np
import pandas as pd
from datetime import datetime, date, timedelta
from typing import Dict, List, Optional, Any, Set
from dataclasses import dataclass, field
import logging
import json

from .equity_etf import compute_equity_etf_features, get_latest_equity_etf_features, EQUITY_ETF_SYMBOLS
from .options_surface import compute_options_surface_features, compute_term_structure_features
from .breadth import compute_breadth_features, compute_sector_breadth, BREADTH_SYMBOLS
from .macro import (
    compute_macro_features,
    compute_extended_macro_features,
    MACRO_SYMBOLS,
    ALL_MACRO_SYMBOLS,
    EXTENDED_MACRO_SYMBOLS
)
from .crypto import compute_crypto_features, compute_crypto_equity_correlation, CRYPTO_SYMBOLS
from .meta import compute_meta_features
from .jerry_features import compute_jerry_features
from .sentiment import compute_all_sentiment_features, SENTIMENT_FEATURE_NAMES

logger = logging.getLogger(__name__)


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class FeatureConfig:
    """
    Configuration for feature pipeline.
    
    Controls which feature families are enabled and their parameters.
    """
    # Feature families to enable
    enable_equity_etf: bool = True
    enable_options_surface: bool = True
    enable_breadth: bool = True
    enable_macro: bool = True
    enable_extended_macro: bool = True  # Index proxies, sectors, credit, commodities, mega-caps
    enable_crypto: bool = True
    enable_meta: bool = True
    enable_jerry: bool = False  # Jerry's fuzzy logic features (Quantor-MTFuzz)
    enable_sentiment: bool = True  # Sentiment features (Fear & Greed, PCR, VIX, News)
    
    # Equity/ETF parameters
    equity_symbols: List[str] = field(default_factory=lambda: list(EQUITY_ETF_SYMBOLS.keys()))
    return_horizons: List[int] = field(default_factory=lambda: [1, 5, 15, 60])
    vol_windows: List[int] = field(default_factory=lambda: [20, 60, 120])
    ma_short: List[int] = field(default_factory=lambda: [10, 20])
    ma_long: List[int] = field(default_factory=lambda: [50, 200])
    
    # Breadth parameters
    breadth_symbols: List[str] = field(default_factory=lambda: BREADTH_SYMBOLS.copy())
    breadth_return_horizon: int = 5
    breadth_ma_window: int = 20
    
    # Macro parameters
    macro_symbols: List[str] = field(default_factory=lambda: list(MACRO_SYMBOLS.keys()))
    
    # Crypto parameters
    crypto_symbols: List[str] = field(default_factory=lambda: list(CRYPTO_SYMBOLS.keys()))
    
    # Primary symbol
    primary_symbol: str = 'SPY'
    
    # Feature prefixing
    use_prefix: bool = True
    
    @classmethod
    def from_dict(cls, config: Dict) -> 'FeatureConfig':
        """Create config from dictionary."""
        import os
        feature_config = config.get('feature_pipeline', {})

        # Check environment variable for Jerry features (overrides config)
        enable_jerry_env = os.environ.get('JERRY_FEATURES', None)
        if enable_jerry_env is not None:
            enable_jerry = enable_jerry_env == '1'
        else:
            enable_jerry = feature_config.get('enable_jerry', False)

        return cls(
            enable_equity_etf=feature_config.get('enable_equity_etf', True),
            enable_options_surface=feature_config.get('enable_options_surface', True),
            enable_breadth=feature_config.get('enable_breadth', True),
            enable_macro=feature_config.get('enable_macro', True),
            enable_extended_macro=feature_config.get('enable_extended_macro', True),
            enable_crypto=feature_config.get('enable_crypto', True),
            enable_meta=feature_config.get('enable_meta', True),
            enable_jerry=enable_jerry,
            enable_sentiment=feature_config.get('enable_sentiment', True),
            equity_symbols=feature_config.get('equity_symbols', list(EQUITY_ETF_SYMBOLS.keys())),
            macro_symbols=feature_config.get('macro_symbols', list(ALL_MACRO_SYMBOLS.keys())),
            primary_symbol=config.get('trading', {}).get('symbol', 'SPY'),
        )
    
    @classmethod
    def from_json(cls, filepath: str) -> 'FeatureConfig':
        """Load config from JSON file."""
        with open(filepath, 'r') as f:
            config = json.load(f)
        return cls.from_dict(config)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'enable_equity_etf': self.enable_equity_etf,
            'enable_options_surface': self.enable_options_surface,
            'enable_breadth': self.enable_breadth,
            'enable_macro': self.enable_macro,
            'enable_extended_macro': self.enable_extended_macro,
            'enable_crypto': self.enable_crypto,
            'enable_meta': self.enable_meta,
            'enable_jerry': self.enable_jerry,
            'enable_sentiment': self.enable_sentiment,
            'equity_symbols': self.equity_symbols,
            'breadth_symbols': self.breadth_symbols,
            'macro_symbols': self.macro_symbols,
            'crypto_symbols': self.crypto_symbols,
            'primary_symbol': self.primary_symbol,
            'return_horizons': self.return_horizons,
            'vol_windows': self.vol_windows,
        }


# =============================================================================
# FEATURE PIPELINE
# =============================================================================

class FeaturePipeline:
    """
    Unified feature pipeline for the SPY options trading bot.
    
    Combines all feature families:
    1. Cross-asset equity/ETF features
    2. Volatility & options surface features
    3. Breadth / cross-section approximations
    4. Rates / macro proxies
    5. Option positioning / put-call metrics
    6. Crypto risk proxies
    7. Meta / regime / time features
    
    Features:
    - Configurable via FeatureConfig
    - Supports both DataFrame output (training) and dict output (inference)
    - Handles missing data gracefully
    - Maintains feature column consistency
    """
    
    def __init__(self, config: Optional[FeatureConfig] = None):
        """
        Initialize feature pipeline.
        
        Args:
            config: Feature configuration (default: enable all)
        """
        self.config = config or FeatureConfig()
        self._feature_columns: Optional[List[str]] = None
        self._vol_history: List[float] = []
        self.logger = logging.getLogger(__name__)
    
    @property
    def feature_columns(self) -> List[str]:
        """Get list of feature columns (after first computation)."""
        return self._feature_columns or []
    
    @property
    def num_features(self) -> int:
        """Get number of features."""
        return len(self.feature_columns)
    
    def get_required_symbols(self) -> Set[str]:
        """Get set of all symbols required for feature computation."""
        symbols = {self.config.primary_symbol}

        if self.config.enable_equity_etf:
            symbols.update(self.config.equity_symbols)

        if self.config.enable_breadth:
            symbols.update(self.config.breadth_symbols)

        if self.config.enable_macro:
            if self.config.enable_extended_macro:
                # Include all macro symbols (original + extended)
                symbols.update(ALL_MACRO_SYMBOLS.keys())
            else:
                symbols.update(self.config.macro_symbols)

        if self.config.enable_crypto:
            symbols.update(self.config.crypto_symbols)

        return symbols
    
    def compute_features(
        self,
        equity_data: Dict[str, pd.DataFrame],
        options_chain: Optional[pd.DataFrame] = None,
        spot_price: Optional[float] = None,
        expiry_date: Optional[date] = None,
        timestamp: Optional[datetime] = None,
        vix_level: Optional[float] = None,
        as_dataframe: bool = False
    ) -> Dict[str, float]:
        """
        Compute all features for the current timestamp.
        
        Args:
            equity_data: Dict mapping symbol -> OHLCV DataFrame
            options_chain: SPY options chain DataFrame (optional)
            spot_price: Current SPY price (optional, derived from equity_data)
            expiry_date: Option expiry date for DTE features
            timestamp: Current timestamp (default: now)
            vix_level: Current VIX level (optional)
            as_dataframe: Return as single-row DataFrame instead of dict
            
        Returns:
            Dict (or DataFrame) with all feature values
        """
        features = {}
        timestamp = timestamp or datetime.now()
        
        # Get spot price from data if not provided
        if spot_price is None and self.config.primary_symbol in equity_data:
            primary_df = equity_data[self.config.primary_symbol]
            if not primary_df.empty:
                close_col = 'close' if 'close' in primary_df.columns else 'Close'
                if close_col in primary_df.columns:
                    spot_price = float(primary_df[close_col].iloc[-1])
        
        spot_price = spot_price or 0.0
        
        try:
            # =================================================================
            # 1. EQUITY/ETF FEATURES
            # =================================================================
            if self.config.enable_equity_etf:
                equity_features = get_latest_equity_etf_features(
                    data=equity_data,
                    primary_symbol=self.config.primary_symbol,
                    symbols=self.config.equity_symbols,
                    return_horizons=self.config.return_horizons,
                    vol_windows=self.config.vol_windows,
                    ma_short=self.config.ma_short,
                    ma_long=self.config.ma_long
                )
                if self.config.use_prefix:
                    equity_features = {f'eq_{k}': v for k, v in equity_features.items()}
                features.update(equity_features)
            
            # =================================================================
            # 2. OPTIONS SURFACE FEATURES
            # =================================================================
            if self.config.enable_options_surface and options_chain is not None:
                surface_features = compute_options_surface_features(
                    chain=options_chain,
                    spot_price=spot_price
                )
                if self.config.use_prefix:
                    surface_features = {f'opt_{k}': v for k, v in surface_features.items()}
                features.update(surface_features)
            
            # =================================================================
            # 3. BREADTH FEATURES
            # =================================================================
            if self.config.enable_breadth:
                breadth_features = compute_breadth_features(
                    data=equity_data,
                    symbols=self.config.breadth_symbols,
                    return_horizon=self.config.breadth_return_horizon,
                    ma_window=self.config.breadth_ma_window
                )
                if self.config.use_prefix:
                    breadth_features = {f'brd_{k}': v for k, v in breadth_features.items()}
                features.update(breadth_features)
                
                # Sector breadth
                sector_features = compute_sector_breadth(
                    data=equity_data,
                    return_horizon=self.config.breadth_return_horizon
                )
                if self.config.use_prefix:
                    sector_features = {f'sec_{k}': v for k, v in sector_features.items()}
                features.update(sector_features)
            
            # =================================================================
            # 4. MACRO FEATURES
            # =================================================================
            if self.config.enable_macro:
                if self.config.enable_extended_macro:
                    # Use extended macro features (includes original + new categories)
                    macro_features = compute_extended_macro_features(
                        data=equity_data,
                        include_original=True,
                        include_relative_strength=True,
                        include_credit=True,
                        include_sectors=True,
                        include_commodities=True,
                        include_megacaps=True,
                        include_index_proxies=True,
                    )
                else:
                    # Use original macro features only
                    macro_features = compute_macro_features(
                        data=equity_data,
                        return_horizons=[1, 5, 20],
                        vol_windows=[20, 60]
                    )
                if self.config.use_prefix:
                    macro_features = {f'mac_{k}': v for k, v in macro_features.items()}
                features.update(macro_features)
            
            # =================================================================
            # 5. CRYPTO FEATURES
            # =================================================================
            if self.config.enable_crypto:
                crypto_features = compute_crypto_features(
                    data=equity_data,  # Crypto symbols should be in equity_data
                    return_horizons=[1, 5, 24],
                    vol_windows=[24]
                )
                if self.config.use_prefix:
                    crypto_features = {f'cry_{k}': v for k, v in crypto_features.items()}
                features.update(crypto_features)
                
                # Crypto-equity correlation
                corr_features = compute_crypto_equity_correlation(
                    crypto_data=equity_data,
                    equity_data=equity_data
                )
                if self.config.use_prefix:
                    corr_features = {f'cry_{k}': v for k, v in corr_features.items()}
                features.update(corr_features)
            
            # =================================================================
            # 6. META FEATURES
            # =================================================================
            if self.config.enable_meta:
                # Compute realized vol for regime classification
                realized_vol = 0.0
                if self.config.primary_symbol in equity_data:
                    primary_df = equity_data[self.config.primary_symbol]
                    close_col = 'close' if 'close' in primary_df.columns else 'Close'
                    if close_col in primary_df.columns and len(primary_df) >= 21:
                        returns = primary_df[close_col].pct_change().dropna()
                        if len(returns) >= 20:
                            realized_vol = float(returns.tail(20).std())
                            self._vol_history.append(realized_vol)
                            # Keep last 100 vol observations
                            self._vol_history = self._vol_history[-100:]
                
                vol_history = np.array(self._vol_history) if len(self._vol_history) >= 20 else None
                
                meta_features = compute_meta_features(
                    timestamp=timestamp,
                    expiry_date=expiry_date,
                    realized_vol=realized_vol,
                    vol_history=vol_history,
                    vix_level=vix_level
                )
                if self.config.use_prefix:
                    meta_features = {f'meta_{k}': v for k, v in meta_features.items()}
                features.update(meta_features)

            # =================================================================
            # 7. JERRY FEATURES (Quantor-MTFuzz fuzzy logic)
            # =================================================================
            if self.config.enable_jerry:
                # Get volume ratio for liquidity score
                volume_ratio = 1.0
                if self.config.primary_symbol in equity_data:
                    primary_df = equity_data[self.config.primary_symbol]
                    vol_col = 'volume' if 'volume' in primary_df.columns else 'Volume'
                    if vol_col in primary_df.columns and len(primary_df) >= 20:
                        recent_vol = float(primary_df[vol_col].iloc[-1])
                        avg_vol = float(primary_df[vol_col].tail(20).mean())
                        if avg_vol > 0:
                            volume_ratio = recent_vol / avg_vol

                # Get bars in regime (approximate from vol history stability)
                bars_in_regime = min(len(self._vol_history), 30)

                jerry_feats = compute_jerry_features(
                    vix_level=vix_level or 18.0,
                    iv_percentile=None,  # Could add from options surface
                    bars_in_regime=bars_in_regime,
                    regime_changes=0,  # Could track from HMM
                    multi_horizon_predictions=None,  # Added at inference time
                    bid_ask_spread_pct=1.0,  # Default
                    volume_ratio=volume_ratio,
                    portfolio_delta=0.0,  # Not available here
                    hmm_confidence=0.5,  # Updated at inference time
                )

                jerry_dict = jerry_feats.to_dict()
                features.update(jerry_dict)

            # =================================================================
            # 8. SENTIMENT FEATURES (Fear & Greed, PCR, VIX, News)
            # =================================================================
            if self.config.enable_sentiment:
                # Get PCR from options surface features if available
                oi_put_call_ratio = features.get('opt_oi_put_call_ratio', 1.0)
                vol_put_call_ratio = features.get('opt_vol_put_call_ratio', None)

                sentiment_features = compute_all_sentiment_features(
                    vix_level=vix_level or 18.0,
                    oi_put_call_ratio=oi_put_call_ratio,
                    vol_put_call_ratio=vol_put_call_ratio,
                    fetch_external=True,  # Fetch Fear & Greed and Polygon news
                    polygon_api_key=None  # Will use config/env
                )
                if self.config.use_prefix:
                    sentiment_features = {f'sent_{k}': v for k, v in sentiment_features.items()}
                features.update(sentiment_features)

        except Exception as e:
            self.logger.error(f"Error computing features: {e}")
        
        # Update known columns
        if self._feature_columns is None:
            self._feature_columns = sorted(features.keys())
        
        # Ensure consistent columns
        result = {col: features.get(col, 0.0) for col in self._feature_columns}
        
        # Add any new columns
        for col in features:
            if col not in result:
                result[col] = features[col]
                if col not in self._feature_columns:
                    self._feature_columns.append(col)
                    self._feature_columns.sort()
        
        if as_dataframe:
            return pd.DataFrame([result], index=[timestamp])
        
        return result
    
    def compute_features_batch(
        self,
        equity_data: Dict[str, pd.DataFrame],
        options_chains: Optional[Dict[str, pd.DataFrame]] = None,
        timestamps: Optional[List[datetime]] = None
    ) -> pd.DataFrame:
        """
        Compute features for multiple timestamps (batch mode for training).
        
        Args:
            equity_data: Dict mapping symbol -> OHLCV DataFrame
            options_chains: Dict mapping timestamp/expiry -> chain DataFrame
            timestamps: List of timestamps to compute features for
            
        Returns:
            DataFrame with features for all timestamps
        """
        if not equity_data or self.config.primary_symbol not in equity_data:
            return pd.DataFrame()
        
        primary_df = equity_data[self.config.primary_symbol]
        
        if primary_df.empty:
            return pd.DataFrame()
        
        # Use primary DataFrame index as timestamps
        if timestamps is None:
            timestamps = list(primary_df.index)
        
        all_features = []
        
        for ts in timestamps:
            try:
                # Get data up to this timestamp
                data_slice = {}
                for symbol, df in equity_data.items():
                    mask = df.index <= ts
                    if mask.any():
                        data_slice[symbol] = df[mask]
                
                # Get options chain if available
                chain = None
                if options_chains:
                    # Find matching chain by timestamp
                    ts_str = str(ts)[:10]  # Date portion
                    for key in options_chains:
                        if ts_str in str(key):
                            chain = options_chains[key]
                            break
                
                # Get spot price
                spot = None
                if self.config.primary_symbol in data_slice:
                    ps_df = data_slice[self.config.primary_symbol]
                    if not ps_df.empty:
                        close_col = 'close' if 'close' in ps_df.columns else 'Close'
                        if close_col in ps_df.columns:
                            spot = float(ps_df[close_col].iloc[-1])
                
                features = self.compute_features(
                    equity_data=data_slice,
                    options_chain=chain,
                    spot_price=spot,
                    timestamp=ts if isinstance(ts, datetime) else None
                )
                features['timestamp'] = ts
                all_features.append(features)
                
            except Exception as e:
                self.logger.debug(f"Error at timestamp {ts}: {e}")
                continue
        
        if not all_features:
            return pd.DataFrame()
        
        result = pd.DataFrame(all_features)
        result = result.set_index('timestamp')
        
        return result
    
    def get_feature_array(
        self,
        equity_data: Dict[str, pd.DataFrame],
        options_chain: Optional[pd.DataFrame] = None,
        spot_price: Optional[float] = None,
        **kwargs
    ) -> np.ndarray:
        """
        Get features as numpy array (for model input).
        
        Args:
            equity_data: Dict mapping symbol -> OHLCV DataFrame
            options_chain: SPY options chain DataFrame
            spot_price: Current SPY price
            **kwargs: Additional arguments passed to compute_features
            
        Returns:
            1D numpy array of feature values
        """
        features = self.compute_features(
            equity_data=equity_data,
            options_chain=options_chain,
            spot_price=spot_price,
            **kwargs
        )
        
        # Ensure consistent ordering
        values = [features.get(col, 0.0) for col in self._feature_columns or sorted(features.keys())]
        
        return np.array(values, dtype=np.float32)
    
    def describe(self) -> str:
        """Get description of enabled features."""
        lines = ["Feature Pipeline Configuration:"]
        lines.append(f"  Primary Symbol: {self.config.primary_symbol}")
        lines.append(f"  Equity/ETF: {'enabled' if self.config.enable_equity_etf else 'disabled'}")
        lines.append(f"  Options Surface: {'enabled' if self.config.enable_options_surface else 'disabled'}")
        lines.append(f"  Breadth: {'enabled' if self.config.enable_breadth else 'disabled'}")
        lines.append(f"  Macro: {'enabled' if self.config.enable_macro else 'disabled'}")
        if self.config.enable_macro:
            lines.append(f"    Extended Macro: {'enabled' if self.config.enable_extended_macro else 'disabled'}")
            if self.config.enable_extended_macro:
                lines.append(f"    (includes sectors, credit, commodities, mega-caps)")
        lines.append(f"  Crypto: {'enabled' if self.config.enable_crypto else 'disabled'}")
        lines.append(f"  Meta: {'enabled' if self.config.enable_meta else 'disabled'}")
        lines.append(f"  Jerry (Quantor-MTFuzz): {'enabled' if self.config.enable_jerry else 'disabled'}")

        if self._feature_columns:
            lines.append(f"\nTotal Features: {len(self._feature_columns)}")
            lines.append("\nFeature Columns:")
            for col in self._feature_columns[:20]:
                lines.append(f"  - {col}")
            if len(self._feature_columns) > 20:
                lines.append(f"  ... and {len(self._feature_columns) - 20} more")
        
        return '\n'.join(lines)


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def create_pipeline(config_path: str = 'config.json') -> FeaturePipeline:
    """Create feature pipeline from config file."""
    try:
        config = FeatureConfig.from_json(config_path)
    except FileNotFoundError:
        config = FeatureConfig()
    
    return FeaturePipeline(config)


def get_all_feature_names() -> List[str]:
    """Get list of all possible feature names (for documentation)."""
    # This is a representative list; actual features depend on config
    features = []
    
    # Equity/ETF pattern
    for symbol in ['SPY', 'QQQ', 'IWM']:
        for suffix in ['ret_1', 'ret_5', 'rvol_20', 'price_zscore_20', 'ma10_over_ma50']:
            features.append(f'eq_{symbol}_{suffix}')
    
    # Options surface
    for suffix in ['atm_iv_avg', 'iv_skew_otm', 'oi_put_call_ratio', 'gex_total']:
        features.append(f'opt_{suffix}')
    
    # Breadth
    for suffix in ['breadth_up_frac_short', 'xsec_ret_mean', 'breadth_divergence']:
        features.append(f'brd_{suffix}')
    
    # Macro
    for suffix in ['tlt_ret_5', 'uup_ret_5', 'bond_equity_corr']:
        features.append(f'mac_{suffix}')
    
    # Crypto
    for suffix in ['btc_ret_5', 'crypto_risk_on', 'btc_spy_corr']:
        features.append(f'cry_{suffix}')
    
    # Meta
    for suffix in ['session_progress', 'days_to_expiry', 'vol_regime_high']:
        features.append(f'meta_{suffix}')
    
    return features


if __name__ == "__main__":
    """Test the feature pipeline."""
    import logging
    
    logging.basicConfig(level=logging.INFO)
    
    print("=" * 70)
    print("FEATURE PIPELINE TEST")
    print("=" * 70)
    
    # Create mock data
    np.random.seed(42)
    n_bars = 200
    
    # Generate base trend
    base_trend = np.cumsum(np.random.randn(n_bars) * 0.002)
    
    symbols = ['SPY', 'QQQ', 'IWM', 'XLF', 'TLT', 'UUP', 'BTC-USD']
    equity_data = {}
    
    for i, symbol in enumerate(symbols):
        # Add symbol-specific noise
        noise = np.random.randn(n_bars) * (0.005 if 'BTC' not in symbol else 0.02)
        multiplier = [450, 380, 200, 40, 100, 27, 50000][i]
        prices = multiplier * np.exp(base_trend + noise)
        
        df = pd.DataFrame({
            'open': prices * (1 + np.random.randn(n_bars) * 0.001),
            'high': prices * (1 + np.abs(np.random.randn(n_bars) * 0.002)),
            'low': prices * (1 - np.abs(np.random.randn(n_bars) * 0.002)),
            'close': prices,
            'volume': np.random.randint(1000000, 10000000, n_bars),
        }, index=pd.date_range('2024-01-01', periods=n_bars, freq='1h'))
        
        equity_data[symbol] = df
    
    # Create mock options chain
    spot = 450.0
    strikes = np.arange(430, 470, 2.5)
    
    records = []
    for strike in strikes:
        for opt_type in ['call', 'put']:
            records.append({
                'type': opt_type,
                'strike': strike,
                'expiration': '2024-01-19',
                'bid': 2.0,
                'ask': 2.5,
                'volume': np.random.randint(100, 5000),
                'open_interest': np.random.randint(1000, 50000),
                'delta': 0.5 if opt_type == 'call' else -0.5,
                'gamma': 0.02,
                'mid_iv': 0.15,
            })
    
    options_chain = pd.DataFrame(records)
    
    # Create pipeline
    config = FeatureConfig(
        enable_equity_etf=True,
        enable_options_surface=True,
        enable_breadth=True,
        enable_macro=True,
        enable_crypto=True,
        enable_meta=True,
        equity_symbols=['SPY', 'QQQ', 'IWM', 'XLF'],
        breadth_symbols=['SPY', 'QQQ', 'IWM', 'XLF'],
        macro_symbols=['TLT', 'UUP'],
        crypto_symbols=['BTC-USD'],
    )
    
    pipeline = FeaturePipeline(config)
    
    print("\n" + pipeline.describe())
    
    # Compute features
    print("\n--- Computing Single Timestamp Features ---")
    features = pipeline.compute_features(
        equity_data=equity_data,
        options_chain=options_chain,
        spot_price=spot,
        expiry_date=date(2024, 1, 19),
        vix_level=18.5
    )
    
    print(f"\nGenerated {len(features)} features")
    print("\nSample features:")
    for k, v in list(features.items())[:20]:
        print(f"  {k}: {v:.4f}")
    
    # Compute batch
    print("\n--- Computing Batch Features ---")
    batch_features = pipeline.compute_features_batch(
        equity_data=equity_data,
        timestamps=list(equity_data['SPY'].index[-10:])
    )
    print(f"Batch shape: {batch_features.shape}")
    
    # Get as array
    print("\n--- Feature Array ---")
    arr = pipeline.get_feature_array(
        equity_data=equity_data,
        options_chain=options_chain,
        spot_price=spot
    )
    print(f"Array shape: {arr.shape}")
    print(f"Array dtype: {arr.dtype}")
    
    print("\n" + "=" * 70)
    print("TEST COMPLETE")
    print("=" * 70)




