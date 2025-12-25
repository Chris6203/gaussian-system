"""
Feature Engineering Module

Provides modular feature families for the SPY options trading bot:
- equity_etf: Cross-asset equity/ETF features
- options_surface: Volatility & options surface features
- breadth: Market breadth approximations
- macro: Rates/macro proxy features
- crypto: Crypto risk-on/off features
- meta: Time and regime features
- pipeline: Unified feature pipeline
"""

from .equity_etf import compute_equity_etf_features
from .options_surface import compute_options_surface_features
from .breadth import compute_breadth_features
from .macro import compute_macro_features
from .crypto import compute_crypto_features
from .meta import compute_meta_features
from .pipeline import FeaturePipeline, FeatureConfig

__all__ = [
    'compute_equity_etf_features',
    'compute_options_surface_features',
    'compute_breadth_features',
    'compute_macro_features',
    'compute_crypto_features',
    'compute_meta_features',
    'FeaturePipeline',
    'FeatureConfig',
]




