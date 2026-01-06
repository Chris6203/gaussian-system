"""
Quantor-MTFuzz Integration for Gaussian System
===============================================

Adapted features from the spy-iron-condor-trading system:
- Fuzzy position sizing with 9-factor membership functions
- Market regime classification (5 regimes)
- Realized volatility and IV skew analytics
- Liquidity gate filtering

Usage:
    from integrations.quantor import QuantorFacade

    quantor = QuantorFacade()

    # Get position size recommendation
    size = quantor.compute_position_size(
        equity=10000,
        max_loss=500,
        market_data=current_data
    )

    # Check if trading is allowed
    if quantor.should_trade(market_data):
        # Execute trade
        pass
"""

from .facade import QuantorFacade, QuantorResult
from .fuzzy_sizer import FuzzyPositionSizer, FuzzyMemberships
from .regime_filter import RegimeFilter, MarketRegime
from .volatility import VolatilityAnalyzer, VolatilityMetrics
from .data_alignment import (
    ChainAlignment,
    AlignmentMode,
    AlignmentPolicy,
    AlignmentConfig,
    DataAligner,
    AlignmentStats,
    AlignmentDiagnosticsTracker,
    AlignedStepState,
    create_alignment_wrapper
)

__all__ = [
    # Facade
    'QuantorFacade',
    'QuantorResult',
    # Fuzzy sizing
    'FuzzyPositionSizer',
    'FuzzyMemberships',
    # Regime filter
    'RegimeFilter',
    'MarketRegime',
    # Volatility
    'VolatilityAnalyzer',
    'VolatilityMetrics',
    # Data alignment
    'ChainAlignment',
    'AlignmentMode',
    'AlignmentPolicy',
    'AlignmentConfig',
    'DataAligner',
    'AlignmentStats',
    'AlignmentDiagnosticsTracker',
    'AlignedStepState',
    'create_alignment_wrapper'
]

__version__ = '1.0.0'
__source__ = 'Adapted from trextrader/spy-iron-condor-trading'
