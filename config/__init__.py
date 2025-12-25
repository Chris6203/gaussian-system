"""
Configuration Package
=====================

Central configuration for the trading system.
"""

from .symbols import (
    CORE_SYMBOLS,
    MEGA_CAP_TECH_SYMBOLS,
    SECTOR_ETF_SYMBOLS,
    MACRO_SYMBOLS,
    CRYPTO_SYMBOLS,
    ALL_DATA_COLLECTION_SYMBOLS,
    get_all_symbols,
    get_equity_symbols,
    get_tech_symbols,
    is_valid_symbol,
    SYMBOL_DESCRIPTIONS,
)

__all__ = [
    'CORE_SYMBOLS',
    'MEGA_CAP_TECH_SYMBOLS',
    'SECTOR_ETF_SYMBOLS',
    'MACRO_SYMBOLS',
    'CRYPTO_SYMBOLS',
    'ALL_DATA_COLLECTION_SYMBOLS',
    'get_all_symbols',
    'get_equity_symbols',
    'get_tech_symbols',
    'is_valid_symbol',
    'SYMBOL_DESCRIPTIONS',
]




