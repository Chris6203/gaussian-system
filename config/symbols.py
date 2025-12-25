"""
Central Symbol Configuration
============================

Defines all symbol lists used by the data collection and trading system.
Update this file to add/remove symbols from data collection.

NOTE: The DataCollector reads from config.json, which should reference or 
incorporate these lists. This module provides the canonical definitions.
"""

from typing import List

# =============================================================================
# CORE SYMBOLS - Market indices and primary trading instruments
# =============================================================================

CORE_SYMBOLS: List[str] = [
    "SPY",     # S&P 500 ETF - Primary trading symbol
    "^VIX",    # CBOE Volatility Index
]

# =============================================================================
# MEGA-CAP TECH SYMBOLS - Top tech/growth names influencing SPY
# =============================================================================

MEGA_CAP_TECH_SYMBOLS: List[str] = [
    "NVDA",    # NVIDIA - AI/GPU leader
    "AAPL",    # Apple - Consumer tech giant
    "MSFT",    # Microsoft - Enterprise/cloud
    "AMZN",    # Amazon - E-commerce/cloud
    "GOOGL",   # Alphabet - Search/advertising
    "META",    # Meta Platforms - Social media
    "TSLA",    # Tesla - EV/energy
    "PLTR",    # Palantir - Data analytics/AI
    "INTC",    # Intel - Semiconductors
    "AMD",     # Advanced Micro Devices - CPUs/GPUs
]

# =============================================================================
# SECTOR ETF SYMBOLS - Sector rotation and breadth analysis
# =============================================================================

SECTOR_ETF_SYMBOLS: List[str] = [
    "XLF",     # Financials
    "XLK",     # Technology
    "XLE",     # Energy
    "XLU",     # Utilities
    "XLY",     # Consumer Discretionary
    "XLP",     # Consumer Staples
    "HYG",     # High Yield Corporate Bonds
    "LQD",     # Investment Grade Corporate Bonds
]

# =============================================================================
# MACRO/RATES SYMBOLS - Interest rates and dollar strength
# =============================================================================

MACRO_SYMBOLS: List[str] = [
    "TLT",     # 20+ Year Treasury ETF
    "IEF",     # 7-10 Year Treasury ETF
    "SHY",     # 1-3 Year Treasury ETF
    "UUP",     # US Dollar Index ETF
]

# =============================================================================
# CRYPTO SYMBOLS - Risk sentiment indicators
# =============================================================================

CRYPTO_SYMBOLS: List[str] = [
    "BTC-USD", # Bitcoin
    "ETH-USD", # Ethereum
]

# =============================================================================
# AGGREGATE SYMBOL LISTS
# =============================================================================

# All symbols for data collection (combines all categories)
ALL_DATA_COLLECTION_SYMBOLS: List[str] = (
    CORE_SYMBOLS + 
    MEGA_CAP_TECH_SYMBOLS + 
    SECTOR_ETF_SYMBOLS + 
    MACRO_SYMBOLS + 
    CRYPTO_SYMBOLS
)

# Remove duplicates while preserving order
def _dedupe(symbols: List[str]) -> List[str]:
    """Remove duplicates while preserving order."""
    seen = set()
    result = []
    for s in symbols:
        if s not in seen:
            seen.add(s)
            result.append(s)
    return result

ALL_DATA_COLLECTION_SYMBOLS = _dedupe(ALL_DATA_COLLECTION_SYMBOLS)


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def get_all_symbols() -> List[str]:
    """Get all symbols for data collection."""
    return ALL_DATA_COLLECTION_SYMBOLS.copy()


def get_equity_symbols() -> List[str]:
    """Get equity symbols only (no VIX, no crypto)."""
    return [s for s in ALL_DATA_COLLECTION_SYMBOLS 
            if not s.startswith('^') and '-USD' not in s]


def get_tech_symbols() -> List[str]:
    """Get mega-cap tech symbols."""
    return MEGA_CAP_TECH_SYMBOLS.copy()


def is_valid_symbol(symbol: str) -> bool:
    """Check if a symbol is in our tracked list."""
    return symbol in ALL_DATA_COLLECTION_SYMBOLS


# =============================================================================
# SYMBOL METADATA (for documentation/UI)
# =============================================================================

SYMBOL_DESCRIPTIONS = {
    # Core
    "SPY": "S&P 500 ETF",
    "QQQ": "Nasdaq 100 ETF",
    "IWM": "Russell 2000 ETF",
    "DIA": "Dow Jones ETF",
    "^VIX": "Volatility Index",
    
    # Mega-cap tech
    "NVDA": "NVIDIA Corporation",
    "AAPL": "Apple Inc.",
    "MSFT": "Microsoft Corporation",
    "AMZN": "Amazon.com Inc.",
    "GOOGL": "Alphabet Inc.",
    "META": "Meta Platforms Inc.",
    "TSLA": "Tesla Inc.",
    "PLTR": "Palantir Technologies",
    "INTC": "Intel Corporation",
    "AMD": "Advanced Micro Devices",
    
    # Sectors
    "XLF": "Financial Select Sector",
    "XLK": "Technology Select Sector",
    "XLE": "Energy Select Sector",
    "XLU": "Utilities Select Sector",
    "XLY": "Consumer Discretionary",
    "XLP": "Consumer Staples",
    "HYG": "High Yield Corporate Bonds",
    "LQD": "Investment Grade Bonds",
    
    # Macro
    "TLT": "20+ Year Treasury ETF",
    "IEF": "7-10 Year Treasury ETF",
    "SHY": "1-3 Year Treasury ETF",
    "UUP": "US Dollar Index ETF",
    
    # Crypto
    "BTC-USD": "Bitcoin",
    "ETH-USD": "Ethereum",
}


if __name__ == "__main__":
    # Print summary when run directly
    print("=" * 60)
    print("SYMBOL CONFIGURATION")
    print("=" * 60)
    print(f"\nCore Symbols ({len(CORE_SYMBOLS)}): {', '.join(CORE_SYMBOLS)}")
    print(f"\nMega-Cap Tech ({len(MEGA_CAP_TECH_SYMBOLS)}): {', '.join(MEGA_CAP_TECH_SYMBOLS)}")
    print(f"\nSector ETFs ({len(SECTOR_ETF_SYMBOLS)}): {', '.join(SECTOR_ETF_SYMBOLS)}")
    print(f"\nMacro ({len(MACRO_SYMBOLS)}): {', '.join(MACRO_SYMBOLS)}")
    print(f"\nCrypto ({len(CRYPTO_SYMBOLS)}): {', '.join(CRYPTO_SYMBOLS)}")
    print(f"\n{'=' * 60}")
    print(f"TOTAL UNIQUE SYMBOLS: {len(ALL_DATA_COLLECTION_SYMBOLS)}")
    print("=" * 60)
