#!/usr/bin/env python3
"""
Rates / Macro Proxy Features

Computes macro proxy features using ETFs:
- Treasury/yield proxies (TLT, IEF, SHY)
- Dollar index proxy (UUP)
- Yield curve slope approximation
- Index proxies (IWM, SMH, RSP)
- Sector ETFs (XLK, XLF, XLE, etc.)
- Credit proxies (HYG, LQD)
- Commodity proxies (USO, GLD)
- Mega-cap movers (AAPL, MSFT, NVDA, etc.)
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)


# =============================================================================
# SYMBOL DEFINITIONS
# =============================================================================

# Original macro symbols (treasury + dollar)
MACRO_SYMBOLS = {
    'TLT': 'iShares 20+ Year Treasury Bond ETF (long duration)',
    'IEF': 'iShares 7-10 Year Treasury Bond ETF (intermediate)',
    'SHY': 'iShares 1-3 Year Treasury Bond ETF (short duration)',
    'UUP': 'Invesco DB US Dollar Index Bullish Fund',
}

# Index/Leadership proxies
INDEX_PROXY_SYMBOLS = {
    'IWM': 'iShares Russell 2000 ETF (small caps)',
    'SMH': 'VanEck Semiconductor ETF (tech leadership)',
    'RSP': 'Invesco S&P 500 Equal Weight ETF (breadth proxy)',
}

# Sector ETFs (SPDR Select Sector)
SECTOR_ETF_SYMBOLS = {
    'XLK': 'Technology Select Sector SPDR',
    'XLF': 'Financial Select Sector SPDR',
    'XLE': 'Energy Select Sector SPDR',
    'XLY': 'Consumer Discretionary Select Sector SPDR',
    'XLP': 'Consumer Staples Select Sector SPDR',
    'XLV': 'Health Care Select Sector SPDR',
    'XLI': 'Industrial Select Sector SPDR',
    'XLU': 'Utilities Select Sector SPDR',
    'XLB': 'Materials Select Sector SPDR',
    'XLRE': 'Real Estate Select Sector SPDR',
}

# Credit proxies
CREDIT_SYMBOLS = {
    'HYG': 'iShares iBoxx High Yield Corporate Bond ETF',
    'LQD': 'iShares iBoxx Investment Grade Corporate Bond ETF',
}

# Commodity proxies
COMMODITY_SYMBOLS = {
    'USO': 'United States Oil Fund (crude oil)',
    'GLD': 'SPDR Gold Shares (gold)',
}

# Mega-cap movers
MEGACAP_SYMBOLS = {
    'AAPL': 'Apple Inc.',
    'MSFT': 'Microsoft Corporation',
    'NVDA': 'NVIDIA Corporation',
    'AMZN': 'Amazon.com Inc.',
    'GOOGL': 'Alphabet Inc. Class A',
    'META': 'Meta Platforms Inc.',
}

# Combined extended symbols (all new symbols)
EXTENDED_MACRO_SYMBOLS = {
    **INDEX_PROXY_SYMBOLS,
    **SECTOR_ETF_SYMBOLS,
    **CREDIT_SYMBOLS,
    **COMMODITY_SYMBOLS,
    **MEGACAP_SYMBOLS,
}

# All symbols needed for macro features
ALL_MACRO_SYMBOLS = {
    **MACRO_SYMBOLS,
    **EXTENDED_MACRO_SYMBOLS,
}


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def _get_close_series(data: Dict[str, pd.DataFrame], symbol: str) -> Optional[np.ndarray]:
    """Get close price series for a symbol."""
    if symbol not in data or data[symbol] is None or data[symbol].empty:
        return None

    df = data[symbol]
    close_col = 'close' if 'close' in df.columns else 'Close'

    if close_col not in df.columns:
        return None

    return df[close_col].values


def _get_return(data: Dict[str, pd.DataFrame], symbol: str, horizon: int) -> Optional[float]:
    """Get return for a symbol over a horizon."""
    close = _get_close_series(data, symbol)

    if close is None or len(close) <= horizon:
        return None

    return (close[-1] - close[-horizon - 1]) / (close[-horizon - 1] + 1e-10)


def _get_volatility(data: Dict[str, pd.DataFrame], symbol: str, window: int) -> Optional[float]:
    """Get realized volatility for a symbol."""
    close = _get_close_series(data, symbol)

    if close is None or len(close) < window + 1:
        return None

    returns = np.diff(close) / close[:-1]
    return float(np.std(returns[-window:]))


def _get_return_series(data: Dict[str, pd.DataFrame], symbol: str, length: int) -> Optional[np.ndarray]:
    """Get return series for a symbol."""
    close = _get_close_series(data, symbol)

    if close is None or len(close) < length + 1:
        return None

    returns = np.diff(close[-length - 1:]) / close[-length - 1:-1]
    return returns


# =============================================================================
# ORIGINAL MACRO FEATURES (Treasury + Dollar)
# =============================================================================

def compute_macro_features(
    data: Dict[str, pd.DataFrame],
    return_horizons: List[int] = [1, 5, 20],
    vol_windows: List[int] = [20, 60]
) -> Dict[str, float]:
    """
    Compute rates/macro proxy features from ETF data.

    Args:
        data: Dict mapping symbol -> OHLCV DataFrame
              Expected symbols: TLT, IEF, SHY, UUP
        return_horizons: Horizons for return calculation
        vol_windows: Windows for volatility calculation

    Returns:
        Dict of macro features (treasury + dollar + cross-asset)
    """
    features = {
        # Treasury
        'tlt_ret_1': 0.0,
        'tlt_ret_5': 0.0,
        'tlt_ret_20': 0.0,
        'ief_ret_5': 0.0,
        'shy_ret_5': 0.0,
        'yield_curve_slope': 0.0,
        'duration_momentum': 0.0,
        'tlt_vol_20': 0.0,

        # Dollar
        'uup_ret_1': 0.0,
        'uup_ret_5': 0.0,
        'dollar_strength': 0.0,
        'dollar_vol_20': 0.0,

        # Cross-asset
        'bond_equity_corr': 0.0,
        'dollar_equity_corr': 0.0,
        'flight_to_safety': 0.0,
    }

    try:
        # =====================================================================
        # TREASURY PROXIES (TLT, IEF, SHY)
        # =====================================================================

        # TLT (20+ year treasury)
        for h in return_horizons:
            ret = _get_return(data, 'TLT', h)
            if ret is not None:
                features[f'tlt_ret_{h}'] = ret

        tlt_vol = _get_volatility(data, 'TLT', 20)
        if tlt_vol is not None:
            features['tlt_vol_20'] = tlt_vol

        # IEF (7-10 year treasury)
        ief_ret = _get_return(data, 'IEF', 5)
        if ief_ret is not None:
            features['ief_ret_5'] = ief_ret

        # SHY (1-3 year treasury)
        shy_ret = _get_return(data, 'SHY', 5)
        if shy_ret is not None:
            features['shy_ret_5'] = shy_ret

        # Yield curve slope proxy: TLT - SHY performance
        tlt_ret_5 = _get_return(data, 'TLT', 5)
        shy_ret_5 = _get_return(data, 'SHY', 5)

        if tlt_ret_5 is not None and shy_ret_5 is not None:
            features['yield_curve_slope'] = (tlt_ret_5 - shy_ret_5) * 100

        # Duration momentum
        if tlt_ret_5 is not None and shy_ret_5 is not None:
            features['duration_momentum'] = 1.0 if tlt_ret_5 > shy_ret_5 else -1.0

        # =====================================================================
        # DOLLAR STRENGTH (UUP)
        # =====================================================================

        for h in return_horizons:
            ret = _get_return(data, 'UUP', h)
            if ret is not None:
                features[f'uup_ret_{h}' if h != 1 else 'uup_ret_1'] = ret

        uup_vol = _get_volatility(data, 'UUP', 20)
        if uup_vol is not None:
            features['dollar_vol_20'] = uup_vol

        # Dollar strength indicator
        uup_ret_5 = _get_return(data, 'UUP', 5)
        if uup_ret_5 is not None:
            features['dollar_strength'] = float(np.clip(uup_ret_5 * 50, -1, 1))

        # =====================================================================
        # CROSS-ASSET CORRELATIONS
        # =====================================================================

        corr_window = 20

        # Bond-equity correlation
        tlt_returns = _get_return_series(data, 'TLT', corr_window)
        spy_returns = _get_return_series(data, 'SPY', corr_window)

        if tlt_returns is not None and spy_returns is not None:
            if len(tlt_returns) == len(spy_returns) and len(tlt_returns) >= 10:
                corr = np.corrcoef(tlt_returns, spy_returns)[0, 1]
                if not np.isnan(corr):
                    features['bond_equity_corr'] = float(corr)

        # Dollar-equity correlation
        uup_returns = _get_return_series(data, 'UUP', corr_window)

        if uup_returns is not None and spy_returns is not None:
            if len(uup_returns) == len(spy_returns) and len(uup_returns) >= 10:
                corr = np.corrcoef(uup_returns, spy_returns)[0, 1]
                if not np.isnan(corr):
                    features['dollar_equity_corr'] = float(corr)

        # Flight to safety
        spy_ret_5 = _get_return(data, 'SPY', 5)
        tlt_ret_5 = _get_return(data, 'TLT', 5)

        if spy_ret_5 is not None and tlt_ret_5 is not None:
            if spy_ret_5 < 0:
                features['flight_to_safety'] = max(0, tlt_ret_5) * 10
            else:
                features['flight_to_safety'] = 0.0

    except Exception as e:
        logger.warning(f"Error computing macro features: {e}")

    return features


# =============================================================================
# NEW EXTENDED MACRO FEATURES
# =============================================================================

def compute_relative_strength_ratios(
    data: Dict[str, pd.DataFrame],
    base_symbol: str = 'SPY',
    horizons: List[int] = [1, 5, 20]
) -> Dict[str, float]:
    """
    Compute relative strength ratios (X/SPY) for all extended symbols.

    Features:
    - {symbol}_rel_{h}: Relative return vs SPY at horizon h
    - {symbol}_rs_momentum: Is symbol outperforming SPY?
    """
    features = {}

    # Initialize all features to 0
    symbols_to_compare = list(INDEX_PROXY_SYMBOLS.keys()) + list(SECTOR_ETF_SYMBOLS.keys())
    for symbol in symbols_to_compare:
        for h in horizons:
            features[f'{symbol.lower()}_rel_{h}'] = 0.0
        features[f'{symbol.lower()}_rs_momentum'] = 0.0

    try:
        base_returns = {}
        for h in horizons:
            base_returns[h] = _get_return(data, base_symbol, h)

        for symbol in symbols_to_compare:
            for h in horizons:
                sym_ret = _get_return(data, symbol, h)
                base_ret = base_returns.get(h)

                if sym_ret is not None and base_ret is not None:
                    # Relative strength: symbol return - SPY return
                    features[f'{symbol.lower()}_rel_{h}'] = sym_ret - base_ret

            # RS momentum: is 5-bar RS positive?
            rs_5 = features.get(f'{symbol.lower()}_rel_5', 0.0)
            features[f'{symbol.lower()}_rs_momentum'] = 1.0 if rs_5 > 0 else -1.0

    except Exception as e:
        logger.warning(f"Error computing relative strength: {e}")

    return features


def compute_credit_spread_features(
    data: Dict[str, pd.DataFrame],
    horizons: List[int] = [1, 5, 20]
) -> Dict[str, float]:
    """
    Compute credit spread proxy features.

    Features:
    - hyg_ret_{h}: High yield bond return
    - lqd_ret_{h}: Investment grade bond return
    - credit_spread_{h}: HYG - LQD differential (risk appetite)
    - credit_risk_on: Is credit spread widening (risk-on)?
    - hyg_spy_corr: Correlation between HYG and SPY
    """
    features = {
        'hyg_ret_1': 0.0,
        'hyg_ret_5': 0.0,
        'hyg_ret_20': 0.0,
        'lqd_ret_1': 0.0,
        'lqd_ret_5': 0.0,
        'lqd_ret_20': 0.0,
        'credit_spread_1': 0.0,
        'credit_spread_5': 0.0,
        'credit_spread_20': 0.0,
        'credit_risk_on': 0.0,
        'hyg_spy_corr': 0.0,
    }

    try:
        for h in horizons:
            hyg_ret = _get_return(data, 'HYG', h)
            lqd_ret = _get_return(data, 'LQD', h)

            if hyg_ret is not None:
                features[f'hyg_ret_{h}'] = hyg_ret
            if lqd_ret is not None:
                features[f'lqd_ret_{h}'] = lqd_ret

            # Credit spread: HYG outperforming LQD = risk-on
            if hyg_ret is not None and lqd_ret is not None:
                features[f'credit_spread_{h}'] = (hyg_ret - lqd_ret) * 100

        # Risk-on indicator based on 5-bar spread
        features['credit_risk_on'] = 1.0 if features['credit_spread_5'] > 0 else -1.0

        # HYG-SPY correlation (20-bar)
        hyg_returns = _get_return_series(data, 'HYG', 20)
        spy_returns = _get_return_series(data, 'SPY', 20)

        if hyg_returns is not None and spy_returns is not None:
            if len(hyg_returns) == len(spy_returns) and len(hyg_returns) >= 10:
                corr = np.corrcoef(hyg_returns, spy_returns)[0, 1]
                if not np.isnan(corr):
                    features['hyg_spy_corr'] = float(corr)

    except Exception as e:
        logger.warning(f"Error computing credit features: {e}")

    return features


def compute_sector_rotation_features(
    data: Dict[str, pd.DataFrame],
    horizon: int = 5
) -> Dict[str, float]:
    """
    Compute sector rotation features.

    Features:
    - sector_dispersion: Std dev of sector returns (high = rotating)
    - risk_on_off_ratio: XLY/XLP ratio (consumer discretionary vs staples)
    - tech_leadership: XLK relative strength
    - defensive_flow: Average of XLU, XLP, XLV (defensive sectors)
    - cyclical_flow: Average of XLY, XLI, XLB (cyclical sectors)
    - sector_leader: Which sector is leading (encoded)
    - sector_laggard: Which sector is lagging (encoded)
    """
    features = {
        'sector_dispersion': 0.0,
        'risk_on_off_ratio': 0.0,
        'tech_leadership': 0.0,
        'defensive_flow': 0.0,
        'cyclical_flow': 0.0,
        'sector_leader_code': 0.0,
        'sector_laggard_code': 0.0,
    }

    sectors = list(SECTOR_ETF_SYMBOLS.keys())
    sector_codes = {s: i / len(sectors) for i, s in enumerate(sectors)}

    try:
        sector_returns = {}
        for sector in sectors:
            ret = _get_return(data, sector, horizon)
            if ret is not None:
                sector_returns[sector] = ret

        if len(sector_returns) >= 5:
            returns_array = np.array(list(sector_returns.values()))

            # Sector dispersion
            features['sector_dispersion'] = float(np.std(returns_array))

            # Leader and laggard
            leader = max(sector_returns, key=sector_returns.get)
            laggard = min(sector_returns, key=sector_returns.get)
            features['sector_leader_code'] = sector_codes[leader]
            features['sector_laggard_code'] = sector_codes[laggard]

        # Risk-on/off ratio: XLY vs XLP
        xly_ret = _get_return(data, 'XLY', horizon)
        xlp_ret = _get_return(data, 'XLP', horizon)

        if xly_ret is not None and xlp_ret is not None:
            features['risk_on_off_ratio'] = (xly_ret - xlp_ret) * 100

        # Tech leadership
        xlk_ret = _get_return(data, 'XLK', horizon)
        spy_ret = _get_return(data, 'SPY', horizon)

        if xlk_ret is not None and spy_ret is not None:
            features['tech_leadership'] = (xlk_ret - spy_ret) * 100

        # Defensive flow (XLU, XLP, XLV)
        defensive_rets = []
        for sym in ['XLU', 'XLP', 'XLV']:
            ret = _get_return(data, sym, horizon)
            if ret is not None:
                defensive_rets.append(ret)
        if defensive_rets:
            features['defensive_flow'] = float(np.mean(defensive_rets))

        # Cyclical flow (XLY, XLI, XLB)
        cyclical_rets = []
        for sym in ['XLY', 'XLI', 'XLB']:
            ret = _get_return(data, sym, horizon)
            if ret is not None:
                cyclical_rets.append(ret)
        if cyclical_rets:
            features['cyclical_flow'] = float(np.mean(cyclical_rets))

    except Exception as e:
        logger.warning(f"Error computing sector features: {e}")

    return features


def compute_commodity_features(
    data: Dict[str, pd.DataFrame],
    horizons: List[int] = [1, 5, 20]
) -> Dict[str, float]:
    """
    Compute commodity proxy features.

    Features:
    - uso_ret_{h}: Oil return
    - gld_ret_{h}: Gold return
    - oil_gold_spread: USO - GLD (risk appetite proxy)
    - oil_momentum: Oil trend indicator
    - gold_flight: Gold strength indicator (safe haven)
    """
    features = {
        'uso_ret_1': 0.0,
        'uso_ret_5': 0.0,
        'uso_ret_20': 0.0,
        'gld_ret_1': 0.0,
        'gld_ret_5': 0.0,
        'gld_ret_20': 0.0,
        'oil_gold_spread': 0.0,
        'oil_momentum': 0.0,
        'gold_flight': 0.0,
    }

    try:
        for h in horizons:
            uso_ret = _get_return(data, 'USO', h)
            gld_ret = _get_return(data, 'GLD', h)

            if uso_ret is not None:
                features[f'uso_ret_{h}'] = uso_ret
            if gld_ret is not None:
                features[f'gld_ret_{h}'] = gld_ret

        # Oil-gold spread (5-bar)
        uso_5 = features['uso_ret_5']
        gld_5 = features['gld_ret_5']
        features['oil_gold_spread'] = (uso_5 - gld_5) * 100

        # Oil momentum
        features['oil_momentum'] = 1.0 if uso_5 > 0 else -1.0

        # Gold flight (when SPY is down and gold is up)
        spy_ret = _get_return(data, 'SPY', 5)
        if spy_ret is not None and spy_ret < 0:
            features['gold_flight'] = max(0, gld_5) * 10

    except Exception as e:
        logger.warning(f"Error computing commodity features: {e}")

    return features


def compute_megacap_features(
    data: Dict[str, pd.DataFrame],
    horizons: List[int] = [1, 5, 20]
) -> Dict[str, float]:
    """
    Compute mega-cap mover features.

    Features:
    - megacap_avg_ret_{h}: Average mega-cap return
    - megacap_dispersion: Std dev of mega-cap returns
    - nvda_leadership: NVDA relative strength (AI proxy)
    - megacap_breadth: % of mega-caps positive
    - tech_faang_momentum: Combined tech momentum
    """
    features = {
        'megacap_avg_ret_1': 0.0,
        'megacap_avg_ret_5': 0.0,
        'megacap_avg_ret_20': 0.0,
        'megacap_dispersion': 0.0,
        'nvda_leadership': 0.0,
        'megacap_breadth': 0.0,
        'tech_faang_momentum': 0.0,
    }

    megacaps = list(MEGACAP_SYMBOLS.keys())

    try:
        for h in horizons:
            rets = []
            for sym in megacaps:
                ret = _get_return(data, sym, h)
                if ret is not None:
                    rets.append(ret)

            if rets:
                features[f'megacap_avg_ret_{h}'] = float(np.mean(rets))

                if h == 5:
                    # Dispersion at 5-bar
                    features['megacap_dispersion'] = float(np.std(rets))
                    # Breadth at 5-bar
                    features['megacap_breadth'] = sum(1 for r in rets if r > 0) / len(rets)

        # NVDA leadership (vs mega-cap average)
        nvda_ret = _get_return(data, 'NVDA', 5)
        avg_ret = features['megacap_avg_ret_5']

        if nvda_ret is not None:
            features['nvda_leadership'] = (nvda_ret - avg_ret) * 100

        # Tech FAANG momentum
        features['tech_faang_momentum'] = 1.0 if features['megacap_avg_ret_5'] > 0 else -1.0

    except Exception as e:
        logger.warning(f"Error computing megacap features: {e}")

    return features


def compute_index_proxy_features(
    data: Dict[str, pd.DataFrame],
    horizons: List[int] = [1, 5, 20]
) -> Dict[str, float]:
    """
    Compute index proxy features.

    Features:
    - iwm_ret_{h}: Small cap return
    - smh_ret_{h}: Semiconductor return
    - rsp_ret_{h}: Equal-weight S&P return
    - small_cap_leadership: IWM vs SPY
    - semi_leadership: SMH vs SPY (tech risk appetite)
    - breadth_proxy: RSP vs SPY (market breadth)
    - risk_appetite: Combined small cap + semi leadership
    """
    features = {
        'iwm_ret_1': 0.0,
        'iwm_ret_5': 0.0,
        'iwm_ret_20': 0.0,
        'smh_ret_1': 0.0,
        'smh_ret_5': 0.0,
        'smh_ret_20': 0.0,
        'rsp_ret_1': 0.0,
        'rsp_ret_5': 0.0,
        'rsp_ret_20': 0.0,
        'small_cap_leadership': 0.0,
        'semi_leadership': 0.0,
        'breadth_proxy': 0.0,
        'risk_appetite': 0.0,
    }

    try:
        for h in horizons:
            for sym in ['IWM', 'SMH', 'RSP']:
                ret = _get_return(data, sym, h)
                if ret is not None:
                    features[f'{sym.lower()}_ret_{h}'] = ret

        spy_ret_5 = _get_return(data, 'SPY', 5)

        # Small cap leadership
        iwm_ret_5 = features['iwm_ret_5']
        if spy_ret_5 is not None:
            features['small_cap_leadership'] = (iwm_ret_5 - spy_ret_5) * 100

        # Semi leadership
        smh_ret_5 = features['smh_ret_5']
        if spy_ret_5 is not None:
            features['semi_leadership'] = (smh_ret_5 - spy_ret_5) * 100

        # Breadth proxy (RSP vs SPY)
        rsp_ret_5 = features['rsp_ret_5']
        if spy_ret_5 is not None:
            features['breadth_proxy'] = (rsp_ret_5 - spy_ret_5) * 100

        # Combined risk appetite
        features['risk_appetite'] = (
            features['small_cap_leadership'] + features['semi_leadership']
        ) / 2

    except Exception as e:
        logger.warning(f"Error computing index proxy features: {e}")

    return features


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

def compute_extended_macro_features(
    data: Dict[str, pd.DataFrame],
    include_original: bool = True,
    include_relative_strength: bool = True,
    include_credit: bool = True,
    include_sectors: bool = True,
    include_commodities: bool = True,
    include_megacaps: bool = True,
    include_index_proxies: bool = True,
) -> Dict[str, float]:
    """
    Compute all macro proxy features.

    This is the main entry point for extended macro features.

    Args:
        data: Dict mapping symbol -> OHLCV DataFrame
        include_*: Flags to enable/disable feature groups

    Returns:
        Dict of all macro features (~100-150 features)
    """
    features = {}

    # Original treasury + dollar features
    if include_original:
        features.update(compute_macro_features(data))

    # Index proxy features
    if include_index_proxies:
        features.update(compute_index_proxy_features(data))

    # Sector rotation features
    if include_sectors:
        features.update(compute_sector_rotation_features(data))

    # Relative strength ratios
    if include_relative_strength:
        features.update(compute_relative_strength_ratios(data))

    # Credit spread features
    if include_credit:
        features.update(compute_credit_spread_features(data))

    # Commodity features
    if include_commodities:
        features.update(compute_commodity_features(data))

    # Mega-cap features
    if include_megacaps:
        features.update(compute_megacap_features(data))

    return features


# =============================================================================
# TEST CODE
# =============================================================================

if __name__ == "__main__":
    """Test macro features with mock data."""
    import logging

    logging.basicConfig(level=logging.INFO)

    print("=" * 70)
    print("EXTENDED MACRO FEATURE TEST")
    print("=" * 70)

    # Create mock data
    np.random.seed(42)
    n_bars = 100

    data = {}

    # Mock SPY
    spy_trend = np.cumsum(np.random.randn(n_bars) * 0.002)
    spy_prices = 450 * np.exp(spy_trend)
    data['SPY'] = pd.DataFrame({
        'close': spy_prices,
        'volume': np.random.randint(1000000, 10000000, n_bars),
    })

    # Mock all symbols
    all_symbols = list(ALL_MACRO_SYMBOLS.keys())
    base_prices = {
        'TLT': 100, 'IEF': 100, 'SHY': 80, 'UUP': 27,
        'IWM': 200, 'SMH': 250, 'RSP': 160,
        'XLK': 200, 'XLF': 40, 'XLE': 90, 'XLY': 180,
        'XLP': 75, 'XLV': 140, 'XLI': 115, 'XLU': 70,
        'XLB': 85, 'XLRE': 45,
        'HYG': 80, 'LQD': 115,
        'USO': 75, 'GLD': 180,
        'AAPL': 190, 'MSFT': 400, 'NVDA': 500,
        'AMZN': 180, 'GOOGL': 175, 'META': 350,
    }

    for symbol in all_symbols:
        base = base_prices.get(symbol, 100)
        # Correlated with SPY + some noise
        corr = 0.5 + np.random.rand() * 0.3
        vol = 0.002 + np.random.rand() * 0.003
        trend = np.cumsum(np.random.randn(n_bars) * vol + spy_trend * corr * 0.5)
        prices = base * np.exp(trend)
        data[symbol] = pd.DataFrame({
            'close': prices,
            'volume': np.random.randint(100000, 5000000, n_bars),
        })

    # Compute all features
    features = compute_extended_macro_features(data)

    print(f"\nSymbols available: {len(data)}")
    print(f"Total features: {len(features)}")

    # Group features by category
    categories = {
        'Treasury': ['tlt_', 'ief_', 'shy_', 'yield_', 'duration_'],
        'Dollar': ['uup_', 'dollar_'],
        'Cross-Asset': ['bond_equity', 'flight_'],
        'Index Proxy': ['iwm_', 'smh_', 'rsp_', 'small_cap', 'semi_', 'breadth_', 'risk_appetite'],
        'Sector': ['sector_', 'risk_on_off', 'tech_leadership', 'defensive_', 'cyclical_'],
        'Relative Strength': ['_rel_', '_rs_momentum'],
        'Credit': ['hyg_', 'lqd_', 'credit_'],
        'Commodities': ['uso_', 'gld_', 'oil_', 'gold_'],
        'Mega-caps': ['megacap_', 'nvda_', 'tech_faang'],
    }

    for cat, prefixes in categories.items():
        cat_features = {k: v for k, v in features.items()
                       if any(p in k for p in prefixes)}
        if cat_features:
            print(f"\n{cat} ({len(cat_features)} features):")
            for k, v in sorted(cat_features.items())[:5]:
                print(f"  {k}: {v:.4f}")
            if len(cat_features) > 5:
                print(f"  ... and {len(cat_features) - 5} more")

    print("\n" + "=" * 70)
    print("TEST COMPLETE")
    print("=" * 70)
