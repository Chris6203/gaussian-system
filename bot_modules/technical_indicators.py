"""
Technical Indicators for Consensus Entry Controller
Computes MACD, Bollinger Bands, Multi-Timeframe HMA, RSI, Market Breadth, and Options Flow
"""
import numpy as np
import pandas as pd
from typing import Dict, Optional, Any


def compute_hma(data: np.ndarray, period: int) -> float:
    """
    Compute Hull Moving Average for a given period.
    HMA = WMA(2*WMA(n/2) - WMA(n), sqrt(n))
    """
    if len(data) < period + 1:
        return data[-1] if len(data) > 0 else 0.0

    def wma(arr, p):
        weights = np.arange(1, p + 1)
        return np.sum(arr[-p:] * weights) / np.sum(weights)

    half_period = max(1, period // 2)
    wma_half = wma(data, half_period)
    wma_full = wma(data, period)
    return float(2 * wma_half - wma_full)


def compute_multi_timeframe_hma(close_prices: np.ndarray) -> Dict[str, float]:
    """
    Compute HMAs for multiple timeframes (10, 20, 50) and return alignment score.

    Returns:
        Dict with hma_10, hma_20, hma_50, hma_trend (-1/0/1), hma_strength (0-1)
    """
    result = {'hma_10': 0.0, 'hma_20': 0.0, 'hma_50': 0.0, 'hma_trend': 0.0, 'hma_strength': 0.0}

    if close_prices is None or len(close_prices) < 51:
        return result

    # Compute HMAs
    hma_10 = compute_hma(close_prices, 10)
    hma_20 = compute_hma(close_prices, 20)
    hma_50 = compute_hma(close_prices, 50)

    # Compute previous for trend
    hma_10_prev = compute_hma(close_prices[:-1], 10) if len(close_prices) > 11 else hma_10
    hma_20_prev = compute_hma(close_prices[:-1], 20) if len(close_prices) > 21 else hma_20
    hma_50_prev = compute_hma(close_prices[:-1], 50) if len(close_prices) > 51 else hma_50

    # Trend direction
    trend_10 = 1 if hma_10 > hma_10_prev else (-1 if hma_10 < hma_10_prev else 0)
    trend_20 = 1 if hma_20 > hma_20_prev else (-1 if hma_20 < hma_20_prev else 0)
    trend_50 = 1 if hma_50 > hma_50_prev else (-1 if hma_50 < hma_50_prev else 0)

    # Alignment
    total = trend_10 + trend_20 + trend_50
    if total >= 2:
        result['hma_trend'] = 1.0
        result['hma_strength'] = total / 3.0
    elif total <= -2:
        result['hma_trend'] = -1.0
        result['hma_strength'] = abs(total) / 3.0

    result['hma_10'], result['hma_20'], result['hma_50'] = hma_10, hma_20, hma_50
    return result


def compute_technical_indicators(
    close_prices: np.ndarray,
    high_prices: Optional[np.ndarray] = None,
    low_prices: Optional[np.ndarray] = None
) -> Dict[str, float]:
    """
    Compute all technical indicators needed for consensus Signal 5.

    Args:
        close_prices: Array of closing prices (most recent last)
        high_prices: Optional array of high prices
        low_prices: Optional array of low prices

    Returns:
        Dict with:
            macd: MACD histogram normalized by price
            bb_position: Bollinger Band position (0=lower, 0.5=SMA, 1=upper)
            hma_trend: Hull MA trend direction (-1, 0, 1)
            rsi: RSI value (0-100)
    """
    result = {
        'macd': 0.0,
        'bb_position': 0.5,
        'hma_trend': 0.0,
        'rsi': 50.0,
    }

    if close_prices is None or len(close_prices) < 20:
        return result

    ca = np.array(close_prices)

    # MACD (12, 26) normalized by price
    if len(ca) >= 26:
        try:
            ema12 = float(pd.Series(ca).ewm(span=12).mean().iloc[-1])
            ema26 = float(pd.Series(ca).ewm(span=26).mean().iloc[-1])
            result['macd'] = (ema12 - ema26) / ca[-1] if ca[-1] != 0 else 0.0
        except:
            pass

    # Bollinger Bands position (0 = lower band, 0.5 = SMA, 1 = upper band)
    if len(ca) >= 20:
        try:
            sma20 = float(np.mean(ca[-20:]))
            std20 = float(np.std(ca[-20:]))
            bb_upper = sma20 + 2.0 * std20
            bb_lower = sma20 - 2.0 * std20
            if bb_upper != bb_lower:
                bb_pos = (ca[-1] - bb_lower) / (bb_upper - bb_lower)
                result['bb_position'] = float(min(max(bb_pos, 0.0), 1.0))
        except:
            pass

    # Multi-timeframe Hull Moving Average (HMA 10, 20, 50)
    # Requires 2/3 timeframes to align for trend signal
    if len(ca) >= 51:
        try:
            hma_data = compute_multi_timeframe_hma(ca)
            result['hma_trend'] = hma_data['hma_trend']
            result['hma_strength'] = hma_data.get('hma_strength', 0.0)
            result['hma_10'] = hma_data.get('hma_10', 0.0)
            result['hma_20'] = hma_data.get('hma_20', 0.0)
            result['hma_50'] = hma_data.get('hma_50', 0.0)
        except:
            pass
    elif len(ca) >= 21:
        # Fallback to single HMA(20) if not enough data for multi-timeframe
        try:
            hma_current = compute_hma(ca, 20)
            hma_prev = compute_hma(ca[:-1], 20)
            result['hma_trend'] = 1.0 if hma_current > hma_prev else (-1.0 if hma_current < hma_prev else 0.0)
        except:
            pass

    # RSI (14-period)
    if len(ca) >= 15:
        try:
            delta = np.diff(ca[-15:])
            gain = np.clip(delta, 0, None)
            loss = np.clip(-delta, 0, None)
            avg_gain = np.mean(gain)
            avg_loss = np.mean(loss)

            if avg_loss > 0:
                rs = avg_gain / avg_loss
                result['rsi'] = 100.0 - (100.0 / (1.0 + rs))
            elif avg_gain > 0:
                result['rsi'] = 100.0
            else:
                result['rsi'] = 50.0
        except:
            pass

    return result


def compute_market_breadth(
    spy_closes: np.ndarray,
    qqq_closes: np.ndarray,
    lookback: int = 10
) -> float:
    """
    Compute market breadth based on SPY/QQQ momentum agreement.

    Args:
        spy_closes: SPY closing prices
        qqq_closes: QQQ closing prices
        lookback: Period for momentum calculation

    Returns:
        1.0 = both bullish
        -1.0 = both bearish
        0.0 = mixed/neutral
    """
    if spy_closes is None or qqq_closes is None:
        return 0.0
    if len(spy_closes) < lookback + 1 or len(qqq_closes) < lookback + 1:
        return 0.0

    try:
        spy_mom = spy_closes[-1] / spy_closes[-lookback - 1] - 1
        qqq_mom = qqq_closes[-1] / qqq_closes[-lookback - 1] - 1

        if spy_mom > 0 and qqq_mom > 0:
            return 1.0  # Both bullish
        elif spy_mom < 0 and qqq_mom < 0:
            return -1.0  # Both bearish
        else:
            return 0.0  # Mixed
    except:
        return 0.0


def compute_options_flow_signal(
    put_call_ratio: Optional[float] = None,
    volume_put_call_ratio: Optional[float] = None,
    unusual_volume_ratio: Optional[float] = None,
    call_volume: Optional[float] = None,
    put_volume: Optional[float] = None
) -> Dict[str, Any]:
    """
    Compute options flow signal from put/call ratios and volume data.

    Args:
        put_call_ratio: Open interest put/call ratio (normal ~0.7-1.0)
        volume_put_call_ratio: Volume put/call ratio (normal ~0.8-1.2)
        unusual_volume_ratio: Today's volume / avg volume (>2 = unusual)
        call_volume: Total call volume
        put_volume: Total put volume

    Returns:
        Dict with:
            flow_signal: -1 (bearish flow), 0 (neutral), 1 (bullish flow)
            flow_strength: 0-1 strength of signal
            unusual_activity: bool - is there unusual activity
            details: explanation
    """
    result = {
        'flow_signal': 0,
        'flow_strength': 0.0,
        'unusual_activity': False,
        'put_call_ratio': put_call_ratio,
        'volume_put_call_ratio': volume_put_call_ratio,
        'details': 'No options data'
    }

    if put_call_ratio is None and volume_put_call_ratio is None:
        return result

    # Use volume P/C ratio if available, else OI P/C ratio
    pcr = volume_put_call_ratio if volume_put_call_ratio is not None else put_call_ratio

    if pcr is None:
        return result

    # Interpret put/call ratio:
    # PCR < 0.7 = bullish (more calls being bought)
    # PCR 0.7-1.0 = neutral
    # PCR > 1.0 = bearish (more puts being bought)
    # BUT contrarian view: extreme PCR can signal reversal

    if pcr < 0.5:
        # Very bullish flow (lots of call buying)
        result['flow_signal'] = 1
        result['flow_strength'] = min((0.7 - pcr) / 0.4, 1.0)
        result['details'] = f"Bullish flow: PCR={pcr:.2f} (heavy call buying)"
    elif pcr < 0.7:
        # Moderately bullish
        result['flow_signal'] = 1
        result['flow_strength'] = 0.3 + 0.4 * ((0.7 - pcr) / 0.2)
        result['details'] = f"Mildly bullish flow: PCR={pcr:.2f}"
    elif pcr > 1.5:
        # Very bearish flow (lots of put buying)
        # BUT could also be hedging = contrarian bullish
        result['flow_signal'] = -1
        result['flow_strength'] = min((pcr - 1.0) / 1.0, 1.0)
        result['details'] = f"Bearish flow: PCR={pcr:.2f} (heavy put buying)"
    elif pcr > 1.0:
        # Moderately bearish
        result['flow_signal'] = -1
        result['flow_strength'] = 0.3 + 0.4 * ((pcr - 1.0) / 0.5)
        result['details'] = f"Mildly bearish flow: PCR={pcr:.2f}"
    else:
        # Neutral (0.7 to 1.0)
        result['flow_signal'] = 0
        result['flow_strength'] = 0.0
        result['details'] = f"Neutral flow: PCR={pcr:.2f}"

    # Check for unusual volume
    if unusual_volume_ratio is not None and unusual_volume_ratio > 2.0:
        result['unusual_activity'] = True
        result['flow_strength'] = min(result['flow_strength'] * 1.5, 1.0)  # Boost signal on unusual activity
        result['details'] += f" | UNUSUAL ACTIVITY: {unusual_volume_ratio:.1f}x avg volume"

    return result


def compute_vix_term_structure(
    vix: Optional[float] = None,
    vix9d: Optional[float] = None,
    vix3m: Optional[float] = None,
    vxst: Optional[float] = None
) -> Dict[str, Any]:
    """
    Compute VIX term structure indicators (Approach 4 - NEW).

    VIX term structure analysis:
    - Contango (normal): VIX < VIX3M (market calm, expecting future vol)
    - Backwardation (fear): VIX > VIX3M (current fear higher than future)
    - Steep contango: Strong bullish signal (complacency)
    - Steep backwardation: Strong bearish signal (panic)

    Args:
        vix: Current VIX (30-day implied vol)
        vix9d: VIX 9-day (short-term implied vol)
        vix3m: VIX 3-month (longer-term implied vol)
        vxst: VXST (9-day VIX, alternative to vix9d)

    Returns:
        Dict with:
            term_structure: 'contango', 'backwardation', or 'flat'
            term_slope: normalized slope (-1 to 1, negative = backwardation)
            short_term_fear: bool - is short-term vol elevated
            signal: -1 (bearish), 0 (neutral), 1 (bullish)
            strength: 0-1 signal strength
            details: explanation
    """
    result = {
        'term_structure': 'unknown',
        'term_slope': 0.0,
        'short_term_fear': False,
        'signal': 0,
        'strength': 0.0,
        'vix': vix,
        'vix3m': vix3m,
        'vix9d': vix9d or vxst,
        'details': 'No VIX data'
    }

    if vix is None:
        return result

    # Use short-term VIX (9D or VXST)
    short_vix = vix9d if vix9d is not None else vxst

    # Compute term structure slope (VIX vs VIX3M)
    if vix3m is not None and vix3m > 0:
        # Positive slope = contango (normal, bullish)
        # Negative slope = backwardation (fear, bearish)
        term_slope = (vix3m - vix) / vix3m

        if term_slope > 0.05:
            result['term_structure'] = 'contango'
            result['signal'] = 1  # Bullish - market calm
            result['strength'] = min(term_slope * 5, 1.0)  # 20% = max strength
        elif term_slope < -0.05:
            result['term_structure'] = 'backwardation'
            result['signal'] = -1  # Bearish - market fear
            result['strength'] = min(abs(term_slope) * 5, 1.0)
        else:
            result['term_structure'] = 'flat'
            result['signal'] = 0
            result['strength'] = 0.0

        result['term_slope'] = term_slope

    # Check short-term fear (VIX9D > VIX)
    if short_vix is not None and vix > 0:
        short_term_ratio = short_vix / vix
        result['short_term_fear'] = short_term_ratio > 1.1  # 10% above VIX = fear

        # Short-term fear can override bullish signal
        if result['short_term_fear'] and result['signal'] > 0:
            result['signal'] = 0  # Neutralize bullish signal
            result['strength'] *= 0.5

    result['details'] = f"VIX={vix:.1f}, VIX3M={vix3m or 'N/A'}, structure={result['term_structure']}, slope={result['term_slope']:.2f}"

    return result


def compute_options_skew(
    atm_iv: Optional[float] = None,
    otm_put_iv: Optional[float] = None,
    otm_call_iv: Optional[float] = None,
    put_25d_iv: Optional[float] = None,
    call_25d_iv: Optional[float] = None
) -> Dict[str, Any]:
    """
    Compute options skew indicators (Approach 4 - NEW).

    Options skew analysis:
    - Put skew: OTM puts more expensive than ATM (protective buying = fear)
    - Call skew: OTM calls more expensive than ATM (speculative buying = greed)
    - Skew ratio: Put IV / Call IV (>1 = bearish, <1 = bullish)

    Args:
        atm_iv: At-the-money implied volatility
        otm_put_iv: Out-of-the-money put IV (e.g., 10% OTM)
        otm_call_iv: Out-of-the-money call IV (e.g., 10% OTM)
        put_25d_iv: 25-delta put IV (standard skew measure)
        call_25d_iv: 25-delta call IV (standard skew measure)

    Returns:
        Dict with:
            skew_ratio: put IV / call IV (>1 = bearish skew)
            put_skew: put IV - ATM IV (positive = elevated put demand)
            call_skew: call IV - ATM IV (positive = elevated call demand)
            signal: -1 (bearish), 0 (neutral), 1 (bullish)
            strength: 0-1 signal strength
            details: explanation
    """
    result = {
        'skew_ratio': 1.0,
        'put_skew': 0.0,
        'call_skew': 0.0,
        'signal': 0,
        'strength': 0.0,
        'details': 'No options skew data'
    }

    # Use 25-delta if available, else OTM
    put_iv = put_25d_iv if put_25d_iv is not None else otm_put_iv
    call_iv = call_25d_iv if call_25d_iv is not None else otm_call_iv

    if put_iv is None or call_iv is None:
        return result

    # Compute skew ratio (put IV / call IV)
    if call_iv > 0:
        skew_ratio = put_iv / call_iv
        result['skew_ratio'] = skew_ratio

        # Interpret skew
        if skew_ratio > 1.15:
            # Heavy put skew = bearish (lots of put buying for protection)
            result['signal'] = -1
            result['strength'] = min((skew_ratio - 1.0) * 2, 1.0)
        elif skew_ratio < 0.90:
            # Heavy call skew = bullish (speculative call buying)
            result['signal'] = 1
            result['strength'] = min((1.0 - skew_ratio) * 2, 1.0)
        else:
            result['signal'] = 0
            result['strength'] = 0.0

    # Compute individual skews relative to ATM
    if atm_iv is not None and atm_iv > 0:
        if put_iv is not None:
            result['put_skew'] = (put_iv - atm_iv) / atm_iv
        if call_iv is not None:
            result['call_skew'] = (call_iv - atm_iv) / atm_iv

    result['details'] = f"Skew ratio={result['skew_ratio']:.2f}, put_skew={result['put_skew']:.2f}, call_skew={result['call_skew']:.2f}"

    return result


def compute_advanced_features(
    vix: Optional[float] = None,
    vix9d: Optional[float] = None,
    vix3m: Optional[float] = None,
    atm_iv: Optional[float] = None,
    put_25d_iv: Optional[float] = None,
    call_25d_iv: Optional[float] = None,
    put_call_ratio: Optional[float] = None
) -> Dict[str, Any]:
    """
    Compute all advanced features for improved prediction (Approach 4 - NEW).

    Combines:
    - VIX term structure
    - Options skew
    - Put/call ratio
    - Composite fear/greed indicator

    Args:
        vix: Current VIX
        vix9d: VIX 9-day
        vix3m: VIX 3-month
        atm_iv: At-the-money IV
        put_25d_iv: 25-delta put IV
        call_25d_iv: 25-delta call IV
        put_call_ratio: Put/call volume or OI ratio

    Returns:
        Dict with all computed features and composite signals
    """
    # Compute individual features
    term_struct = compute_vix_term_structure(vix, vix9d, vix3m)
    skew = compute_options_skew(atm_iv, put_25d_iv=put_25d_iv, call_25d_iv=call_25d_iv)

    # Compute composite fear/greed indicator
    # Combine: term structure + skew + put/call ratio
    signals = []
    weights = []

    if term_struct['signal'] != 0:
        signals.append(term_struct['signal'] * term_struct['strength'])
        weights.append(0.4)  # 40% weight to term structure

    if skew['signal'] != 0:
        signals.append(skew['signal'] * skew['strength'])
        weights.append(0.3)  # 30% weight to skew

    if put_call_ratio is not None:
        # PCR < 0.7 = bullish, PCR > 1.0 = bearish
        if put_call_ratio < 0.7:
            pcr_signal = 1.0 * min((0.7 - put_call_ratio) / 0.3, 1.0)
        elif put_call_ratio > 1.0:
            pcr_signal = -1.0 * min((put_call_ratio - 1.0) / 0.5, 1.0)
        else:
            pcr_signal = 0.0
        signals.append(pcr_signal)
        weights.append(0.3)  # 30% weight to PCR

    # Compute weighted composite
    if signals and weights:
        total_weight = sum(weights)
        composite = sum(s * w for s, w in zip(signals, weights)) / total_weight
    else:
        composite = 0.0

    # Determine composite signal
    if composite > 0.2:
        composite_signal = 1  # Bullish
        composite_strength = min(abs(composite) / 0.5, 1.0)
    elif composite < -0.2:
        composite_signal = -1  # Bearish
        composite_strength = min(abs(composite) / 0.5, 1.0)
    else:
        composite_signal = 0
        composite_strength = 0.0

    return {
        'vix_term_structure': term_struct,
        'options_skew': skew,
        'put_call_ratio': put_call_ratio,
        'composite_signal': composite_signal,
        'composite_strength': composite_strength,
        'composite_raw': composite,
        'details': f"Term={term_struct['term_structure']}, Skew={skew['skew_ratio']:.2f}, PCR={put_call_ratio or 'N/A'}, Composite={composite:.2f}"
    }


def compute_volume_profile(
    volumes: np.ndarray,
    lookback: int = 20
) -> Dict[str, float]:
    """
    Compute volume profile indicators.

    Args:
        volumes: Array of volume values
        lookback: Lookback period for average

    Returns:
        Dict with volume_ratio, volume_trend, is_spike
    """
    if volumes is None or len(volumes) < lookback + 1:
        return {
            'volume_ratio': 1.0,
            'volume_trend': 0.0,
            'is_spike': False
        }

    try:
        current_vol = float(volumes[-1])
        avg_vol = float(np.mean(volumes[-lookback-1:-1]))

        if avg_vol <= 0:
            return {'volume_ratio': 1.0, 'volume_trend': 0.0, 'is_spike': False}

        ratio = current_vol / avg_vol

        # Volume trend: compare recent vs older average
        if len(volumes) >= lookback * 2:
            recent_avg = float(np.mean(volumes[-lookback:]))
            older_avg = float(np.mean(volumes[-lookback*2:-lookback]))
            trend = (recent_avg - older_avg) / older_avg if older_avg > 0 else 0
        else:
            trend = 0.0

        return {
            'volume_ratio': ratio,
            'volume_trend': trend,
            'is_spike': ratio > 2.0  # Volume > 2x average = spike
        }
    except:
        return {'volume_ratio': 1.0, 'volume_trend': 0.0, 'is_spike': False}


def add_indicators_to_signal(
    signal: Dict,
    price_data: Optional[pd.DataFrame],
    spy_data: Optional[pd.DataFrame] = None,
    qqq_data: Optional[pd.DataFrame] = None,
    options_data: Optional[Dict] = None
) -> None:
    """
    Add all technical indicators to a signal dict in-place.

    Args:
        signal: Signal dict to update
        price_data: DataFrame with 'close' column
        spy_data: Optional SPY data for market breadth
        qqq_data: Optional QQQ data for market breadth
        options_data: Optional dict with put_call_ratio, volume data
    """
    if signal is None:
        return

    # Compute core technical indicators
    if price_data is not None and len(price_data) >= 20:
        close_col = 'Close' if 'Close' in price_data.columns else 'close'
        indicators = compute_technical_indicators(price_data[close_col].values)
        signal['macd'] = indicators['macd']
        signal['bb_position'] = indicators['bb_position']
        signal['hma_trend'] = indicators['hma_trend']
        signal['rsi'] = indicators['rsi']

        # Compute volume profile if volume column exists
        vol_col = 'Volume' if 'Volume' in price_data.columns else 'volume'
        if vol_col in price_data.columns:
            vol_profile = compute_volume_profile(price_data[vol_col].values)
            signal['volume_ratio'] = vol_profile['volume_ratio']
            signal['volume_trend'] = vol_profile['volume_trend']
            signal['volume_spike'] = vol_profile['is_spike']
    else:
        signal['macd'] = 0.0
        signal['bb_position'] = 0.5
        signal['hma_trend'] = 0.0
        signal['rsi'] = 50.0

    # Compute market breadth
    if spy_data is not None and qqq_data is not None:
        spy_col = 'Close' if 'Close' in spy_data.columns else 'close'
        qqq_col = 'Close' if 'Close' in qqq_data.columns else 'close'
        signal['market_breadth'] = compute_market_breadth(
            spy_data[spy_col].values,
            qqq_data[qqq_col].values
        )
    else:
        signal['market_breadth'] = 0.0

    # Compute options flow signal if data available
    if options_data is not None:
        flow = compute_options_flow_signal(
            put_call_ratio=options_data.get('oi_put_call_ratio'),
            volume_put_call_ratio=options_data.get('vol_put_call_ratio'),
            unusual_volume_ratio=options_data.get('unusual_volume_ratio'),
            call_volume=options_data.get('call_volume'),
            put_volume=options_data.get('put_volume')
        )
        signal['options_flow_signal'] = flow['flow_signal']
        signal['options_flow_strength'] = flow['flow_strength']
        signal['options_unusual_activity'] = flow['unusual_activity']
        signal['put_call_ratio'] = flow.get('put_call_ratio', 1.0)
    else:
        signal['options_flow_signal'] = 0
        signal['options_flow_strength'] = 0.0
        signal['options_unusual_activity'] = False
        signal['put_call_ratio'] = 1.0
