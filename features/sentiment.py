#!/usr/bin/env python3
"""
Sentiment Features Module

Integrates multiple sentiment data sources for SPY options trading:
- Fear & Greed Index (alternative.me API)
- Polygon.io News Sentiment
- Put/Call Ratio (from options data)
- VIX-based sentiment

These features can be used as:
1. Hard filters (only trade on extreme sentiment)
2. Soft features (input to neural network)
3. Position sizing multipliers
"""

import requests
import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple
from datetime import datetime, timedelta
import logging
import os
import json
import time

logger = logging.getLogger(__name__)

# =============================================================================
# CONFIGURATION
# =============================================================================

# Fear & Greed thresholds
FEAR_GREED_EXTREME_FEAR = 25    # Below this = extreme fear (contrarian bullish)
FEAR_GREED_FEAR = 40            # Below this = fear
FEAR_GREED_GREED = 60           # Above this = greed
FEAR_GREED_EXTREME_GREED = 75   # Above this = extreme greed (contrarian bearish)

# Put/Call Ratio thresholds
PCR_EXTREME_FEAR = 1.3          # High PCR = fear (contrarian bullish)
PCR_FEAR = 1.1
PCR_GREED = 0.8
PCR_EXTREME_GREED = 0.6         # Low PCR = greed (contrarian bearish)

# VIX thresholds
VIX_EXTREME_FEAR = 30           # Very high VIX = fear
VIX_ELEVATED = 20               # Elevated VIX
VIX_COMPLACENT = 12             # Low VIX = complacency

# Cache settings
_cache = {}
_cache_ttl = 300  # 5 minutes

# =============================================================================
# FEAR & GREED INDEX
# =============================================================================

def fetch_fear_greed_index(use_cache: bool = True) -> Dict:
    """
    Fetch CNN Fear & Greed Index from alternative.me API.

    Returns:
        Dict with:
        - value: 0-100 (0=extreme fear, 100=extreme greed)
        - classification: 'Extreme Fear', 'Fear', 'Neutral', 'Greed', 'Extreme Greed'
        - timestamp: When the reading was taken
    """
    cache_key = 'fear_greed'

    # Check cache
    if use_cache and cache_key in _cache:
        cached_time, cached_data = _cache[cache_key]
        if time.time() - cached_time < _cache_ttl:
            return cached_data

    try:
        response = requests.get(
            'https://api.alternative.me/fng/',
            timeout=10
        )
        response.raise_for_status()
        data = response.json()

        if data.get('data') and len(data['data']) > 0:
            entry = data['data'][0]
            result = {
                'value': int(entry.get('value', 50)),
                'classification': entry.get('value_classification', 'Neutral'),
                'timestamp': int(entry.get('timestamp', 0)),
                'source': 'alternative.me'
            }

            # Cache result
            _cache[cache_key] = (time.time(), result)
            logger.info(f"Fear & Greed Index: {result['value']} ({result['classification']})")
            return result

    except Exception as e:
        logger.warning(f"Failed to fetch Fear & Greed Index: {e}")

    # Return neutral on failure
    return {
        'value': 50,
        'classification': 'Neutral',
        'timestamp': int(time.time()),
        'source': 'default'
    }


def get_fear_greed_features(fg_data: Optional[Dict] = None) -> Dict[str, float]:
    """
    Convert Fear & Greed Index to trading features.

    Returns normalized features for neural network input.
    """
    if fg_data is None:
        fg_data = fetch_fear_greed_index()

    value = fg_data.get('value', 50)

    # Normalized value (0-1)
    fg_normalized = value / 100.0

    # Extreme indicators (binary-ish, smoothed)
    extreme_fear = max(0, (FEAR_GREED_EXTREME_FEAR - value) / FEAR_GREED_EXTREME_FEAR)
    extreme_greed = max(0, (value - FEAR_GREED_EXTREME_GREED) / (100 - FEAR_GREED_EXTREME_GREED))

    # Contrarian signal (-1 to +1, positive = bullish)
    # Extreme fear = bullish, extreme greed = bearish
    contrarian = (50 - value) / 50.0

    return {
        'fg_value': fg_normalized,
        'fg_extreme_fear': extreme_fear,
        'fg_extreme_greed': extreme_greed,
        'fg_contrarian_signal': contrarian,
        'fg_is_fear': 1.0 if value < FEAR_GREED_FEAR else 0.0,
        'fg_is_greed': 1.0 if value > FEAR_GREED_GREED else 0.0,
    }


# =============================================================================
# POLYGON NEWS SENTIMENT
# =============================================================================

def fetch_polygon_news_sentiment(
    ticker: str = 'SPY',
    api_key: Optional[str] = None,
    limit: int = 10,
    use_cache: bool = True
) -> Dict:
    """
    Fetch news sentiment from Polygon.io API.

    Args:
        ticker: Stock symbol (default SPY)
        api_key: Polygon API key (or from env/config)
        limit: Number of recent news articles to analyze

    Returns:
        Dict with aggregated sentiment scores
    """
    cache_key = f'polygon_news_{ticker}'

    # Check cache
    if use_cache and cache_key in _cache:
        cached_time, cached_data = _cache[cache_key]
        if time.time() - cached_time < _cache_ttl:
            return cached_data

    # Get API key
    if api_key is None:
        api_key = os.environ.get('POLYGON_API_KEY')
        if api_key is None:
            # Try loading from config
            try:
                with open('config.json') as f:
                    config = json.load(f)
                    api_key = config.get('credentials', {}).get('polygon', {}).get('api_key')
            except:
                pass

    if not api_key:
        logger.warning("No Polygon API key available")
        return {'sentiment_score': 0.0, 'article_count': 0, 'source': 'default'}

    try:
        url = f'https://api.polygon.io/v2/reference/news'
        params = {
            'ticker': ticker,
            'limit': limit,
            'order': 'desc',
            'apiKey': api_key
        }

        response = requests.get(url, params=params, timeout=15)
        response.raise_for_status()
        data = response.json()

        articles = data.get('results', [])

        if not articles:
            return {'sentiment_score': 0.0, 'article_count': 0, 'source': 'polygon'}

        # Aggregate sentiment from articles
        # Polygon provides sentiment in article insights
        sentiments = []
        for article in articles:
            insights = article.get('insights', [])
            for insight in insights:
                if insight.get('ticker') == ticker:
                    sentiment = insight.get('sentiment', 'neutral')
                    if sentiment == 'positive':
                        sentiments.append(1.0)
                    elif sentiment == 'negative':
                        sentiments.append(-1.0)
                    else:
                        sentiments.append(0.0)

        avg_sentiment = np.mean(sentiments) if sentiments else 0.0

        result = {
            'sentiment_score': float(avg_sentiment),
            'article_count': len(articles),
            'positive_count': sum(1 for s in sentiments if s > 0),
            'negative_count': sum(1 for s in sentiments if s < 0),
            'source': 'polygon'
        }

        # Cache result
        _cache[cache_key] = (time.time(), result)
        logger.info(f"Polygon News Sentiment for {ticker}: {avg_sentiment:.2f} ({len(articles)} articles)")
        return result

    except Exception as e:
        logger.warning(f"Failed to fetch Polygon news: {e}")
        return {'sentiment_score': 0.0, 'article_count': 0, 'source': 'default'}


def get_polygon_sentiment_features(
    ticker: str = 'SPY',
    api_key: Optional[str] = None
) -> Dict[str, float]:
    """
    Convert Polygon news sentiment to trading features.
    """
    data = fetch_polygon_news_sentiment(ticker, api_key)

    sentiment = data.get('sentiment_score', 0.0)
    article_count = data.get('article_count', 0)

    # Normalized sentiment (-1 to +1 already)
    # Scale article count (more articles = more signal confidence)
    article_confidence = min(1.0, article_count / 10.0)

    return {
        'news_sentiment': sentiment,
        'news_sentiment_abs': abs(sentiment),
        'news_article_count': article_confidence,
        'news_bullish': 1.0 if sentiment > 0.3 else 0.0,
        'news_bearish': 1.0 if sentiment < -0.3 else 0.0,
    }


# =============================================================================
# PUT/CALL RATIO SENTIMENT
# =============================================================================

def get_pcr_sentiment_features(
    oi_put_call_ratio: float,
    vol_put_call_ratio: Optional[float] = None
) -> Dict[str, float]:
    """
    Convert Put/Call Ratio to sentiment features.

    High PCR = fear (contrarian bullish)
    Low PCR = greed (contrarian bearish)
    """
    pcr = oi_put_call_ratio

    # Normalized PCR (center around 1.0)
    pcr_normalized = (pcr - 1.0) / 0.5  # -1 to +1 range for typical values
    pcr_normalized = np.clip(pcr_normalized, -2, 2)

    # Extreme indicators
    extreme_fear = max(0, (pcr - PCR_EXTREME_FEAR) / 0.5)
    extreme_greed = max(0, (PCR_EXTREME_GREED - pcr) / 0.3)

    # Contrarian signal
    contrarian = pcr_normalized  # High PCR = bullish signal

    features = {
        'pcr_value': pcr,
        'pcr_normalized': pcr_normalized,
        'pcr_extreme_fear': extreme_fear,
        'pcr_extreme_greed': extreme_greed,
        'pcr_contrarian_signal': contrarian,
        'pcr_is_fear': 1.0 if pcr > PCR_FEAR else 0.0,
        'pcr_is_greed': 1.0 if pcr < PCR_GREED else 0.0,
    }

    # Add volume-based PCR if available
    if vol_put_call_ratio is not None:
        vol_pcr_norm = (vol_put_call_ratio - 1.0) / 0.5
        features['vol_pcr_normalized'] = np.clip(vol_pcr_norm, -2, 2)

    return features


# =============================================================================
# VIX-BASED SENTIMENT
# =============================================================================

def get_vix_sentiment_features(vix_level: float) -> Dict[str, float]:
    """
    Convert VIX level to sentiment features.

    High VIX = fear
    Low VIX = complacency
    """
    # Normalized VIX (typical range 10-40)
    vix_normalized = (vix_level - 20) / 15  # Center around 20
    vix_normalized = np.clip(vix_normalized, -1, 2)

    # Extreme indicators
    extreme_fear = max(0, (vix_level - VIX_EXTREME_FEAR) / 10)
    complacent = max(0, (VIX_COMPLACENT - vix_level) / 5)

    # Contrarian signal (high VIX = bullish, low VIX = bearish)
    contrarian = vix_normalized

    return {
        'vix_sentiment': vix_normalized,
        'vix_extreme_fear': extreme_fear,
        'vix_complacent': complacent,
        'vix_contrarian_signal': contrarian,
        'vix_is_elevated': 1.0 if vix_level > VIX_ELEVATED else 0.0,
        'vix_is_low': 1.0 if vix_level < VIX_COMPLACENT else 0.0,
    }


# =============================================================================
# COMBINED SENTIMENT
# =============================================================================

def compute_all_sentiment_features(
    vix_level: float = 20.0,
    oi_put_call_ratio: float = 1.0,
    vol_put_call_ratio: Optional[float] = None,
    fetch_external: bool = True,
    polygon_api_key: Optional[str] = None
) -> Dict[str, float]:
    """
    Compute all sentiment features from available sources.

    Args:
        vix_level: Current VIX level
        oi_put_call_ratio: Open interest put/call ratio
        vol_put_call_ratio: Volume put/call ratio (optional)
        fetch_external: Whether to fetch external APIs (Fear & Greed, Polygon)
        polygon_api_key: API key for Polygon (optional)

    Returns:
        Dict of all sentiment features
    """
    features = {}

    # VIX sentiment (always available)
    features.update(get_vix_sentiment_features(vix_level))

    # PCR sentiment (always available if options data loaded)
    features.update(get_pcr_sentiment_features(oi_put_call_ratio, vol_put_call_ratio))

    # External APIs (if enabled)
    if fetch_external:
        # Fear & Greed Index
        try:
            features.update(get_fear_greed_features())
        except Exception as e:
            logger.warning(f"Failed to get Fear & Greed features: {e}")
            features.update({
                'fg_value': 0.5,
                'fg_extreme_fear': 0.0,
                'fg_extreme_greed': 0.0,
                'fg_contrarian_signal': 0.0,
                'fg_is_fear': 0.0,
                'fg_is_greed': 0.0,
            })

        # Polygon News Sentiment
        try:
            features.update(get_polygon_sentiment_features('SPY', polygon_api_key))
        except Exception as e:
            logger.warning(f"Failed to get Polygon sentiment: {e}")
            features.update({
                'news_sentiment': 0.0,
                'news_sentiment_abs': 0.0,
                'news_article_count': 0.0,
                'news_bullish': 0.0,
                'news_bearish': 0.0,
            })

    # Composite sentiment score
    # Combine all contrarian signals
    contrarian_signals = [
        features.get('fg_contrarian_signal', 0),
        features.get('pcr_contrarian_signal', 0),
        features.get('vix_contrarian_signal', 0),
    ]
    features['composite_contrarian'] = np.mean([s for s in contrarian_signals if s != 0] or [0])

    # Overall fear level (0-1)
    fear_indicators = [
        features.get('fg_extreme_fear', 0),
        features.get('pcr_extreme_fear', 0),
        features.get('vix_extreme_fear', 0),
    ]
    features['composite_fear'] = np.mean(fear_indicators)

    # Overall greed level (0-1)
    greed_indicators = [
        features.get('fg_extreme_greed', 0),
        features.get('pcr_extreme_greed', 0),
        features.get('vix_complacent', 0),
    ]
    features['composite_greed'] = np.mean(greed_indicators)

    return features


def should_trade_sentiment(
    features: Dict[str, float],
    direction: str,  # 'call' or 'put'
    require_alignment: bool = True
) -> Tuple[bool, str]:
    """
    Determine if sentiment conditions favor trading.

    Args:
        features: Sentiment features dict
        direction: 'call' for bullish, 'put' for bearish
        require_alignment: If True, require sentiment to align with direction

    Returns:
        (should_trade, reason)
    """
    contrarian = features.get('composite_contrarian', 0)
    fear = features.get('composite_fear', 0)
    greed = features.get('composite_greed', 0)

    # Extreme sentiment is actionable
    if fear > 0.5:
        if direction == 'call':
            return True, f"Extreme fear (contrarian bullish) - fear={fear:.2f}"
        else:
            return False, f"Extreme fear favors calls, not puts"

    if greed > 0.5:
        if direction == 'put':
            return True, f"Extreme greed (contrarian bearish) - greed={greed:.2f}"
        else:
            return False, f"Extreme greed favors puts, not calls"

    # Moderate alignment check
    if require_alignment:
        if direction == 'call' and contrarian > 0.2:
            return True, f"Sentiment aligns with call - contrarian={contrarian:.2f}"
        elif direction == 'put' and contrarian < -0.2:
            return True, f"Sentiment aligns with put - contrarian={contrarian:.2f}"
        else:
            return False, f"Sentiment neutral/misaligned - contrarian={contrarian:.2f}"

    # No strong sentiment signal
    return True, "No sentiment filter applied"


# =============================================================================
# FEATURE NAMES (for feature pipeline integration)
# =============================================================================

SENTIMENT_FEATURE_NAMES = [
    # VIX-based
    'vix_sentiment',
    'vix_extreme_fear',
    'vix_complacent',
    'vix_contrarian_signal',
    'vix_is_elevated',
    'vix_is_low',
    # PCR-based
    'pcr_value',
    'pcr_normalized',
    'pcr_extreme_fear',
    'pcr_extreme_greed',
    'pcr_contrarian_signal',
    'pcr_is_fear',
    'pcr_is_greed',
    # Fear & Greed
    'fg_value',
    'fg_extreme_fear',
    'fg_extreme_greed',
    'fg_contrarian_signal',
    'fg_is_fear',
    'fg_is_greed',
    # Polygon News
    'news_sentiment',
    'news_sentiment_abs',
    'news_article_count',
    'news_bullish',
    'news_bearish',
    # Composites
    'composite_contrarian',
    'composite_fear',
    'composite_greed',
]


if __name__ == '__main__':
    # Test the sentiment features
    logging.basicConfig(level=logging.INFO)

    print("Testing Sentiment Features Module")
    print("=" * 50)

    # Test Fear & Greed
    print("\n1. Fear & Greed Index:")
    fg = fetch_fear_greed_index()
    print(f"   Value: {fg['value']} ({fg['classification']})")

    # Test Polygon (if API key available)
    print("\n2. Polygon News Sentiment:")
    poly = fetch_polygon_news_sentiment()
    print(f"   Sentiment: {poly['sentiment_score']:.2f} ({poly['article_count']} articles)")

    # Test combined features
    print("\n3. Combined Sentiment Features:")
    features = compute_all_sentiment_features(
        vix_level=22.5,
        oi_put_call_ratio=1.15,
        fetch_external=True
    )
    print(f"   Composite Contrarian: {features['composite_contrarian']:.2f}")
    print(f"   Composite Fear: {features['composite_fear']:.2f}")
    print(f"   Composite Greed: {features['composite_greed']:.2f}")

    # Test trading decision
    print("\n4. Trading Decisions:")
    should_call, reason = should_trade_sentiment(features, 'call')
    print(f"   CALL: {should_call} - {reason}")
    should_put, reason = should_trade_sentiment(features, 'put')
    print(f"   PUT: {should_put} - {reason}")
