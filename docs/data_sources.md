# Data Sources Documentation

## Overview

This document describes the data sources, symbols, and data flow in the SPY options trading bot.

## Current Data Sources

### 1. Tradier (Primary)
- **Module**: `backend/tradier_data_source.py`
- **Class**: `TradierDataSource`
- **Capabilities**:
  - Real-time equity quotes (`get_quote()`)
  - Historical OHLCV data (`get_historical_data()`)
  - Intraday bars (`_get_intraday_data()`)
  - Options chains with greeks (`get_options_chain()`)
  - Options expirations (`get_options_expirations()`)
- **Rate Limiting**: 1 second between requests
- **Auth**: Bearer token via `TRADIER_API_KEY` or `config.json`

### 2. Polygon.io (Secondary)
- **Module**: `backend/polygon_data_source.py`
- **Class**: `PolygonDataSource`
- **Capabilities**:
  - Historical aggregate bars (`get_historical_data()`)
  - Last trade quotes (`get_quote()`)
- **Rate Limiting**: 5 requests/minute (free tier)
- **Auth**: API key via `POLYGON_API_KEY` or `config.json`

### 3. Yahoo Finance (Fallback)
- **Module**: `backend/data_sources.py`
- **Class**: `YahooFinanceDataSource`
- **Capabilities**:
  - Historical OHLCV data
  - Current prices
  - Basic options chains
- **Rate Limiting**: 10 seconds between requests, 100 daily limit
- **Auth**: None required

## Unified Data Manager

- **Module**: `backend/enhanced_data_sources.py`
- **Class**: `EnhancedDataSource`
- **Purpose**: Provides unified interface with automatic fallback
- **Fallback Order**: Tradier → Polygon → Yahoo Finance

## Symbols Currently Fetched

| Symbol | Description | Primary Source | Used For |
|--------|-------------|----------------|----------|
| SPY | SPDR S&P 500 ETF | Tradier | Primary trading instrument |
| QQQ | Invesco QQQ Trust | Tradier | Tech sector correlation |
| ^VIX | CBOE Volatility Index | Tradier (as VIX) | Volatility regime |
| BTC-USD | Bitcoin | Yahoo Finance | Crypto risk-on/off signal |
| UUP | Invesco DB USD Index | Tradier | Dollar strength proxy |

## Data Flow Architecture

```
config.json
    │
    ├── credentials.tradier.data_api_token
    ├── credentials.polygon.api_key
    └── data_fetching.symbols
           │
           ▼
    ┌──────────────────────────────────────┐
    │         EnhancedDataSource           │
    │     (backend/enhanced_data_sources)  │
    └──────────────────────────────────────┘
           │
           ├──► TradierDataSource (primary)
           ├──► PolygonDataSource (secondary)
           └──► YahooFinanceDataSource (fallback)
                      │
                      ▼
    ┌──────────────────────────────────────┐
    │         UnifiedOptionsBot            │
    │   (unified_options_trading_bot.py)   │
    └──────────────────────────────────────┘
           │
           ├── create_features() → 50 features for LSTM
           ├── get_cross_asset_features() → 8 cross-asset features
           └── _maybe_train_hmm() → HMM regime detection
```

## Feature Engineering Entry Points

### Main Features (`create_features()`)
- Location: `unified_options_trading_bot.py` lines 1287-1620
- Output: 50-element numpy array
- Features:
  - Price & returns (4)
  - Volatility (3)
  - Volume + derivatives (6)
  - RSI (1)
  - Bollinger Bands (2)
  - MACD (1)
  - Range features (2)
  - Jerk analysis (6)
  - VIX features (4)
  - HMM regime (10)
  - Momentum & trend (11)

### Cross-Asset Features (`get_cross_asset_features()`)
- Location: `unified_options_trading_bot.py` lines 1171-1285
- Output: 8-element dict
- Features:
  - `spy_qqq_corr`: SPY/QQQ correlation
  - `spy_qqq_rel_strength`: QQQ relative strength
  - `dollar_strength`: UUP momentum
  - `dollar_equity_corr`: Dollar-equity correlation
  - `crypto_corr`: BTC correlation
  - `market_breadth`: SPY/QQQ direction agreement
  - `risk_on_off`: Combined risk indicator
  - `context_confidence`: Data availability score

## Configuration

Data sources are configured in `config.json`:

```json
{
  "credentials": {
    "tradier": { "data_api_token": "..." },
    "polygon": { "api_key": "..." }
  },
  "data_fetching": {
    "symbols": ["BTC-USD", "^VIX", "SPY", "QQQ", "UUP"]
  },
  "data_sources": {
    "fallback_order": ["tradier", "polygon", "yahoo", "csv"],
    "interval": "1m",
    "historical_period": "7d"
  }
}
```

## Database Storage

- **Historical Data**: `data/db/historical.db`
- **Paper Trading**: `data/paper_trading.db`
- **Market Data Cache**: `data/market_data_cache.db`




