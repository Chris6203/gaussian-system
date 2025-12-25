# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Data Manager is a Python application for collecting and storing historical market data (equities and options). It fetches price data from yfinance and Tradier APIs, stores it in SQLite, and provides a Flask web dashboard for configuration and monitoring.

## Commands

```bash
# Setup (Ubuntu)
./install_ubuntu.sh /opt/data-manager

# Activate virtual environment
source venv/bin/activate

# Run collector (continuous)
python run.py run

# Run collector once
python run.py run --once

# Start web dashboard
python run.py web --host 0.0.0.0 --port 5050

# Backfill historical data (N days)
python run.py backfill 30

# Database utilities
python run.py stats    # Show collection statistics
python run.py check    # Check data integrity
python run.py dedup    # Remove duplicate records
```

## Architecture

### Entry Point
- `run.py` - CLI with subcommands: `run`, `web`, `backfill`, `stats`, `check`, `dedup`

### Core Modules (app/)
- `collector.py` - Main collection loop, orchestrates price/options fetching
- `fetchers.py` - `DataFetcher` (prices via yfinance/Tradier), `LiquidityFetcher` (options chains via Tradier)
- `storage.py` - `DataStorage` class wrapping SQLite with thread-safe writes
- `settings.py` - Loads config from `config.json` and `.env` files
- `web.py` - Flask dashboard with session auth and REST API
- `auth.py` - `UserStore` for bcrypt-based file authentication
- `market_hours.py` - US market hours detection

### Data Flow
1. `Collector.run_forever()` loops on configurable interval
2. For each symbol, fetches latest price via `DataFetcher` (tries yfinance first, falls back to Tradier)
3. During market hours, optionally fetches options chain snapshots via `LiquidityFetcher`
4. All data saved to SQLite tables: `historical_data`, `liquidity_snapshots`, `collection_log`

### Configuration
- `config.json` - Symbols, collector settings, options chain parameters
- `.env` - API tokens (TRADIER_DATA_API_TOKEN, TRADIER_ACCESS_TOKEN) and path overrides (DM_DB_PATH, DM_CONFIG_PATH, etc.)

### Web API Endpoints (require auth)
- `/api/stats` - Collection statistics
- `/api/symbols` - GET/POST tracked symbols
- `/api/collector` - GET/POST collector settings
- `/api/options_chain` - GET/POST options chain collection config
- `/api/credentials` - GET/POST API tokens (stored in .env)
- `/api/backfill` - POST to start background backfill
- `/api/service/restart` - POST to restart systemd services
- `/api/sentiment` - GET/POST sentiment records (supports external bot integration)
- `/api/sentiment/summary` - GET sentiment statistics

### Sentiment API
External bots can POST sentiment data from news/headline analysis:
```json
{
  "symbol": "SPY",
  "sentiment_type": "bearish",
  "value": -65,
  "source": "reuters",
  "headline": "Fed signals more rate hikes...",
  "url": "https://...",
  "confidence": 0.92,
  "model": "finbert"
}
```
Duplicate headlines (same source + headline + symbol) are automatically ignored.
