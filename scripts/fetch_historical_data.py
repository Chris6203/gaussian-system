#!/usr/bin/env python3
"""
Fetch Historical 1-Minute Data

This script fetches historical 1-minute data from FMP and stores it in the database.
Useful for backfilling data for training and backtesting.

Usage:
    python scripts/fetch_historical_data.py                    # Fetch default symbols
    python scripts/fetch_historical_data.py SPY QQQ AAPL       # Fetch specific symbols
    python scripts/fetch_historical_data.py --days 30 SPY      # Fetch 30 days
"""

import sys
import os
import argparse
import logging
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from config_loader import load_config

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description='Fetch historical 1-minute data from FMP')
    parser.add_argument('symbols', nargs='*', help='Stock symbols to fetch (default: SPY QQQ IWM)')
    parser.add_argument('--days', type=int, default=7, 
                       help='Days of history to fetch (default: 7, FMP free tier limits may apply)')
    parser.add_argument('--interval', type=str, default='1min',
                       choices=['1min', '5min', '15min', '30min', '1hour'],
                       help='Data interval (default: 1min)')
    args = parser.parse_args()
    
    # Default symbols if none provided
    symbols = args.symbols if args.symbols else ['SPY', 'QQQ', 'IWM', 'DIA']
    
    logger.info("=" * 60)
    logger.info("Historical Data Fetcher (FMP)")
    logger.info("=" * 60)
    logger.info(f"Symbols: {', '.join(symbols)}")
    logger.info(f"Days: {args.days}")
    logger.info(f"Interval: {args.interval}")
    logger.info("=" * 60)
    
    # Load config
    try:
        config = load_config("config.json")
        config_dict = config.config
    except Exception as e:
        logger.error(f"Could not load config: {e}")
        return 1
    
    # Check FMP credentials
    fmp_key = config_dict.get('credentials', {}).get('fmp', {}).get('api_key')
    if not fmp_key:
        logger.error("FMP API key not found in config.json")
        logger.error("Add your FMP API key under credentials.fmp.api_key")
        return 1
    
    # Initialize FMP data source
    try:
        from backend.fmp_data_source import FMPDataSource
        
        fmp = FMPDataSource(api_key=fmp_key)
        logger.info("✓ FMP data source initialized")
        
    except ImportError as e:
        logger.error(f"Could not import FMP data source: {e}")
        return 1
    except Exception as e:
        logger.error(f"Could not initialize FMP: {e}")
        return 1
    
    # Fetch data for each symbol
    results = fmp.fetch_and_store_historical(symbols, args.days, args.interval)
    
    # Print summary
    logger.info("")
    logger.info("=" * 60)
    logger.info("Results Summary")
    logger.info("=" * 60)
    
    total_records = 0
    for symbol, result in results.items():
        if result.get('success'):
            records = result.get('records', 0)
            total_records += records
            start = result.get('start')
            end = result.get('end')
            logger.info(f"✓ {symbol}: {records:,} records ({start} to {end})")
        else:
            error = result.get('error', 'Unknown error')
            logger.warning(f"✗ {symbol}: Failed - {error}")
    
    logger.info("")
    logger.info(f"Total records saved: {total_records:,}")
    logger.info(f"Database: data/db/historical.db")
    logger.info("")
    
    # Show how to query the data
    logger.info("To query this data in Python:")
    logger.info("  from backend.fmp_data_source import FMPDataSource")
    logger.info("  fmp = FMPDataSource(api_key='...')")
    logger.info("  df = fmp.get_from_database('SPY', limit=1000)")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())



