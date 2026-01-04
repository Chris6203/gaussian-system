#!/usr/bin/env python3
"""
Fetch SPY data using Polygon API (gaussian-system's data source)
and generate synthetic options for jerry-bot backtesting.
"""

import os
import sys
import datetime as dt
import pandas as pd
import numpy as np
from pathlib import Path

# Add project paths
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from core.config import RunConfig
from data_factory.SyntheticOptionsEngine import SyntheticOptionsEngine, save_synthetic_options

# Try to use polygon client from jerry-bot or main system
try:
    from polygon import RESTClient
except ImportError:
    print("[WARNING] polygon-api-client not installed, trying requests...")
    RESTClient = None

import requests


def fetch_spy_from_polygon(api_key: str, start_date: str, end_date: str, timeframe: str = "5") -> pd.DataFrame:
    """
    Fetch SPY bars from Polygon.io API
    """
    print(f"[Polygon] Fetching SPY {timeframe}-minute bars from {start_date} to {end_date}...")

    # Map timeframe to Polygon multiplier
    tf_map = {
        "1": ("1", "minute"),
        "5": ("5", "minute"),
        "15": ("15", "minute"),
        "30": ("30", "minute"),
        "60": ("1", "hour"),
    }

    multiplier, timespan = tf_map.get(timeframe, ("5", "minute"))

    url = f"https://api.polygon.io/v2/aggs/ticker/SPY/range/{multiplier}/{timespan}/{start_date}/{end_date}"
    params = {
        "adjusted": "true",
        "sort": "asc",
        "limit": 50000,
        "apiKey": api_key
    }

    all_results = []

    while url:
        response = requests.get(url, params=params)
        data = response.json()

        if data.get('status') != 'OK' and data.get('status') != 'DELAYED':
            print(f"[ERROR] Polygon API error: {data.get('status')} - {data.get('message', 'Unknown error')}")
            if 'results' not in data:
                break

        results = data.get('results', [])
        all_results.extend(results)
        print(f"  Fetched {len(results)} bars (total: {len(all_results)})")

        # Check for pagination
        next_url = data.get('next_url')
        if next_url:
            url = next_url
            params = {"apiKey": api_key}  # Only need API key for next page
        else:
            url = None

    if not all_results:
        print("[WARNING] No data returned from Polygon")
        return pd.DataFrame()

    # Convert to DataFrame
    df = pd.DataFrame(all_results)

    # Rename columns to match jerry-bot format
    df = df.rename(columns={
        't': 'timestamp',
        'o': 'open',
        'h': 'high',
        'l': 'low',
        'c': 'close',
        'v': 'volume',
        'vw': 'vwap',
        'n': 'trade_count'
    })

    # Convert timestamp from milliseconds to datetime
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df['symbol'] = 'SPY'

    # Sort by timestamp
    df = df.sort_values('timestamp').reset_index(drop=True)

    print(f"[OK] Fetched {len(df)} bars from {df['timestamp'].min()} to {df['timestamp'].max()}")

    return df


def generate_mtf_data(df_5min: pd.DataFrame, output_dir: str) -> None:
    """
    Generate 15m and 60m timeframe data from 5m bars
    """
    df = df_5min.copy()
    df.set_index('timestamp', inplace=True)

    # Generate 15-minute bars
    df_15m = df.resample('15min').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum',
        'vwap': 'mean',
        'trade_count': 'sum'
    }).dropna()
    df_15m['symbol'] = 'SPY'
    df_15m = df_15m.reset_index()

    out_15m = os.path.join(output_dir, "SPY_15.csv")
    df_15m.to_csv(out_15m, index=False)
    print(f"[OK] Saved 15m data: {len(df_15m)} bars -> {out_15m}")

    # Generate 60-minute bars
    df_60m = df.resample('60min').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum',
        'vwap': 'mean',
        'trade_count': 'sum'
    }).dropna()
    df_60m['symbol'] = 'SPY'
    df_60m = df_60m.reset_index()

    out_60m = os.path.join(output_dir, "SPY_60.csv")
    df_60m.to_csv(out_60m, index=False)
    print(f"[OK] Saved 60m data: {len(df_60m)} bars -> {out_60m}")


def generate_synthetic_options(spot_df: pd.DataFrame, output_path: str, config: RunConfig) -> None:
    """
    Generate synthetic options data from spot prices using Black-Scholes
    """
    print(f"\n[Synthetic Options] Generating options chain from {len(spot_df)} spot prices...")

    engine = SyntheticOptionsEngine(config)

    # Prepare spot data with date column
    spot_for_engine = spot_df.copy()
    spot_for_engine['date'] = spot_for_engine['timestamp'].dt.date

    # Sample to daily closes for efficiency (options chain per day, not per bar)
    daily_closes = spot_for_engine.groupby('date').agg({
        'close': 'last'
    }).reset_index()

    print(f"  Generating options for {len(daily_closes)} trading days...")

    options_df = engine.generate_full_backtest_data(daily_closes, "SPY")

    # Save to CSV
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    options_df.to_csv(output_path, index=False)

    print(f"[OK] Generated {len(options_df)} option records -> {output_path}")
    print(f"  Date range: {options_df['date'].min()} to {options_df['date'].max()}")
    print(f"  Strikes: ${options_df['strike'].min():.0f} to ${options_df['strike'].max():.0f}")


def main():
    print("=" * 60)
    print("Jerry-Bot Data Setup: Fetching SPY data from Polygon")
    print("=" * 60)

    # Load config
    config = RunConfig()

    if not config.polygon_key:
        print("[ERROR] No Polygon API key found in config.json")
        print("  Please add your Polygon API key to E:/gaussian-system/config.json")
        return

    # Paths
    jerry_bot_dir = Path(__file__).parent.parent
    spy_data_dir = jerry_bot_dir / "reports" / "SPY"
    options_dir = jerry_bot_dir / "data" / "synthetic_options"

    os.makedirs(spy_data_dir, exist_ok=True)
    os.makedirs(options_dir, exist_ok=True)

    # Date range for 2024 (default backtest period)
    start_date = "2024-01-01"
    end_date = "2024-12-31"

    # Fetch 5-minute SPY data
    df_5m = fetch_spy_from_polygon(config.polygon_key, start_date, end_date, "5")

    if df_5m.empty:
        print("[ERROR] Failed to fetch SPY data")
        return

    # Save 5-minute data
    out_5m = spy_data_dir / "SPY_5.csv"
    df_5m.to_csv(out_5m, index=False)
    print(f"[OK] Saved 5m data: {len(df_5m)} bars -> {out_5m}")

    # Generate 15m and 60m data for MTF
    generate_mtf_data(df_5m, str(spy_data_dir))

    # Generate synthetic options
    options_path = options_dir / "SPY_5min.csv"
    generate_synthetic_options(df_5m, str(options_path), config)

    print("\n" + "=" * 60)
    print("DATA SETUP COMPLETE!")
    print("=" * 60)
    print(f"  SPY 5m data: {out_5m}")
    print(f"  SPY 15m data: {spy_data_dir}/SPY_15.csv")
    print(f"  SPY 60m data: {spy_data_dir}/SPY_60.csv")
    print(f"  Options data: {options_path}")
    print("\nRun backtest with:")
    print("  python core/main.py --mode backtest --use-mtf --dynamic-sizing --bt-samples 0")


if __name__ == "__main__":
    main()
