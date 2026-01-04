# data_factory/AlpacaGetData.py - Fixed Configuration Import
import os
import datetime as dt
import pandas as pd
import pytz
import requests
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# === FIXED: Import and instantiate RunConfig properly ===
from core.config import RunConfig

# Create a config instance to access keys
_run_config = RunConfig()

# Initialize Alpaca client with instance attributes
client = StockHistoricalDataClient(_run_config.alpaca_key, _run_config.alpaca_secret)

def main():
    # 1. SYMBOL SELECTION
    symbol = input("Enter symbol (or ? to list available): ").strip().upper()
    
    if symbol == "?":
        choice = input("Select categories: {F=Forex, C=Crypto, E=Equities, T=ETFs, All=All}: ").strip().upper()
        url = "https://paper-api.alpaca.markets/v2/assets"
        headers = {
            "APCA-API-KEY-ID": _run_config.alpaca_key,
            "APCA-API-SECRET-KEY": _run_config.alpaca_secret
        }
        response = requests.get(url, headers=headers)
        assets = response.json()
        
        filtered = []
        for a in assets:
            if a['status'] != 'active':
                continue
            
            cls = a.get('class', '').upper()
            name = a.get('name', '').upper()
            is_etf = 'ETF' in name or 'EXCHANGE TRADED FUND' in name
            
            if choice == 'ALL':
                filtered.append(a['symbol'])
            elif choice == 'F' and cls == 'FX':
                filtered.append(a['symbol'])
            elif choice == 'C' and cls == 'CRYPTO':
                filtered.append(a['symbol'])
            elif choice == 'E' and cls == 'US_EQUITY' and not is_etf:
                filtered.append(a['symbol'])
            elif choice == 'T' and is_etf:
                filtered.append(a['symbol'])

        if filtered:
            print(f"\n--- Found {len(filtered)} Symbols ---")
            for i in range(0, len(filtered), 8):
                print(", ".join(filtered[i:i+8]))
        else:
            print(f"No active symbols found for category: {choice}")
            
        symbol = input("\nEnter symbol from list: ").strip().upper()

    # 2. TIMEFRAME SELECTION
    print("\nAvailable: 1, 5, 15, 30, 60, 2H, 4H, D, W, M")
    tf_input = input("Enter timeframes (comma separated): ")
    selected_labels = [x.strip() for x in tf_input.split(",")]

    tf_map = {
        "1":   TimeFrame(1, TimeFrameUnit.Minute),
        "5":   TimeFrame(5, TimeFrameUnit.Minute),
        "15":  TimeFrame(15, TimeFrameUnit.Minute),
        "30":  TimeFrame(30, TimeFrameUnit.Minute),
        "60":  TimeFrame(1, TimeFrameUnit.Hour),
        "2H":  TimeFrame(2, TimeFrameUnit.Hour),
        "4H":  TimeFrame(4, TimeFrameUnit.Hour),
        "D":   TimeFrame(1, TimeFrameUnit.Day),
        "W":   TimeFrame(1, TimeFrameUnit.Week),
        "M":   TimeFrame(1, TimeFrameUnit.Month)
    }

    base_dir = os.path.join("reports", symbol)
    os.makedirs(base_dir, exist_ok=True)
    summary_stats = []

    # 3. INCREMENTAL UPDATE WITH SIP DELAY
    # Free tier requires 15-minute delay on SIP data
    current_now = dt.datetime.now(pytz.utc) - dt.timedelta(minutes=16)

    for label in selected_labels:
        if label not in tf_map:
            continue
        
        out_path = os.path.join(base_dir, f"{symbol}_{label}.csv")
        
        # INCREMENTAL APPEND LOGIC
        if os.path.exists(out_path):
            existing_df = pd.read_csv(out_path, parse_dates=["timestamp"])
            if not existing_df.empty:
                # Start 1 second after last timestamp to prevent duplicates
                start_dt = existing_df["timestamp"].max() + dt.timedelta(seconds=1)
                print(f"\n[Update] {label}: Resuming from {start_dt}")
                mode = 'update'
            else:
                start_dt = dt.datetime(2024, 1, 1, tzinfo=pytz.utc)
                mode = 'fresh'
        else:
            start_dt = dt.datetime(2024, 1, 1, tzinfo=pytz.utc)
            mode = 'fresh'
            print(f"\n[New] {label}: Starting fresh download from 2024-01-01")

        try:
            print(f"[Fetching] {symbol} {label} bars from {start_dt} to {current_now}")
            bars = client.get_stock_bars(StockBarsRequest(
                symbol_or_symbols=symbol,
                timeframe=tf_map[label],
                start=start_dt,
                end=current_now,
                adjustment='split'
            ))
            
            new_df = bars.df.reset_index()
            
            if mode == 'update' and not new_df.empty:
                df = pd.concat([existing_df, new_df]).drop_duplicates(subset=['timestamp'])
            else:
                df = new_df

            if not df.empty:
                df.to_csv(out_path, index=False)
                print(f"[Saved] Raw data: {len(df)} rows")
                
                # CLEANUP & INTERPOLATE
                final_count = sweep_and_interpolate(out_path, symbol)
                summary_stats.append(f"{label}: {final_count} total rows")
            else:
                print(f"[Warning] No new data found for {label} (API delay may apply)")
                
        except Exception as e:
            print(f"[Error] Processing {label}: {e}")

    print("\n" + "="*60)
    print(f"DATA DOWNLOAD COMPLETE: {symbol}")
    print("="*60)
    for s in summary_stats:
        print(f"  {s}")
    print("="*60)

def sweep_and_interpolate(file_path, symbol_name):
    """Clean data: remove duplicates, fill gaps, interpolate missing values"""
    df = pd.read_csv(file_path, parse_dates=["timestamp"])
    df = df.sort_values("timestamp").drop_duplicates(subset=["timestamp"]).reset_index(drop=True)
    
    # Determine expected frequency
    diffs = df["timestamp"].diff().dt.total_seconds()
    if diffs.isna().all():
        return len(df)
    
    expected = int(diffs.mode()[0])
    
    # Create full timestamp range
    full_range = pd.date_range(df["timestamp"].min(), df["timestamp"].max(), freq=f"{expected}s")
    df = df.set_index("timestamp").reindex(full_range)
    
    # Interpolate numeric columns
    numeric_cols = ["open", "high", "low", "close", "volume", "trade_count", "vwap"]
    existing = [c for c in numeric_cols if c in df.columns]
    df[existing] = df[existing].interpolate(method="linear")
    
    # Round price columns
    price_cols = ["open", "high", "low", "close", "vwap"]
    for col in price_cols:
        if col in df.columns:
            df[col] = df[col].round(2)
    
    df["symbol"] = symbol_name
    df = df.reset_index().rename(columns={"index": "timestamp"})
    df.to_csv(file_path, index=False)
    
    print(f"[Cleaned] {len(df)} rows after interpolation")
    return len(df)

if __name__ == "__main__":
    main()