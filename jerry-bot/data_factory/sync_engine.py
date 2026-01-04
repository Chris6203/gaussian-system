# data_factory/sync_engine.py
import pandas as pd
import os

class MTFSyncEngine:
    def __init__(self, symbol, timeframes=['5', '60', 'D']):
        self.symbol = symbol
        self.timeframes = timeframes
        self.data_store = {}
        self._load_all()

    def _load_all(self):
        print(f"\n[DataSync] Syncing Timeframes for {self.symbol}...")
        for tf in self.timeframes:
            path = os.path.join("reports", self.symbol, f"{self.symbol}_{tf}.csv")
            
            if os.path.exists(path):
                df = pd.read_csv(path, parse_dates=['timestamp'])
                
                # Strip TZ awareness for consistent comparison
                df['timestamp'] = pd.to_datetime(df['timestamp']).dt.tz_localize(None)
                
                df.set_index('timestamp', inplace=True)
                df.sort_index(inplace=True)
                self.data_store[tf] = df
                print(f"  -> {tf} Loaded & Normalized (Naive)")
            else:
                print(f"  !! Missing {tf} at {path}")

    def get_snapshot(self, current_time):
        # Ensure the incoming clock time is also naive for comparison
        if current_time.tzinfo is not None:
            current_time = current_time.replace(tzinfo=None)
            
        snapshot = {}
        for tf, df in self.data_store.items():
            # Standard "Look-back" mask
            mask = df.index <= current_time
            if mask.any():
                snapshot[tf] = df.loc[mask].iloc[-1].to_dict()

            else:
                snapshot[tf] = None
        return snapshot