# data_factory/polygon_client.py
import datetime as dt
from typing import List
from dataclasses import dataclass
import random

# Replace these stubs with real Polygon REST calls.
# See Polygon docs for /v3/reference/options/contracts, /v3/snapshot/options, etc.

@dataclass
class PolyOption:
    symbol: str
    expiration: dt.date
    strike: float
    is_call: bool
    bid: float
    ask: float
    mid: float
    delta: float
    iv: float

class PolygonClient:
    def __init__(self, api_key: str):
        self.api_key = api_key

    def get_spot(self, symbol: str) -> float:
        # Stub: replace with /v2/last/trade/{ticker} or snapshot
        return 480.0 + random.uniform(-2, 2)

    def get_vix(self) -> float:
        # Stub: use VIX ticker data (e.g., ^VIX via Polygon if available)
        return 18.0 + random.uniform(-1.0, 1.0)

    def get_iv_rank(self, symbol: str, lookback_days: int = 252) -> float:
        # Stub: compute percentile of current IV vs lookback
        return 35.0 + random.uniform(-5.0, 5.0)

    def get_expirations(self, symbol: str) -> List[dt.date]:
        today = dt.date.today()
        # Typical SPY weekly expirations; real implementation: query contracts and collect unique expirations
        exps = [today + dt.timedelta(days=d) for d in range(7, 90, 7)]
        return exps

    def get_option_chain(self, symbol: str, expiration: dt.date) -> List[PolyOption]:
        # Stub: synthesize chain with deltas around target bands
        spot = self.get_spot(symbol)
        strikes = [round(spot + k, 1) for k in range(-50, 51, 5)]
        chain = []
        for s in strikes:
            # Simple delta model: farther OTM -> smaller delta
            call_delta = max(0.01, min(0.80, 1.0 / max(1.0, abs(s - spot)/10.0)))
            put_delta  = call_delta  # symmetric for stub
            iv = 0.20 + abs(s - spot)/100.0
            bid = max(0.01, abs(s - spot)/50.0)
            ask = bid + 0.05
            mid = (bid + ask) / 2.0
            chain.append(PolyOption(symbol=f"SPY {expiration} C {s}", expiration=expiration, strike=s, is_call=True,
                                    bid=bid, ask=ask, mid=mid, delta=call_delta, iv=iv))
            chain.append(PolyOption(symbol=f"SPY {expiration} P {s}", expiration=expiration, strike=s, is_call=False,
                                    bid=bid, ask=ask, mid=mid, delta=put_delta, iv=iv))
        return chain

# Adapter to Strategy OptionQuote
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from strategies.options_strategy import OptionQuote

def poly_to_quote(p: PolyOption) -> OptionQuote:
    return OptionQuote(
        symbol=p.symbol, expiration=p.expiration, strike=p.strike, is_call=p.is_call,
        bid=p.bid, ask=p.ask, mid=p.mid, delta=p.delta, iv=p.iv
    )

# Convenience: convert full chain
def convert_chain(chain: List[PolyOption]) -> List[OptionQuote]:
    return [poly_to_quote(p) for p in chain]