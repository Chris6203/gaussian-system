# core/broker.py
import datetime as dt
from typing import List, Dict, Optional
from dataclasses import dataclass
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from strategies.options_strategy import OptionQuote, IronCondorLegs, PositionState
from data_factory.polygon_client import PolygonClient

class BrokerAPI:
    def get_spot(self, symbol: str) -> float: ...
    def get_iv_rank(self, symbol: str, lookback_days: int = 252) -> float: ...
    def get_vix(self) -> float: ...
    def get_option_chain(self, symbol: str, expiration: dt.date) -> List[OptionQuote]: ...
    def get_expirations(self, symbol: str) -> List[dt.date]: ...
    def place_iron_condor(self, legs: IronCondorLegs, quantity: int, limit_price: float) -> str: ...
    def get_open_positions(self, symbol: str) -> List[PositionState]: ...
    def close_position(self, position_id: str, limit_price: Optional[float] = None) -> None: ...
    def roll_vertical(self, position_id: str, side: str, new_short: OptionQuote, new_long: OptionQuote, limit_price: float) -> None: ...
    def get_account_metrics(self) -> Dict[str, float]: ...
    def trade_shares(self, symbol: str, quantity: int, side: str, limit_price: Optional[float] = None) -> None: ...

@dataclass
class TradeEvent:
    time: dt.datetime
    type: str  # "open","close","roll","hedge"
    position_id: str
    details: Dict

class PaperBroker(BrokerAPI):
    def __init__(self, polygon_client: PolygonClient, starting_equity: float = 100000.0):
        self.poly = polygon_client
        self.equity = starting_equity
        self.positions: Dict[str, PositionState] = {}
        self.trade_log: List[TradeEvent] = []

    def get_spot(self, symbol: str) -> float:
        return self.poly.get_spot(symbol)

    def get_iv_rank(self, symbol: str, lookback_days: int = 252) -> float:
        return self.poly.get_iv_rank(symbol, lookback_days)

    def get_vix(self) -> float:
        return self.poly.get_vix()

    def get_option_chain(self, symbol: str, expiration: dt.date) -> List[OptionQuote]:
        return self.poly.get_option_chain(symbol, expiration)

    def get_expirations(self, symbol: str) -> List[dt.date]:
        return self.poly.get_expirations(symbol)

    def place_iron_condor(self, legs: IronCondorLegs, quantity: int, limit_price: float) -> str:
        pid = f"IC-{len(self.positions)+1}"
        pos = PositionState(
            id=pid, legs=legs, open_time=dt.datetime.now(),
            credit_received=limit_price, quantity=quantity, adjustments_done={"call":0,"put":0}
        )
        self.positions[pid] = pos
        self.trade_log.append(TradeEvent(dt.datetime.now(), "open", pid, {"credit": limit_price, "qty": quantity}))
        return pid

    def get_open_positions(self, symbol: str) -> List[PositionState]:
        return list(self.positions.values())

    def close_position(self, position_id: str, limit_price: Optional[float] = None) -> None:
        pos = self.positions.pop(position_id, None)
        if pos:
            self.trade_log.append(TradeEvent(dt.datetime.now(), "close", position_id, {"limit_price": limit_price}))

    def roll_vertical(self, position_id: str, side: str, new_short: OptionQuote, new_long: OptionQuote, limit_price: float) -> None:
        pos = self.positions.get(position_id)
        if not pos:
            return
        # Replace legs on the rolled side
        if side == "call":
            pos.legs.short_call = new_short
            pos.legs.long_call = new_long
        else:
            pos.legs.short_put = new_short
            pos.legs.long_put = new_long
        pos.adjustments_done[side] += 1
        self.trade_log.append(TradeEvent(dt.datetime.now(), "roll", position_id, {"side": side, "credit": limit_price}))

    def get_account_metrics(self) -> Dict[str, float]:
        # Simple approximation
        positions_value = sum(p.credit_received * p.quantity for p in self.positions.values())
        return {"equity": self.equity, "positions_value": positions_value}

    def trade_shares(self, symbol: str, quantity: int, side: str, limit_price: Optional[float] = None) -> None:
        self.trade_log.append(TradeEvent(dt.datetime.now(), "hedge", "N/A", {"symbol": symbol, "qty": quantity, "side": side, "price": limit_price}))

    def collect_trade_log(self) -> List[TradeEvent]:
        return self.trade_log

class AlpacaBroker(BrokerAPI):
    """
    Live/Paper Broker using Alpaca-Py SDK for execution 
    and Polygon for market intelligence.
    """
    def __init__(self, run_cfg, polygon_client: PolygonClient):
        from alpaca.trading.client import TradingClient
        
        self.poly = polygon_client
        self.r_cfg = run_cfg
        # Use paper=True by default for safety
        self.client = TradingClient(run_cfg.alpaca_key, run_cfg.alpaca_secret, paper=True)
        
    def get_spot(self, symbol: str) -> float:
        return self.poly.get_spot(symbol)

    def get_iv_rank(self, symbol: str, lookback_days: int = 252) -> float:
        return self.poly.get_iv_rank(symbol, lookback_days)

    def get_vix(self) -> float:
        return self.poly.get_vix()

    def get_option_chain(self, symbol: str, expiration: dt.date) -> List[OptionQuote]:
        return self.poly.get_option_chain(symbol, expiration)

    def get_expirations(self, symbol: str) -> List[dt.date]:
        return self.poly.get_expirations(symbol)

    def place_iron_condor(self, legs: IronCondorLegs, quantity: int, limit_price: float) -> str:
        from alpaca.trading.requests import MarketOrderRequest
        from alpaca.trading.enums import OrderSide, TimeInForce
        
        leg_data = [
            (legs.short_call.symbol, OrderSide.SELL),
            (legs.long_call.symbol, OrderSide.BUY),
            (legs.short_put.symbol, OrderSide.SELL),
            (legs.long_put.symbol, OrderSide.BUY)
        ]
        
        print(f"[Alpaca] Submitting {quantity} Iron Condor(s)...")
        results = []
        for symbol, side in leg_data:
            req = MarketOrderRequest(
                symbol=symbol,
                qty=quantity,
                side=side,
                time_in_force=TimeInForce.GTC
            )
            order = self.client.submit_order(req)
            results.append(order.id)
            
        return "MULTI-" + "-".join([str(r)[:8] for r in results])

    def get_open_positions(self, symbol: str) -> List[PositionState]:
        # Implementation for production would map Alpaca positions back to our structures.
        return []

    def get_account_metrics(self) -> Dict[str, float]:
        acc = self.client.get_account()
        return {
            "equity": float(acc.equity),
            "buying_power": float(acc.options_buying_power),
            "positions_value": float(acc.long_market_value) + float(acc.short_market_value)
        }

    def trade_shares(self, symbol: str, quantity: int, side: str, limit_price: Optional[float] = None) -> None:
        from alpaca.trading.requests import MarketOrderRequest
        from alpaca.trading.enums import OrderSide, TimeInForce
        req = MarketOrderRequest(
            symbol=symbol,
            qty=quantity,
            side=OrderSide.BUY if side == "buy" else OrderSide.SELL,
            time_in_force=TimeInForce.DAY
        )
        self.client.submit_order(req)