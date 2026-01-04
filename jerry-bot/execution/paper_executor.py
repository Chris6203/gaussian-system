from dataclasses import dataclass
from datetime import datetime

@dataclass
class PaperTrade:
    trade_id: str
    symbol: str
    option_symbol: str
    quantity: int
    entry_price: float
    entry_time: datetime
    status: str = "OPEN"
    exit_price: float = None
    exit_time: datetime = None
    pnl: float = 0.0
    reason: str = ""

class PaperExecutor:
    def __init__(self, run_config):
        self.run_config = run_config
        self.trades = []

    def open_trade(self, trade_decision):
        trade = PaperTrade(
            trade_id=trade_decision.trade_id,
            symbol=trade_decision.symbol,
            option_symbol=trade_decision.option_symbol,
            quantity=trade_decision.quantity,
            entry_price=trade_decision.entry_price,
            entry_time=datetime.utcnow(),
            reason=trade_decision.rationale
        )
        self.trades.append(trade)
        return trade

    def close_trade(self, trade, exit_price):
        trade.exit_price = exit_price
        trade.exit_time = datetime.utcnow()
        trade.pnl = (exit_price - trade.entry_price) * trade.quantity * 100
        trade.status = "CLOSED"
