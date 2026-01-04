from dataclasses import dataclass
from datetime import datetime
from typing import List


@dataclass
class TradeDecision:
    trade_id: str
    symbol: str
    option_symbol: str
    direction: str              # "LONG", "SHORT", "NONE"
    quantity: int
    entry_price: float
    agreement_score: float
    approved: bool
    rationale: List[str]
    timestamp: datetime
