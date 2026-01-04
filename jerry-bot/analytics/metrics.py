# analytics/metrics.py
from typing import List
import matplotlib.pyplot as plt
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.broker import TradeEvent

def summarize(trades: List[TradeEvent]):
    opens = [t for t in trades if t.type == "open"]
    closes = [t for t in trades if t.type == "close"]
    rolls = [t for t in trades if t.type == "roll"]
    hedges = [t for t in trades if t.type == "hedge"]

    # Basic counts
    summary = {
        "opens": len(opens),
        "closes": len(closes),
        "rolls": len(rolls),
        "hedges": len(hedges),
        "net_profit": sum(t.details.get("credit", 0.0) for t in opens) - sum(t.details.get("limit_price", 0.0) or 0.0 for t in closes)
    }
    return summary

def plot_metrics(trades: List[TradeEvent], title: str = "Metrics"):
    summary = summarize(trades)
    print("Live/Paper summary:", summary)

    # Simple bar chart
    labels = list(summary.keys())
    values = [summary[k] if isinstance(summary[k], (int, float)) else 0 for k in labels]

    plt.figure(figsize=(8,4))
    plt.bar(labels, values)
    plt.title(title)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()