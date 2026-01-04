# analytics/audit_logger.py
import datetime as dt

def log_broker_efficiency(order_id, bid, ask, fill):
    """The 'Keep the Broker Honest' audit log."""
    mid = (bid + ask) / 2.0
    slippage = fill - mid
    with open('reports/broker_audit.log', 'a') as f:
        f.write(f'{dt.datetime.now()} | ID: {order_id} | Slippage: {slippage:.4f}\n')