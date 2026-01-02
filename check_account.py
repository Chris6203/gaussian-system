#!/usr/bin/env python3
"""Quick script to check Tradier live account status."""

import json
import requests

config = json.load(open('config.json'))
live = config['credentials']['tradier']['live']
token = live['access_token']
account_id = live['account_number']

headers = {'Authorization': f'Bearer {token}', 'Accept': 'application/json'}
base = 'https://api.tradier.com/v1'

# Balance
bal = requests.get(f'{base}/accounts/{account_id}/balances', headers=headers).json()
b = bal.get('balances', {})
print('='*50)
print('ACCOUNT STATUS')
print('='*50)
print(f"Total Equity: ${b.get('total_equity', 0):,.2f}")
print(f"Cash:         ${b.get('total_cash', 0):,.2f}")
print(f"Market Value: ${b.get('market_value', 0):,.2f}")

# Positions
pos = requests.get(f'{base}/accounts/{account_id}/positions', headers=headers).json()
print()
print('='*50)
print('OPEN POSITIONS')
print('='*50)
positions = pos.get('positions', 'null')
if positions == 'null' or not positions:
    print('None')
else:
    p = positions.get('position', [])
    if isinstance(p, dict):
        p = [p]
    for pos in p:
        print(f"{pos['symbol']}: {pos['quantity']} shares @ ${pos.get('cost_basis', 0):.2f}")

# Orders
orders = requests.get(f'{base}/accounts/{account_id}/orders', headers=headers).json()
print()
print('='*50)
print('RECENT ORDERS')
print('='*50)
order_data = orders.get('orders', 'null')
if order_data == 'null' or not order_data:
    print('None')
else:
    o = order_data.get('order', [])
    if isinstance(o, dict):
        o = [o]
    for order in o[-10:]:
        sym = order.get('option_symbol', order.get('symbol', 'N/A'))
        side = order.get('side', '')
        status = order.get('status', '')
        qty = order.get('quantity', 0)
        filled = order.get('avg_fill_price', 0)
        date = order.get('create_date', '')[:16]
        print(f"{date} | {status:8} | {side:15} {qty}x @ ${filled:.2f}")
        print(f"           {sym}")
