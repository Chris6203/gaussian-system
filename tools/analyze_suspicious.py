#!/usr/bin/env python3
"""Analyze suspicious trades."""
import sqlite3
from datetime import datetime

# Check historical SPY prices around those trade times
conn = sqlite3.connect('data/db/historical.db')
c = conn.cursor()

print('SPY Prices on 2025-12-03:')
print('='*60)

# Check around 11:37 (trade entry)
c.execute('''
    SELECT timestamp, close_price FROM historical_data 
    WHERE symbol='SPY' AND timestamp LIKE '2025-12-03T11:3%'
    ORDER BY timestamp
''')
print('Around 11:37 (entry time):')
for r in c.fetchall():
    print(f'  {r[0]}: ${r[1]:.2f}')

# Check around 10:25 (trade entry)  
c.execute('''
    SELECT timestamp, close_price FROM historical_data 
    WHERE symbol='SPY' AND timestamp LIKE '2025-12-03T10:2%'
    ORDER BY timestamp
''')
print('\nAround 10:25 (entry time):')
for r in c.fetchall():
    print(f'  {r[0]}: ${r[1]:.2f}')

# Check around 16:00 (market close)  
c.execute('''
    SELECT timestamp, close_price FROM historical_data 
    WHERE symbol='SPY' AND timestamp >= '2025-12-03T15:50' AND timestamp <= '2025-12-03T16:30'
    ORDER BY timestamp
''')
print('\nAround market close (15:50-16:30):')
for r in c.fetchall():
    print(f'  {r[0]}: ${r[1]:.2f}')

# Get the latest SPY price
c.execute('''
    SELECT timestamp, close_price FROM historical_data 
    WHERE symbol='SPY'
    ORDER BY timestamp DESC LIMIT 10
''')
print('\nLatest SPY prices:')
for r in c.fetchall():
    print(f'  {r[0]}: ${r[1]:.2f}')

conn.close()

# Now check the paper trading database for how exits are calculated
print('\n' + '='*60)
print('Checking paper trading exit calculations...')
print('='*60)

conn = sqlite3.connect('data/paper_trading.db')
c = conn.cursor()

# Get the suspicious trades
c.execute('''
    SELECT id, timestamp, strike_price, premium_paid, exit_price, profit_loss, quantity, entry_price, exit_timestamp
    FROM trades 
    WHERE timestamp LIKE '2025-12-03%'
    ORDER BY timestamp
''')
print("\nToday's trades:")
for t in c.fetchall():
    id, ts, strike, prem, exit_p, pnl, qty, spot, exit_ts = t
    # Calculate expected P&L
    expected_pnl = (exit_p - prem) * qty * 100
    print(f'  Trade at {ts[:16]}:')
    print(f'    Strike: ${strike:.0f}')
    print(f'    SPY at entry: ${spot:.2f}')
    print(f'    Option premium paid: ${prem:.2f}')
    print(f'    Option exit price: ${exit_p:.2f}')
    print(f'    Quantity: {qty}')
    print(f'    Exit time: {exit_ts}')
    print(f'    Recorded P/L: ${pnl:+.2f}')
    print(f'    Expected P/L: ${expected_pnl:+.2f}')
    print(f'    Diff: ${abs(expected_pnl - pnl):.2f}')
    print()

# Check what the exit price should be based on current SPY price
print('='*60)
print('Exit price analysis:')
print('='*60)

# For a $685 CALL with SPY at ~$684:
# The option should have very little intrinsic value
# If SPY closed at $684, a $685 CALL is OTM and worth mostly theta
# An exit price of $9.12 for a $685 CALL would require SPY to be around $694!

print("""
Trade #1 analysis (11:37 entry):
  - $685 CALL bought at $3.54 premium
  - SPY was at $682.35 at entry (OTM by $2.65)
  - Exit price was $9.12
  - For this to be real, SPY would need to be ~$694 at exit
  - But SPY only went up to ~$684 max today
  
This suggests the option pricing model may be using incorrect prices!
""")

conn.close()

