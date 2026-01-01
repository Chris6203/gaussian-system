import sqlite3
conn = sqlite3.connect('data/paper_trading.db')
cursor = conn.cursor()

print('=== REAL MONEY TRADES (Dec 31, 2025) ===')
print('')
cursor.execute('''
SELECT timestamp, option_type, strike_price, premium_paid, status, profit_loss
FROM trades
WHERE is_real_trade = 1 AND timestamp LIKE "2025-12-31%"
ORDER BY timestamp
''')
total_pnl = 0
for row in cursor.fetchall():
    ts = row[0][11:16]
    opt = row[1]
    strike = row[2]
    prem = row[3]
    status = row[4]
    pnl = row[5] or 0
    total_pnl += pnl
    result = 'WIN' if pnl > 0 else 'LOSS'
    print(f'{ts} | {opt} ${strike:.0f} | Prem: ${prem:.2f} | {status} | P&L: ${pnl:.2f} ({result})')

print(f'')
print(f'Total Real P&L: ${total_pnl:.2f}')

cursor.execute('SELECT COUNT(*) FROM trades WHERE is_real_trade = 1 AND status = "FILLED"')
open_count = cursor.fetchone()[0]
print(f'Open Positions: {open_count}')

conn.close()
