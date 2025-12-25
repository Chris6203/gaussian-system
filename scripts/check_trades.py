#!/usr/bin/env python3
"""Check trading P&L from paper trading database."""
import sqlite3

db_path = 'data/paper_trading.db'
conn = sqlite3.connect(db_path)
c = conn.cursor()

# Get all trades ordered by entry time
c.execute('''
    SELECT id, timestamp, symbol, option_type, strike_price, premium_paid, 
           entry_price, exit_price, quantity, profit_loss, status, exit_timestamp,
           expiration_date, ml_confidence, ml_prediction
    FROM trades 
    ORDER BY timestamp DESC
''')
trades = c.fetchall()

print(f'Database: {db_path}')
print(f'Total Trades: {len(trades)}')
print('=' * 160)
print(f"{'#':>3} | {'Time':>16} | {'Sym':>4} | {'Type':>4} | {'Strike':>7} | {'Premium':>8} | {'Spot':>7} | {'Exit$':>8} | {'Qty':>3} | {'P/L':>12} | {'Return%':>8} | {'Status':>15} | Prediction")
print('-' * 160)

total_pnl = 0
wins = 0
losses = 0
suspicious = []

for i, t in enumerate(trades, 1):
    id, timestamp, sym, opt_type, strike, premium, entry_price, exit_price, qty, pnl, status, exit_time, expiry, conf, pred = t
    
    total_pnl += pnl or 0
    if pnl:
        if pnl > 0:
            wins += 1
        else:
            losses += 1
    
    # Calculate return %
    if premium and exit_price:
        return_pct = ((exit_price - premium) / premium) * 100
    else:
        return_pct = 0
    
    # Flag suspicious trades
    is_suspicious = False
    reason = ""
    
    # Check for unrealistic returns
    if abs(return_pct) > 200:
        is_suspicious = True
        reason = f"Extreme return: {return_pct:.1f}%"
    
    # Check for PUT trades when prediction was CALLS or vice versa
    if pred and opt_type:
        if ('PUT' in pred.upper() and opt_type.upper() == 'CALL') or \
           ('CALL' in pred.upper() and opt_type.upper() == 'PUT'):
            is_suspicious = True
            reason = f"Direction mismatch: {pred} vs {opt_type}"
    
    # Check if exit price is much higher than entry for a losing trade
    if pnl and pnl < -50 and exit_price and premium:
        if exit_price > premium:
            is_suspicious = True
            reason = f"Loss but exit > entry: ${premium:.2f} -> ${exit_price:.2f}"
    
    # Format strings
    pnl_str = f'${pnl:+.2f}' if pnl else 'OPEN'
    premium_str = f'${premium:.2f}' if premium else 'N/A'
    exit_str = f'${exit_price:.2f}' if exit_price else 'N/A'
    entry_str = f'${entry_price:.2f}' if entry_price else 'N/A'
    time_str = timestamp[:16] if timestamp else 'N/A'
    return_str = f'{return_pct:+.1f}%' if premium and exit_price else 'N/A'
    
    marker = " ⚠️" if is_suspicious else ""
    print(f'{i:>3} | {time_str:>16} | {sym:>4} | {opt_type:>4} | ${strike:>6.0f} | {premium_str:>8} | {entry_str:>7} | {exit_str:>8} | {qty:>3} | {pnl_str:>12} | {return_str:>8} | {status:>15} | {pred or "N/A"}{marker}')
    
    if is_suspicious:
        suspicious.append({
            'id': id,
            'time': timestamp,
            'sym': sym,
            'type': opt_type,
            'strike': strike,
            'premium': premium,
            'exit': exit_price,
            'pnl': pnl,
            'reason': reason
        })

print('-' * 160)
print(f'\nSUMMARY:')
print(f'  Total P/L: ${total_pnl:+.2f}')
print(f'  Wins: {wins}, Losses: {losses}')
if (wins + losses) > 0:
    print(f'  Win Rate: {100*wins/(wins+losses):.1f}%')

# Get account state
c.execute('SELECT balance, total_profit_loss, max_drawdown FROM account_state')
acc = c.fetchone()
if acc:
    print(f'\nACCOUNT STATE:')
    print(f'  Current Balance: ${acc[0]:,.2f}')
    print(f'  Total P/L: ${acc[1]:+,.2f}')
    print(f'  Max Drawdown: {acc[2]*100:.1f}%')

# Show suspicious trades
if suspicious:
    print(f'\n⚠️  SUSPICIOUS TRADES ({len(suspicious)}):')
    print('-' * 80)
    for s in suspicious:
        print(f"  {s['time'][:16]} | {s['sym']} {s['type']} ${s['strike']:.0f} | ${s['premium']:.2f} -> ${s['exit']:.2f} | P/L: ${s['pnl']:+.2f}")
        print(f"    Reason: {s['reason']}")

# Analyze by option type
print('\n--- BY OPTION TYPE ---')
c.execute('''
    SELECT option_type, COUNT(*), SUM(profit_loss), AVG(profit_loss)
    FROM trades WHERE profit_loss IS NOT NULL
    GROUP BY option_type
''')
for row in c.fetchall():
    opt_type, count, total, avg = row
    print(f'  {opt_type}: {count} trades, Total: ${total:+.2f}, Avg: ${avg:+.2f}')

# Analyze by status
print('\n--- BY EXIT STATUS ---')
c.execute('''
    SELECT status, COUNT(*), SUM(profit_loss), AVG(profit_loss)
    FROM trades WHERE profit_loss IS NOT NULL
    GROUP BY status
''')
for row in c.fetchall():
    status, count, total, avg = row
    print(f'  {status}: {count} trades, Total: ${total:+.2f}, Avg: ${avg:+.2f}')

conn.close()
