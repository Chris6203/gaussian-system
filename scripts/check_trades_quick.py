import sqlite3
conn = sqlite3.connect('data/paper_trading.db')
cursor = conn.cursor()

# Get recent trades
cursor.execute('''
    SELECT id, timestamp, option_type, strike_price, premium_paid, entry_price, 
           exit_price, exit_timestamp, status, profit_loss, is_real_trade, tradier_option_symbol 
    FROM trades 
    ORDER BY timestamp DESC 
    LIMIT 15
''')
rows = cursor.fetchall()

print("Recent Trades Analysis")
print("="*120)
print(f"{'ID':<10} | {'Entry Time':<18} | {'Type':<10} | {'Strike':<8} | {'Premium':<8} | {'Exit$':<8} | {'P/L':<10} | {'Status':<15} | {'Real'}")
print("-"*120)

for r in rows:
    trade_id = r[0][:8] if r[0] else "N/A"
    timestamp = r[1][:18] if r[1] else "N/A"
    opt_type = r[2] if r[2] else "N/A"
    strike = f"${r[3]:.2f}" if r[3] else "N/A"
    entry_premium = f"${r[4]:.2f}" if r[4] else "N/A"
    exit_price = f"${r[6]:.2f}" if r[6] else "OPEN"
    pnl = f"${r[9]:.2f}" if r[9] else "$0.00"
    status = r[8] if r[8] else "N/A"
    is_real = "YES" if r[10] else "NO"
    
    print(f"{trade_id:<10} | {timestamp:<18} | {opt_type:<10} | {strike:<8} | {entry_premium:<8} | {exit_price:<8} | {pnl:<10} | {status:<15} | {is_real}")

# Get summary stats
cursor.execute("SELECT COUNT(*), SUM(profit_loss), SUM(CASE WHEN profit_loss > 0 THEN 1 ELSE 0 END), SUM(CASE WHEN profit_loss < 0 THEN 1 ELSE 0 END) FROM trades WHERE status != 'FILLED'")
summary = cursor.fetchone()
print("\n" + "="*60)
print("CLOSED TRADES SUMMARY")
print("="*60)
print(f"Total Closed: {summary[0]}")
print(f"Total P/L: ${summary[1]:.2f}" if summary[1] else "Total P/L: $0.00")
print(f"Winners: {summary[2] or 0}")
print(f"Losers: {summary[3] or 0}")
if summary[2] and summary[3] and (summary[2] + summary[3]) > 0:
    print(f"Win Rate: {summary[2] / (summary[2] + summary[3]) * 100:.1f}%")

# Check for directional accuracy
print("\n" + "="*80)
print("CHECKING OPTION PREMIUM ACCURACY (Entry Premium vs Exit Premium)")
print("="*80)

cursor.execute('''
    SELECT id, option_type, premium_paid, exit_price, entry_price, profit_loss, status, strike_price
    FROM trades 
    WHERE status IN ('STOPPED_OUT', 'PROFIT_TAKEN') 
    ORDER BY timestamp DESC 
    LIMIT 10
''')

print(f"{'ID':<10} | {'Type':<6} | {'Entry$':<8} | {'Exit$':<8} | {'Chg$':<8} | {'P/L':<10} | {'Check'}")
print("-"*80)

for row in cursor.fetchall():
    trade_id = row[0][:8]
    opt_type = "CALL" if "CALL" in str(row[1]) else "PUT"
    entry_premium = row[2]  # premium_paid (option premium at entry)
    exit_premium = row[3]   # exit_price (option premium at exit)
    underlying_entry = row[4]  # entry_price (underlying price at entry)
    pnl = row[5]
    status = row[6]
    strike = row[7]
    
    if entry_premium and exit_premium:
        premium_change = exit_premium - entry_premium
        # P/L should be (exit - entry) * 100 * quantity (usually 1)
        expected_pnl = premium_change * 100  # Assuming 1 contract
        
        # Check if P/L makes sense
        # Positive premium_change should mean profit
        if premium_change > 0:
            expected_result = "PROFIT"
        else:
            expected_result = "LOSS"
        
        actual_result = "PROFIT" if pnl > 0 else "LOSS"
        
        # Also check if the math is roughly correct (accounting for fees)
        pnl_diff = abs(pnl - expected_pnl)
        math_ok = pnl_diff < 50  # Allow $50 difference for fees/rounding
        
        if expected_result == actual_result and math_ok:
            check = "✓ OK"
        elif expected_result != actual_result:
            check = f"✗ Direction! Expected {expected_result}"
        else:
            check = f"~ Math off by ${pnl_diff:.2f}"
        
        print(f"{trade_id:<10} | {opt_type:<6} | ${entry_premium:<7.2f} | ${exit_premium:<7.2f} | ${premium_change:<+7.2f} | ${pnl:<+9.2f} | {check}")
    else:
        print(f"{trade_id:<10} | {opt_type:<6} | Missing data")

# Check the big winning trades specifically
print("\n" + "="*80)
print("ANALYZING BIG WINNERS (The CALLs you asked about)")
print("="*80)

cursor.execute('''
    SELECT id, option_type, premium_paid, exit_price, entry_price, profit_loss, status, strike_price, timestamp, exit_timestamp
    FROM trades 
    WHERE profit_loss > 400
    ORDER BY profit_loss DESC
''')

for row in cursor.fetchall():
    trade_id = row[0][:8]
    opt_type = "CALL" if "CALL" in str(row[1]) else "PUT"
    entry_premium = row[2]
    exit_premium = row[3]
    underlying_entry = row[4]
    pnl = row[5]
    strike = row[7]
    entry_time = row[8][:19] if row[8] else "N/A"
    exit_time = row[9][:19] if row[9] else "N/A"
    
    print(f"\n Trade: {trade_id}")
    print(f"  Type: {opt_type}, Strike: ${strike:.2f}")
    print(f"  Entry Time: {entry_time}")
    print(f"  Exit Time: {exit_time}")
    print(f"  Underlying at Entry: ${underlying_entry:.2f}")
    print(f"  Option Premium at Entry: ${entry_premium:.2f}")
    print(f"  Option Premium at Exit: ${exit_premium:.2f}")
    print(f"  Premium Change: ${exit_premium - entry_premium:.2f}")
    print(f"  P/L: ${pnl:.2f}")
    
    # Calculate expected P/L
    expected = (exit_premium - entry_premium) * 100
    print(f"  Expected P/L (premium change × 100): ${expected:.2f}")
    
    if abs(expected - pnl) > 50:
        print(f"  ⚠️ P/L differs from expected by ${abs(expected - pnl):.2f} (fees?)")
    else:
        print(f"  ✓ P/L matches expected")

conn.close()
