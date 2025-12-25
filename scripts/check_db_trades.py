import sqlite3
import pandas as pd
import os

db_path = "data/paper_trading.db"
print(f"Checking DB at: {os.path.abspath(db_path)}")

if not os.path.exists(db_path):
    print("DB file not found!")
    # Try looking in other locations
    possible_paths = [
        "data/db/paper_trading.db",
        "../data/paper_trading.db",
        "output3/data/paper_trading.db"
    ]
    for p in possible_paths:
        if os.path.exists(p):
            print(f"Found DB at: {p}")
            db_path = p
            break

try:
    conn = sqlite3.connect(db_path)
    
    # Check tables
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()
    print(f"Tables: {tables}")
    
    # Get recent trades
    query = """
    SELECT trade_id, symbol, option_type, entry_time, exit_time, 
           entry_price, exit_price, quantity, profit_loss, profit_loss_pct, status
    FROM trades
    WHERE status IN ('PROFIT_TAKEN', 'STOPPED_OUT', 'CLOSED', 'EXPIRED')
    ORDER BY exit_time DESC
    LIMIT 10
    """
    
    df = pd.read_sql_query(query, conn)
    
    if not df.empty:
        print("\nRecent Closed Trades:")
        print(df.to_string())
        
        # Calculate stats for today
        today = '2025-06-09' # Based on terminal output simulation time
        print(f"\nStats for simulation date >= {today}:")
        
        stats_query = f"""
        SELECT COUNT(*) as count, 
               SUM(CASE WHEN profit_loss > 0 THEN 1 ELSE 0 END) as wins,
               SUM(profit_loss) as total_pnl
        FROM trades
        WHERE exit_time >= '{today}'
        """
        stats = pd.read_sql_query(stats_query, conn)
        print(stats.to_string())
        
    else:
        print("No closed trades found.")
        
    conn.close()

except Exception as e:
    print(f"Error: {e}")










