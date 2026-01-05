#!/usr/bin/env python3
"""
Real-time Trading Dashboard - Modern Console Edition
================================================================================
Matches the training_dashboard.html web UI with Rich console rendering.
Run this WHILE train_then_go_live.py or go_live_only.py is running.

Features:
- Phase indicator (Training/Live Paper/Live Trading)
- Stats bar (Balance, Positions, Win Rate, Cycles, Trades, RL Updates)
- Market data (SPY, VIX, QQQ, BTC)
- HMM Regime detection
- TCN/LSTM multi-timeframe predictions
- Open positions with unrealized P&L
- Recent trades table with P&L tracking

Usage:
    python dashboard.py
"""

import re
import time
from datetime import datetime
from collections import deque
from pathlib import Path
import glob
import json
import sqlite3

from rich.console import Console
from rich.live import Live
from rich.table import Table
from rich.panel import Panel
from rich.layout import Layout
from rich.text import Text
from rich import box
from rich.progress import Progress, BarColumn, TextColumn

console = Console()


class OpenPosition:
    """Represents an open trading position"""
    def __init__(self, trade_id, option_type, entry_price, strike_price, entry_time, current_premium=None):
        self.trade_id = trade_id
        self.option_type = option_type  # CALL or PUT
        self.entry_price = entry_price
        self.strike_price = strike_price
        self.entry_time = entry_time
        self.current_premium = current_premium or entry_price
        self.unrealized_pnl = 0.0
        self.is_real_trade = False
        
    def update_premium(self, new_premium):
        self.current_premium = new_premium
        # P&L = (current - entry) * 100 (for 1 contract)
        self.unrealized_pnl = (new_premium - self.entry_price) * 100


class LogDashboard:
    """Modern dashboard that matches the training web UI"""
    
    def __init__(self):
        # Session info
        self.start_time = time.time()
        self.phase = "Starting..."
        self.trading_mode = "Unknown"
        self.account_id = None
        
        # Progress counters
        self.cycles = 0
        self.signals = 0
        self.trades = 0
        self.rl_updates = 0
        self.exit_decisions = 0
        
        # Account metrics
        self.initial_balance = 1000.0
        self.current_balance = 1000.0
        self.pnl = 0.0
        self.pnl_pct = 0.0
        self.unrealized_pnl = 0.0
        
        # Position tracking
        self.current_positions = 0
        self.max_positions = 2
        self.open_positions = []  # List of OpenPosition objects
        
        # Win/Loss tracking
        self.wins = 0
        self.losses = 0
        self.win_rate = 0.0
        
        # Signal info
        self.last_signal = "WAITING"
        self.signal_confidence = 0.0
        
        # Market prices
        self.spy_price = 0.0
        self.qqq_price = 0.0
        self.vix_value = 0.0
        self.btc_price = 0.0
        self.uup_price = 0.0
        
        # Trading symbol from config
        self.symbol = self._load_symbol_from_config()
        self.symbol_price = 0.0
        
        # HMM Regime
        self.hmm_regime = "Unknown"
        self.hmm_confidence = 0.0
        self.hmm_trend = ""
        self.hmm_volatility = ""
        
        # TCN/LSTM Predictions
        self.lstm_predictions = {}
        self.lstm_warmup_current = 0
        self.lstm_warmup_needed = 30
        
        # Liquidity
        self.liquidity_spread = 0.0
        self.liquidity_volume = 0
        self.liquidity_oi = 0
        self.liquidity_quality = 0
        
        # Recent trades
        self.recent_trades = deque(maxlen=10)
        
        # Simulated time
        self.simulated_date = "----"
        self.simulated_time = "--:--:--"
        
        # Log file state
        self.last_file_position = 0
        self.log_file = None
        self.log_file_path = None
        self.last_file_check = 0
        self.last_db_check = 0
        self.session_started = False
        
    def _load_symbol_from_config(self):
        """Load trading symbol from config.json"""
        try:
            with open('config.json', 'r') as f:
                config = json.load(f)
            return config.get('trading', {}).get('symbol', 'SPY')
        except:
            return 'SPY'
    
    def _load_initial_balance_from_config(self):
        """Load initial balance from config.json"""
        try:
            with open('config.json', 'r') as f:
                config = json.load(f)
            return config.get('trading', {}).get('initial_balance', 1000.0)
        except:
            return 1000.0
        
    def find_latest_log(self):
        """Find the most recent log file"""
        # Check for daily rotating log first
        daily_log = Path("logs/trading_bot.log")
        if daily_log.exists():
            return str(daily_log)
        
        # Fallback to old naming conventions
        log_files = glob.glob("logs/real_bot_*.log")
        if not log_files:
            log_files = glob.glob("logs/*.log")
        
        if not log_files:
            return None
        
        return max(log_files, key=lambda f: Path(f).stat().st_mtime)
    
    def _load_from_database(self):
        """Load data from paper_trading.db"""
        try:
            conn = sqlite3.connect('data/paper_trading.db')
            cursor = conn.cursor()
            
            # Load wins/losses
            cursor.execute('''
                SELECT COUNT(*) FROM trades 
                WHERE status = 'CLOSED' AND profit_loss > 0
            ''')
            self.wins = cursor.fetchone()[0]
            
            cursor.execute('''
                SELECT COUNT(*) FROM trades 
                WHERE status = 'CLOSED' AND profit_loss < 0
            ''')
            self.losses = cursor.fetchone()[0]
            
            total = self.wins + self.losses
            self.win_rate = (self.wins / total * 100) if total > 0 else 0
            
            # Load open positions
            cursor.execute('''
                SELECT id, option_type, entry_price, strike_price, timestamp, is_real_trade
                FROM trades 
                WHERE status IN ('OPEN', 'FILLED')
                ORDER BY id DESC
                LIMIT 10
            ''')
            rows = cursor.fetchall()
            
            self.open_positions = []
            for row in rows:
                trade_id, option_type, entry_price, strike_price, timestamp, is_real = row
                pos = OpenPosition(
                    trade_id=trade_id,
                    option_type=option_type or 'UNKNOWN',
                    entry_price=entry_price or 0,
                    strike_price=strike_price or 0,
                    entry_time=timestamp
                )
                pos.is_real_trade = bool(is_real)
                self.open_positions.append(pos)
            
            self.current_positions = len(self.open_positions)
            
            # Calculate unrealized P&L
            self.unrealized_pnl = sum(p.unrealized_pnl for p in self.open_positions)
            
            # Load recent closed trades
            cursor.execute('''
                SELECT timestamp, option_type, entry_price, exit_price, 
                       profit_loss, status, exit_timestamp, strike_price, is_real_trade
                FROM trades 
                ORDER BY id DESC 
                LIMIT 10
            ''')
            rows = cursor.fetchall()
            
            self.recent_trades.clear()
            for row in reversed(rows):
                timestamp, option_type, entry_price, exit_price, profit_loss, status, exit_timestamp, strike, is_real = row
                
                try:
                    trade_dt = datetime.fromisoformat(timestamp)
                    trade_date = trade_dt.strftime('%m/%d')
                    trade_time = trade_dt.strftime('%H:%M')
                except:
                    trade_date = '--'
                    trade_time = '--'
                
                exit_time_str = '--'
                if exit_timestamp:
                    try:
                        exit_dt = datetime.fromisoformat(exit_timestamp)
                        exit_time_str = exit_dt.strftime('%H:%M')
                    except:
                        pass
                
                pnl_str = ''
                if status == 'CLOSED' and profit_loss is not None:
                    pnl_str = f"${profit_loss:+.2f}"
                
                self.recent_trades.append({
                    'date': trade_date,
                    'time': trade_time,
                    'action': option_type or 'UNKNOWN',
                    'strike': strike or 0,
                    'entry_price': entry_price or 0,
                    'exit_price': exit_price or 0,
                    'exit_time': exit_time_str,
                    'pnl': pnl_str,
                    'pnl_value': profit_loss or 0,
                    'status': status or 'UNKNOWN',
                    'is_real': bool(is_real)
                })
            
            conn.close()
        except Exception as e:
            pass
    
    def parse_log_line(self, line: str):
        """Parse log lines and update metrics"""
        try:
            # Session reset detection
            if "FULL RESET" in line and "Clearing all old data" in line:
                self._reset_session()
                self.session_started = True
            
            if "Bot Configuration:" in line:
                self.session_started = True
            
            # Phase detection
            if "LIVE SHADOW TRADING" in line:
                self.phase = "Live Trading"
            elif "Paper Trading" in line or "simulation_mode = True" in line:
                self.phase = "Paper Trading"
            elif "Training" in line and "mode" in line.lower():
                self.phase = "Training"
            elif "Tradier Sandbox" in line or "sandbox" in line.lower():
                self.phase = "Sandbox"
                self.trading_mode = "Tradier Sandbox"
            
            # Account ID
            if "Account VA" in line or "account=" in line:
                account_match = re.search(r'(VA\d+)', line)
                if account_match:
                    self.account_id = account_match.group(1)
            
            # Simulated timestamp from CYCLE header
            if "[CYCLE" in line and "]" in line and "|" in line:
                timestamp_match = re.search(r'\[CYCLE\s+\d+\]\s+\|\s+(\d{4}-\d{2}-\d{2})\s+(\d{2}:\d{2}:\d{2})', line)
                if timestamp_match:
                    self.simulated_date = timestamp_match.group(1)
                    self.simulated_time = timestamp_match.group(2)
                    self.lstm_predictions = {}  # Clear for new cycle
            
            # Price parsing
            price_pattern = rf'{self.symbol}(?:\s+Price)?:\s*\$([\d.]+)'
            price_match = re.search(price_pattern, line, re.IGNORECASE)
            if price_match:
                self.symbol_price = float(price_match.group(1))
            
            if "SPY Price:" in line or "SPY:" in line:
                match = re.search(r'SPY(?:\s+Price)?:\s*\$([\d.]+)', line)
                if match:
                    self.spy_price = float(match.group(1))
            
            if "QQQ Price:" in line or "QQQ:" in line:
                match = re.search(r'QQQ(?:\s+Price)?:\s*\$([\d.]+)', line)
                if match:
                    self.qqq_price = float(match.group(1))
            
            if "BTC" in line and "Price:" in line:
                match = re.search(r'(?:BTC-USD|BTC|BITX)\s*Price:\s*\$([\d,.]+)', line)
                if match:
                    self.btc_price = float(match.group(1).replace(',', ''))
            
            if "VIX:" in line and "VIX data" not in line.lower():
                match = re.search(r'VIX:\s*\$?([\d.]+)', line)
                if match:
                    val = float(match.group(1))
                    if val < 100:
                        self.vix_value = val
            
            # Balance parsing
            if "Balance:" in line and "$" in line:
                if "Initial" not in line and "Configuration" not in line and "Bot" not in line:
                    match = re.search(r'Balance:\s*\$?([\d,]+\.[\d]+)', line)
                    if match:
                        self.current_balance = float(match.group(1).replace(',', ''))
                        self.pnl = self.current_balance - self.initial_balance
                        self.pnl_pct = (self.pnl / self.initial_balance) * 100 if self.initial_balance > 0 else 0
            
            # Positions
            if "Active Positions:" in line:
                match = re.search(r'Active Positions:\s*(\d+)/(\d+)', line)
                if match:
                    self.current_positions = int(match.group(1))
                    self.max_positions = int(match.group(2))
            
            # Progress summary
            if "Progress:" in line and "Cycles:" in line:
                match = re.search(r'Cycles:\s*([\d,]+)', line)
                if match:
                    self.cycles = int(match.group(1).replace(',', ''))
                
                match = re.search(r'Signals:\s*([\d,]+)', line)
                if match:
                    self.signals = int(match.group(1).replace(',', ''))
                
                match = re.search(r'Trades:\s*([\d,]+)', line)
                if match:
                    self.trades = int(match.group(1).replace(',', ''))
            
            # HMM Regime
            if "[HMM] Regime:" in line:
                match = re.search(r'\[HMM\] Regime:\s*([^,]+),\s*Confidence:\s*([\d.]+)%,\s*Trend:\s*([^,]+),\s*Vol:\s*(.+)', line)
                if match:
                    self.hmm_regime = match.group(1).strip()
                    self.hmm_confidence = float(match.group(2))
                    self.hmm_trend = match.group(3).strip()
                    self.hmm_volatility = match.group(4).strip()
            
            # TCN/LSTM predictions
            if "[LSTM]" in line and "direction=" in line:
                match = re.search(r'\[LSTM\]\s+([^:]+):\s+direction=(\w+),\s+return=([+-]?[\d.]+)%,\s+confidence=([\d.]+)%', line)
                if match:
                    timeframe = match.group(1).strip()
                    self.lstm_warmup_current = 0
                    self.lstm_predictions[timeframe] = {
                        'direction': match.group(2),
                        'return': float(match.group(3)),
                        'confidence': float(match.group(4))
                    }
            
            # LSTM warmup
            if "[LSTM] WARMUP:" in line:
                match = re.search(r'WARMUP:\s*(\d+)/(\d+)', line)
                if match and not self.lstm_predictions:
                    self.lstm_warmup_current = int(match.group(1))
                    self.lstm_warmup_needed = int(match.group(2))
            
            # Signal
            if "FINAL Signal:" in line:
                if "BUY_CALLS" in line:
                    self.last_signal = "BUY CALLS"
                elif "BUY_PUTS" in line:
                    self.last_signal = "BUY PUTS"
                elif "BUY_STRADDLE" in line:
                    self.last_signal = "STRADDLE"
                elif "HOLD" in line:
                    self.last_signal = "HOLD"
                
                match = re.search(r'confidence:\s*([\d.]+)%?\)', line)
                if match:
                    val = float(match.group(1))
                    self.signal_confidence = val if val > 1 else val * 100
            
            # RL updates
            if "Learning:" in line:
                match = re.search(r'Learning:\s*([\d,]+)', line)
                if match:
                    self.rl_updates = int(match.group(1).replace(',', ''))
            
            # Exit decisions
            if "[RL] Exit Policy:" in line:
                match = re.search(r'(\d+)\s+exits', line)
                if match:
                    self.exit_decisions = int(match.group(1))
            
            # Liquidity
            if "[LIQUIDITY]" in line and "Spread:" in line:
                match = re.search(r'Spread:\s*([\d.]+)%', line)
                if match:
                    self.liquidity_spread = float(match.group(1))
                match = re.search(r'Volume:\s*(\d+)', line)
                if match:
                    self.liquidity_volume = int(match.group(1))
                match = re.search(r'OI:\s*(\d+)', line)
                if match:
                    self.liquidity_oi = int(match.group(1))
                match = re.search(r'Quality:\s*(\d+)/100', line)
                if match:
                    self.liquidity_quality = int(match.group(1))
            
            # Trade placement
            if "TRADE PLACED:" in line:
                match = re.search(r'TRADE PLACED:\s*(\w+)\s+(?:@|at)\s+\$?([\d.]+)', line)
                if match:
                    action = match.group(1)
                    price = match.group(2)
                    trade_time = f"{self.simulated_date} {self.simulated_time[:5]}" if self.simulated_date != "----" else datetime.now().strftime('%m/%d %H:%M')
                    
                    # Extract strike if available
                    strike_match = re.search(r'Strike:\s*\$?([\d.]+)', line)
                    strike = float(strike_match.group(1)) if strike_match else 0
                    
                    self.recent_trades.append({
                        'date': trade_time.split()[0] if ' ' in trade_time else '--',
                        'time': trade_time.split()[1] if ' ' in trade_time else trade_time,
                        'action': action,
                        'strike': strike,
                        'entry_price': float(price),
                        'exit_price': 0,
                        'exit_time': '--',
                        'pnl': '',
                        'pnl_value': 0,
                        'status': 'OPEN',
                        'is_real': False
                    })
            
            # Net P&L (closed trade)
            if "Net P&L:" in line and "$" in line:
                match = re.search(r'Net P&L:\s*\$([+-]?[\d.]+)', line)
                if match:
                    pnl_value = float(match.group(1))
                    pnl_str = f"${pnl_value:+.2f}"
                    
                    if pnl_value > 0:
                        self.wins += 1
                    elif pnl_value < 0:
                        self.losses += 1
                    
                    # Update recent trade
                    for trade in reversed(self.recent_trades):
                        if not trade.get('pnl') or trade.get('pnl') == '':
                            trade['pnl'] = pnl_str
                            trade['pnl_value'] = pnl_value
                            trade['status'] = 'CLOSED'
                            break
            
            # Cycle summary
            if "Completed cycles:" in line:
                match = re.search(r'Completed cycles:\s*([\d,]+)', line)
                if match:
                    self.cycles = int(match.group(1).replace(',', ''))
            
            if "Signals generated:" in line:
                match = re.search(r'Signals generated:\s*([\d,]+)', line)
                if match:
                    self.signals = int(match.group(1).replace(',', ''))
            
            if "Trades made:" in line:
                match = re.search(r'Trades made:\s*([\d,]+)', line)
                if match:
                    self.trades = int(match.group(1).replace(',', ''))
            
            if "RL updates:" in line:
                match = re.search(r'RL updates:\s*([\d,]+)', line)
                if match:
                    self.rl_updates = int(match.group(1).replace(',', ''))
                    
        except Exception as e:
            pass
    
    def _reset_session(self):
        """Reset all metrics for new session"""
        self.cycles = 0
        self.signals = 0
        self.trades = 0
        self.rl_updates = 0
        self.exit_decisions = 0
        self.initial_balance = self._load_initial_balance_from_config()
        self.current_balance = self.initial_balance
        self.pnl = 0.0
        self.pnl_pct = 0.0
        self.wins = 0
        self.losses = 0
        self.win_rate = 0.0
        self.current_positions = 0
        self.open_positions = []
        self.unrealized_pnl = 0.0
        self.lstm_predictions = {}
        self.hmm_regime = "Unknown"
        self.hmm_confidence = 0.0
        self.recent_trades.clear()
    
    def read_new_lines(self):
        """Read new lines from log file"""
        current_time = time.time()
        
        # Check for new/rotated log file every second
        if current_time - self.last_file_check > 1:
            self.last_file_check = current_time
            latest_log = self.find_latest_log()
            
            if latest_log and latest_log != self.log_file_path:
                if self.log_file:
                    self.log_file.close()
                self.log_file = None
                self.log_file_path = latest_log
                self.last_file_position = 0
        
        # Load from database every 5 seconds
        if current_time - self.last_db_check > 5:
            self.last_db_check = current_time
            self._load_from_database()
        
        if not self.log_file:
            log_path = self.log_file_path or self.find_latest_log()
            if not log_path:
                return
            try:
                self.log_file = open(log_path, 'r', encoding='utf-8', errors='ignore')
                self.log_file_path = log_path
                self.log_file.seek(0, 0)
                self.last_file_position = self.log_file.tell()
            except Exception:
                return
        
        try:
            self.log_file.seek(self.last_file_position)
            new_lines = self.log_file.readlines()
            self.last_file_position = self.log_file.tell()
            
            for line in new_lines:
                self.parse_log_line(line)
        except Exception:
            pass
    
    def generate_layout(self) -> Layout:
        """Generate modern dashboard layout matching training_dashboard.html"""
        layout = Layout()
        
        # Main structure: header, stats_bar, body, footer
        layout.split(
            Layout(name="header", size=3),
            Layout(name="stats_bar", size=5),
            Layout(name="body"),
            Layout(name="footer", size=3)
        )
        
        # Body splits into left (main content) and right (sidebar)
        layout["body"].split_row(
            Layout(name="main", ratio=3),
            Layout(name="sidebar", ratio=2)
        )
        
        # Main content splits
        layout["main"].split(
            Layout(name="cards_row", size=12),
            Layout(name="trades_section")
        )
        
        # Cards row
        layout["cards_row"].split_row(
            Layout(name="signal_card"),
            Layout(name="market_card"),
            Layout(name="hmm_card")
        )
        
        # Sidebar splits
        layout["sidebar"].split(
            Layout(name="lstm_card", ratio=2),
            Layout(name="positions_card", ratio=1)
        )
        
        # ========== HEADER ==========
        header_text = Text()
        header_text.append("📊 ", style="bold")
        header_text.append(f"{self.symbol} TRADING DASHBOARD", style="bold cyan")
        header_text.append(" │ ", style="dim")
        
        # Phase badge
        phase_style = "bold green" if "Live" in self.phase else "bold yellow" if "Paper" in self.phase else "bold magenta"
        phase_emoji = "🟢" if "Live" in self.phase else "🟡" if "Paper" in self.phase else "🔵"
        header_text.append(f"{phase_emoji} {self.phase}", style=phase_style)
        
        if self.account_id:
            header_text.append(f" ({self.account_id})", style="dim")
        
        header_text.append(" │ ", style="dim")
        header_text.append(f"📅 {self.simulated_date} ", style="white")
        header_text.append(f"{self.simulated_time}", style="bold cyan")
        
        layout["header"].update(Panel(header_text, border_style="cyan", padding=(0, 1)))
        
        # ========== STATS BAR ==========
        stats_table = Table(box=box.SIMPLE, show_header=True, padding=(0, 2), expand=True)
        stats_table.add_column("💰 BALANCE", justify="center", style="cyan", width=15)
        stats_table.add_column("📍 POSITIONS", justify="center", style="cyan", width=12)
        stats_table.add_column("🎯 WIN RATE", justify="center", style="cyan", width=12)
        stats_table.add_column("🔄 CYCLES", justify="center", style="cyan", width=10)
        stats_table.add_column("💼 TRADES", justify="center", style="cyan", width=10)
        stats_table.add_column("🧠 RL UPDATES", justify="center", style="cyan", width=12)
        
        # Balance with P&L
        pnl_style = "bold green" if self.pnl >= 0 else "bold red"
        pnl_sign = "+" if self.pnl >= 0 else ""
        balance_text = Text()
        balance_text.append(f"${self.current_balance:,.2f}\n", style="bold white")
        balance_text.append(f"{pnl_sign}${self.pnl:,.2f} ({pnl_sign}{self.pnl_pct:.1f}%)", style=pnl_style)
        
        # Positions with unrealized
        positions_text = Text()
        positions_text.append(f"{self.current_positions}/{self.max_positions}\n", style="bold cyan")
        unreal_style = "green" if self.unrealized_pnl >= 0 else "red"
        unreal_sign = "+" if self.unrealized_pnl >= 0 else ""
        positions_text.append(f"Unreal: {unreal_sign}${self.unrealized_pnl:.2f}", style=unreal_style)
        
        # Win rate
        total_closed = self.wins + self.losses
        self.win_rate = (self.wins / total_closed * 100) if total_closed > 0 else 0
        wr_style = "bold green" if self.win_rate >= 50 else "bold yellow" if self.win_rate >= 40 else "bold red"
        winrate_text = Text()
        winrate_text.append(f"{self.win_rate:.1f}%\n", style=wr_style)
        winrate_text.append(f"{self.wins}W / {self.losses}L", style="dim")
        
        # Runtime
        elapsed = time.time() - self.start_time
        hours = int(elapsed // 3600)
        minutes = int((elapsed % 3600) // 60)
        seconds = int(elapsed % 60)
        cycles_text = Text()
        cycles_text.append(f"{self.cycles:,}\n", style="bold white")
        cycles_text.append(f"{hours:02d}:{minutes:02d}:{seconds:02d}", style="dim")
        
        # Trades
        trades_text = Text()
        trades_text.append(f"{self.trades:,}\n", style="bold white")
        trades_text.append(f"{self.signals:,} signals", style="dim")
        
        # RL Updates
        rl_text = Text()
        rl_text.append(f"{self.rl_updates:,}\n", style="bold cyan")
        rl_text.append(f"{self.exit_decisions} exits", style="dim")
        
        stats_table.add_row(balance_text, positions_text, winrate_text, cycles_text, trades_text, rl_text)
        layout["stats_bar"].update(Panel(stats_table, border_style="blue", title="[bold]📈 Statistics"))
        
        # ========== SIGNAL CARD ==========
        signal_text = Text()
        signal_text.append("\n", style="white")
        
        if "CALL" in self.last_signal:
            signal_text.append("🟢 BUY CALLS", style="bold green")
        elif "PUT" in self.last_signal:
            signal_text.append("🔴 BUY PUTS", style="bold red")
        elif "STRADDLE" in self.last_signal:
            signal_text.append("🟣 STRADDLE", style="bold magenta")
        else:
            signal_text.append("⚪ HOLD", style="bold dim")
        
        signal_text.append("\n\n", style="white")
        conf_style = "bold green" if self.signal_confidence >= 60 else "yellow" if self.signal_confidence >= 40 else "red"
        signal_text.append(f"Confidence: ", style="dim")
        signal_text.append(f"{self.signal_confidence:.1f}%", style=conf_style)
        
        # Confidence bar
        bar_width = 20
        filled = int((self.signal_confidence / 100) * bar_width)
        bar_char = "█" * filled + "░" * (bar_width - filled)
        signal_text.append(f"\n[{bar_char}]", style=conf_style)
        
        layout["signal_card"].update(Panel(signal_text, title="[bold]⚡ Signal", border_style="yellow"))
        
        # ========== MARKET CARD ==========
        market_table = Table(box=box.SIMPLE, show_header=False, padding=(0, 1))
        market_table.add_column("Symbol", style="dim", width=6)
        market_table.add_column("Price", style="bold white", width=10)
        
        # Primary symbol
        if self.symbol_price > 0:
            market_table.add_row(self.symbol, f"${self.symbol_price:.2f}")
        elif self.spy_price > 0:
            market_table.add_row("SPY", f"${self.spy_price:.2f}")
        
        # VIX with color coding
        if self.vix_value > 0:
            vix_style = "bold red" if self.vix_value > 25 else "bold yellow" if self.vix_value > 15 else "bold green"
            market_table.add_row(Text("VIX", style="dim"), Text(f"{self.vix_value:.2f}", style=vix_style))
        
        if self.qqq_price > 0:
            market_table.add_row("QQQ", f"${self.qqq_price:.2f}")
        
        if self.btc_price > 0:
            market_table.add_row("BTC", f"${self.btc_price:,.0f}")
        
        layout["market_card"].update(Panel(market_table, title="[bold]📊 Market", border_style="green"))
        
        # ========== HMM CARD ==========
        hmm_text = Text()
        hmm_text.append("\n", style="white")
        
        regime_style = "bold magenta"
        if "Bull" in self.hmm_regime:
            regime_style = "bold green"
        elif "Bear" in self.hmm_regime:
            regime_style = "bold red"
        elif "High Vol" in self.hmm_regime:
            regime_style = "bold yellow"
        
        hmm_text.append(f"{self.hmm_regime}\n", style=regime_style)
        
        conf_style = "green" if self.hmm_confidence >= 70 else "yellow" if self.hmm_confidence >= 50 else "red"
        hmm_text.append(f"Conf: ", style="dim")
        hmm_text.append(f"{self.hmm_confidence:.1f}%\n", style=conf_style)
        
        if self.hmm_trend:
            hmm_text.append(f"Trend: {self.hmm_trend}\n", style="dim")
        if self.hmm_volatility:
            hmm_text.append(f"Vol: {self.hmm_volatility}", style="dim")
        
        layout["hmm_card"].update(Panel(hmm_text, title="[bold]🔮 HMM Regime", border_style="magenta"))
        
        # ========== TCN/LSTM CARD ==========
        lstm_text = Text()
        
        if self.lstm_predictions:
            # Sort by timeframe
            tf_order = ['5min', '10min', '15min', '20min', '30min', '40min', '50min', '1h', '1h20m', '1h40m', '2h']
            sorted_preds = sorted(
                self.lstm_predictions.items(),
                key=lambda x: tf_order.index(x[0]) if x[0] in tf_order else 99
            )
            
            for tf, pred in sorted_preds[:8]:  # Show max 8
                direction = pred.get('direction', 'N/A')
                ret = pred.get('return', 0)
                conf = pred.get('confidence', 0)
                
                dir_style = "bold green" if direction == "UP" else "bold red" if direction == "DOWN" else "dim"
                arrow = "↑" if direction == "UP" else "↓" if direction == "DOWN" else "→"
                
                lstm_text.append(f"{tf:>6}: ", style="cyan")
                lstm_text.append(f"{arrow} ", style=dir_style)
                lstm_text.append(f"{ret:+.1f}% ", style="white")
                lstm_text.append(f"({conf:.0f}%)\n", style="dim")
        
        elif self.lstm_warmup_current > 0 and self.lstm_warmup_current < self.lstm_warmup_needed:
            progress_pct = (self.lstm_warmup_current / self.lstm_warmup_needed) * 100
            lstm_text.append("⏳ Warming up...\n\n", style="yellow")
            lstm_text.append(f"Progress: {self.lstm_warmup_current}/{self.lstm_warmup_needed}\n", style="cyan")
            
            bar_width = 20
            filled = int((progress_pct / 100) * bar_width)
            bar_char = "█" * filled + "░" * (bar_width - filled)
            lstm_text.append(f"[{bar_char}] {progress_pct:.0f}%", style="yellow")
        else:
            lstm_text.append("\nWaiting for predictions...", style="dim")
        
        layout["lstm_card"].update(Panel(lstm_text, title="[bold]🧠 TCN Predictions", border_style="cyan"))
        
        # ========== POSITIONS CARD ==========
        positions_text = Text()
        
        if self.open_positions:
            for pos in self.open_positions[:5]:
                is_call = "CALL" in pos.option_type.upper()
                type_style = "green" if is_call else "red"
                emoji = "🟢" if is_call else "🔴"
                
                positions_text.append(f"{emoji} {pos.option_type}\n", style=type_style)
                positions_text.append(f"  ${pos.strike_price:.0f} @ ${pos.entry_price:.2f}", style="dim")
                
                if pos.is_real_trade:
                    positions_text.append(" [REAL]", style="bold yellow")
                
                pnl_style = "green" if pos.unrealized_pnl >= 0 else "red"
                pnl_sign = "+" if pos.unrealized_pnl >= 0 else ""
                positions_text.append(f" → {pnl_sign}${pos.unrealized_pnl:.2f}\n", style=pnl_style)
        else:
            positions_text.append("\nNo open positions", style="dim")
        
        layout["positions_card"].update(Panel(positions_text, title="[bold]📍 Open Positions", border_style="blue"))
        
        # ========== TRADES TABLE ==========
        trades_table = Table(box=box.ROUNDED, show_header=True, padding=(0, 1), expand=True)
        trades_table.add_column("Date", style="cyan", width=6)
        trades_table.add_column("Time", style="cyan", width=6)
        trades_table.add_column("Type", style="yellow", width=8)
        trades_table.add_column("Entry", style="green", width=8)
        trades_table.add_column("Exit", style="white", width=8)
        trades_table.add_column("P&L", style="white", width=10)
        trades_table.add_column("Status", style="white", width=8)
        
        if self.recent_trades:
            for trade in list(self.recent_trades)[-8:]:
                # Type with color
                action = trade.get('action', 'UNKNOWN')
                type_style = "bold green" if "CALL" in action else "bold red" if "PUT" in action else "bold magenta"
                type_text = Text(action, style=type_style)
                
                # Entry/Exit
                entry_text = f"${trade.get('entry_price', 0):.2f}"
                exit_price = trade.get('exit_price', 0)
                exit_text = f"${exit_price:.2f}" if exit_price > 0 else "--"
                
                # P&L
                pnl_value = trade.get('pnl_value', 0)
                status = trade.get('status', 'UNKNOWN')
                
                if status in ['OPEN', 'FILLED']:
                    pnl_text = Text("--", style="cyan")
                    status_text = Text("OPEN", style="bold cyan")
                else:
                    pnl_style = "bold green" if pnl_value > 0 else "bold red" if pnl_value < 0 else "white"
                    pnl_sign = "+" if pnl_value > 0 else ""
                    pnl_text = Text(f"{pnl_sign}${pnl_value:.2f}", style=pnl_style)
                    status_text = Text(status, style="dim")
                
                # Real trade indicator
                if trade.get('is_real'):
                    type_text.append(" ⭐", style="yellow")
                
                trades_table.add_row(
                    trade.get('date', '--'),
                    trade.get('time', '--'),
                    type_text,
                    entry_text,
                    exit_text,
                    pnl_text,
                    status_text
                )
        else:
            trades_table.add_row("--", "--", "No trades yet", "--", "--", "--", "--")
        
        layout["trades_section"].update(Panel(trades_table, title="[bold]🎯 Recent Trades", border_style="magenta"))
        
        # ========== FOOTER ==========
        footer_text = Text()
        footer_text.append("📁 ", style="white")
        log_name = Path(self.log_file.name).name if self.log_file else "Searching..."
        footer_text.append(f"{log_name}", style="cyan")
        footer_text.append(" │ ", style="dim")
        footer_text.append("Press ", style="white")
        footer_text.append("Ctrl+C", style="bold red")
        footer_text.append(" to stop", style="white")
        
        if self.liquidity_quality > 0:
            footer_text.append(" │ ", style="dim")
            liq_style = "green" if self.liquidity_quality >= 70 else "yellow" if self.liquidity_quality >= 40 else "red"
            footer_text.append(f"💧 Liquidity: {self.liquidity_quality}/100", style=liq_style)
        
        layout["footer"].update(Panel(footer_text, border_style="cyan"))
        
        return layout


def main():
    """Run the modern dashboard"""
    console.clear()
    console.print()
    console.print("[bold cyan]📊 Trading Dashboard - Modern Console Edition[/bold cyan]")
    console.print("[dim]Matching the training web dashboard...[/dim]")
    console.print()
    
    # Check for daily rotating log first
    daily_log = Path("logs/trading_bot.log")
    if daily_log.exists():
        console.print(f"[bold green]✓ Found daily log: {daily_log}[/bold green]")
    else:
        log_files = glob.glob("logs/real_bot_*.log")
        if not log_files:
            log_files = glob.glob("logs/*.log")
        
        if not log_files:
            console.print("[bold red]✗ No log file found![/bold red]")
            console.print()
            console.print("[yellow]Please start the bot first:[/yellow]")
            console.print("[bold cyan]python go_live_only.py models/run_XXXXXXXX[/bold cyan]")
            console.print("[dim]or[/dim]")
            console.print("[bold cyan]python train_then_go_live.py[/bold cyan]")
            console.print()
            return
        else:
            console.print(f"[bold green]✓ Found log files[/bold green]")
    
    console.print("[bold green]Starting dashboard...[/bold green]")
    console.print()
    time.sleep(1)
    
    dashboard = LogDashboard()
    
    try:
        with Live(dashboard.generate_layout(), refresh_per_second=2, console=console) as live:
            while True:
                dashboard.read_new_lines()
                live.update(dashboard.generate_layout())
                time.sleep(0.5)
    except KeyboardInterrupt:
        console.print("\n[yellow]Dashboard stopped[/yellow]")
        if dashboard.log_file:
            dashboard.log_file.close()


if __name__ == "__main__":
    main()
