#!/usr/bin/env python3
"""
Training Dashboard - Log File Viewer
Watches the training log and displays progress
Run this WHILE train_then_go_live.py is running
"""

import sys
import os

# Fix encoding on Windows to support emojis
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
    os.environ['PYTHONIOENCODING'] = 'utf-8'

import re
import time
import glob
import json
import sqlite3
from datetime import datetime
from collections import deque
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, Optional

from rich.console import Console, Group
from rich.live import Live
from rich.table import Table
from rich.panel import Panel
from rich.layout import Layout
from rich.text import Text
from rich import box

# ============================================================================
# CONSTANTS
# ============================================================================

DB_PATH = 'data/paper_trading.db'
CONFIG_PATH = 'config.json'
LOG_PATTERN = 'logs/real_bot_simulation*.log'
FALLBACK_LOG_PATTERN = 'logs/*.log'

DEFAULT_INITIAL_BALANCE = 1000.0
DEFAULT_MAX_POSITIONS = 2
DB_REFRESH_SECONDS = 5  # Only for win/loss stats, don't need frequent refresh
LOG_CHECK_SECONDS = 1

console = Console()


# ============================================================================
# DATA CLASSES
# ============================================================================

@dataclass
class MarketData:
    """Current market prices"""
    spy: float = 0.0
    vix: float = 0.0


@dataclass
class TradingMetrics:
    """Core training metrics"""
    cycles: int = 0
    signals: int = 0
    trades: int = 0
    rl_updates: int = 0
    current_positions: int = 0
    max_positions: int = DEFAULT_MAX_POSITIONS
    skipped_hours: int = 0  # Off-market hours skipped
    cooldown_remaining: int = 0  # Cycles remaining in losing streak cooldown
    consecutive_losses: int = 0  # Current consecutive loss count


@dataclass
class AccountState:
    """Training account balance and P&L"""
    initial_balance: float = DEFAULT_INITIAL_BALANCE
    current_balance: float = DEFAULT_INITIAL_BALANCE
    pnl: float = 0.0
    pnl_pct: float = 0.0
    winning_trades: int = 0
    losing_trades: int = 0
    replenish_count: int = 0  # Times balance was auto-replenished
    replenish_total: float = 0.0  # Total $ fed during training


@dataclass
class SignalState:
    """Current signal information"""
    last_signal: str = "Waiting..."
    confidence: float = 0.0
    hmm_regime: str = "Unknown"
    hmm_confidence: float = 0.0
    hmm_trend: str = ""
    hmm_volatility: str = ""
    lstm_predictions: Dict = field(default_factory=dict)


@dataclass
class LiquidityMetrics:
    """Liquidity and execution quality"""
    spread: float = 0.0
    volume: int = 0
    exec_score: float = 0.0


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_style_for_value(value: float, thresholds: tuple, styles: tuple = ("bold green", "yellow", "red")) -> str:
    """Get style based on value thresholds (high, medium)"""
    high, medium = thresholds
    if value >= high:
        return styles[0]
    elif value >= medium:
        return styles[1]
    return styles[2]


def format_pnl(value: float) -> tuple:
    """Format P&L value and return (text, style)"""
    style = "bold green" if value >= 0 else "bold red"
    symbol = "+" if value >= 0 else ""
    return f"{symbol}${value:,.2f}", style


def load_config_balance() -> float:
    """Load initial balance from config"""
    try:
        if Path(CONFIG_PATH).exists():
            with open(CONFIG_PATH, "r") as f:
                cfg = json.load(f)
                return float(cfg.get("trading", {}).get("initial_balance", DEFAULT_INITIAL_BALANCE))
    except Exception:
        pass
    return DEFAULT_INITIAL_BALANCE


# ============================================================================
# MAIN DASHBOARD CLASS
# ============================================================================

class TrainingDashboard:
    """Dashboard for training progress"""
    
    def __init__(self):
        self.start_time = time.time()
        
        # State containers
        self.market = MarketData()
        self.metrics = TradingMetrics()
        self.account = AccountState(initial_balance=load_config_balance())
        self.account.current_balance = self.account.initial_balance
        self.signal = SignalState()
        self.liquidity = LiquidityMetrics()
        
        # Recent trades (max 5)
        self.recent_trades = deque(maxlen=5)
        
        # Archive of previous training sessions
        self.archived_sessions = []  # List of {run_id, trades, final_balance, pnl, wins, losses}
        self.current_run_id = None
        
        # Open positions tracking (unrealized P&L)
        self.unrealized_pnl = 0.0
        self.open_positions_invested = 0.0
        self.open_positions_value = 0.0
        
        # Session info
        self.simulated_date = "Not started"
        self.simulated_time = ""
        self.phase = "Training"  # "Training" or "Live Paper"
        
        # File tracking
        self.log_file = None
        self.log_file_path = None
        self.last_file_position = 0
        self.last_file_check = 0
        self.last_db_check = 0
    
    # ========================================================================
    # LOG FILE MANAGEMENT
    # ========================================================================
    
    def find_latest_log(self) -> Optional[str]:
        """Find the most recent log file"""
        log_files = glob.glob(LOG_PATTERN)
        if not log_files:
            log_files = glob.glob(FALLBACK_LOG_PATTERN)
        if not log_files:
            return None
        return max(log_files, key=lambda f: Path(f).stat().st_mtime)
    
    def read_new_lines(self):
        """Read new lines from log file"""
        current_time = time.time()
        
        # Check for new log file periodically
        if current_time - self.last_file_check > LOG_CHECK_SECONDS:
            self.last_file_check = current_time
            latest_log = self.find_latest_log()
            
            if latest_log and latest_log != self.log_file_path:
                if self.log_file:
                    self.log_file.close()
                self.log_file = None
                self.log_file_path = latest_log
                self.last_file_position = 0
        
        # Open log file if needed
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
        
        # Read and parse new lines
        try:
            self.log_file.seek(self.last_file_position)
            for line in self.log_file.readlines():
                self._parse_log_line(line)
            self.last_file_position = self.log_file.tell()
            
            # Refresh from database periodically
            if current_time - self.last_db_check > DB_REFRESH_SECONDS:
                self._load_from_database()
                self.last_db_check = current_time
        except Exception:
            pass
    
    # ========================================================================
    # LOG PARSING
    # ========================================================================
    
    def _parse_log_line(self, line: str):
        """Parse a single log line and update state"""
        try:
            self._parse_session_info(line)
            self._parse_prices(line)
            self._parse_signals(line)
            self._parse_metrics(line)
            self._parse_trades(line)
            self._parse_hmm_lstm(line)
            self._parse_liquidity(line)
        except Exception:
            pass
    
    def _parse_session_info(self, line: str):
        """Parse session and timestamp info"""
        # Detect new training run - multiple triggers
        new_run_detected = False
        new_run_id = None
        
        # Trigger 1: Full reset message
        if "FULL RESET" in line or "[RESET] Clearing all old data" in line:
            new_run_detected = True
        
        # Trigger 2: New model run directory (most reliable)
        # "[NN] Using timestamped model directory: models/run_20251126_211528"
        if "[NN] Using timestamped model directory:" in line or "MODEL_RUN_DIR" in line:
            match = re.search(r'run_(\d{8}_\d{6})', line)
            if match:
                new_run_id = match.group(1)
                if new_run_id != self.current_run_id:
                    new_run_detected = True
        
        # Trigger 3: TIME-TRAVEL TRAINING header at start
        if "TIME-TRAVEL TRAINING V2:" in line:
            new_run_detected = True
        
        # Archive old session and reset if new run detected
        if new_run_detected:
            self._archive_and_reset_session(new_run_id)
        
        # Cycle header: "[CYCLE 55] | 2025-09-29 17:31:00"
        if "[CYCLE" in line and "|" in line:
            match = re.search(r'\[CYCLE\s+(\d+)\]\s+\|\s+(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2})', line)
            if match:
                self.metrics.cycles = int(match.group(1))
                ts = match.group(2).split()
                self.simulated_date = ts[0]
                self.simulated_time = ts[1]
                self.signal.lstm_predictions = {}  # Clear stale predictions
        
        # Phase detection
        if "PHASE 2" in line or "LIVE SHADOW TRADING" in line:
            self.phase = "Live Paper"
        elif "TIME-TRAVEL TRAINING" in line or "PHASE 1" in line:
            self.phase = "Training"
        
        # Auto-replenish tracking: "💰 [TRAINING] Auto-replenished $1000.00 to continue learning"
        if "[TRAINING] Auto-replenished" in line:
            self.account.replenish_count += 1
            match = re.search(r'Auto-replenished\s*\$?([\d,]+\.?\d*)', line)
            if match:
                amount = float(match.group(1).replace(',', ''))
                self.account.replenish_total += amount
    
    def _parse_prices(self, line: str):
        """Parse market prices"""
        # SPY
        if "SPY:" in line or "SPY Price:" in line:
            match = re.search(r'SPY(?:\s+Price)?:\s*\$([\d.]+)', line)
            if match:
                self.market.spy = float(match.group(1))
        
        # VIX
        if ("VIX:" in line or "📊 VIX:" in line) and "data" not in line.lower():
            match = re.search(r'VIX:\s*\$?([\d.]+)', line)
            if match:
                self.market.vix = float(match.group(1))
    
    def _parse_signals(self, line: str):
        """Parse signal information"""
        if "FINAL Signal:" in line:
            if "BUY_CALLS" in line:
                self.signal.last_signal = "🟢 BUY CALLS"
            elif "BUY_PUTS" in line:
                self.signal.last_signal = "🔴 BUY PUTS"
            elif "HOLD" in line:
                self.signal.last_signal = "⚪ HOLD"
            
            # Parse confidence
            match = re.search(r'confidence:\s*([\d.]+)%?\)', line)
            if match:
                conf = float(match.group(1))
                self.signal.confidence = conf * 100 if conf < 1.0 else conf
    
    def _parse_metrics(self, line: str):
        """Parse training metrics and balance"""
        # Cycle summary: "- Completed cycles: 50"
        if "Completed cycles:" in line:
            match = re.search(r'Completed cycles:\s*([\d,]+)', line)
            if match:
                self.metrics.cycles = int(match.group(1).replace(',', ''))
        
        if "Signals generated:" in line:
            match = re.search(r'Signals generated:\s*([\d,]+)', line)
            if match:
                self.metrics.signals = int(match.group(1).replace(',', ''))
        
        if "Trades made:" in line:
            match = re.search(r'Trades made:\s*([\d,]+)', line)
            if match:
                self.metrics.trades = int(match.group(1).replace(',', ''))
        
        if "RL updates:" in line:
            match = re.search(r'RL updates:\s*([\d,]+)', line)
            if match:
                self.metrics.rl_updates = int(match.group(1).replace(',', ''))
        
        # Balance - multiple formats with flexible regex
        # Format 1: "Current balance: $1,000.00" or "- Current balance: $1,000.00"
        if "balance:" in line.lower() and "$" in line:
            # Flexible match: any dollar amount with optional commas and decimals
            match = re.search(r'\$([\d,]+\.?\d*)', line)
            if match:
                balance_str = match.group(1).replace(',', '')
                try:
                    new_balance = float(balance_str)
                    if new_balance > 0:  # Sanity check
                        self.account.current_balance = new_balance
                        self._update_pnl()
                except ValueError:
                    pass
        
        # Positions
        if "Active:" in line or "Active Positions:" in line:
            match = re.search(r'Active(?:\s+Positions)?:\s*(\d+)/(\d+)', line)
            if match:
                self.metrics.current_positions = int(match.group(1))
                self.metrics.max_positions = int(match.group(2))
        
        # Cooldown tracking (losing streak protection)
        if "[COOLDOWN]" in line:
            # Parse "Skipping trade - X cycles remaining in cooldown"
            match = re.search(r'(\d+)\s*cycles?\s*remaining', line)
            if match:
                self.metrics.cooldown_remaining = int(match.group(1))
            # Parse "Losing streak of X detected"
            match = re.search(r'Losing streak of (\d+)', line)
            if match:
                self.metrics.consecutive_losses = int(match.group(1))
        
        # Consecutive loss count: "📉 Consecutive losses: X/Y"
        if "Consecutive losses:" in line:
            match = re.search(r'Consecutive losses:\s*(\d+)/(\d+)', line)
            if match:
                self.metrics.consecutive_losses = int(match.group(1))
        
        # Feature buffer
        if "Feature Buffer:" in line or "Feature buffer:" in line:
            match = re.search(r'Feature [Bb]uffer:\s*(\d+)/(\d+)', line)
            if match:
                pass  # Could track this if needed
    
    def _parse_trades(self, line: str):
        """Parse trade entries and P&L"""
        # Trade placed
        if "TRADE PLACED:" in line:
            match = re.search(r'TRADE PLACED:\s*(\w+)\s+(?:@|at)\s+\$?([\d.]+)', line)
            if match:
                self.recent_trades.append({
                    'time': self.simulated_time or datetime.now().strftime('%H:%M:%S'),
                    'action': match.group(1),
                    'price': f"${match.group(2)}",
                    'pnl': 'OPEN'
                })
        
        # Trade closed P&L - match multiple formats:
        # Format 1: "  • Net P&L: $12.34"
        # Format 2: "[CLOSE] Trade closed: P&L=$12.34, Reason: ..."
        pnl_match = None
        if "Net P&L:" in line:
            pnl_match = re.search(r'Net P&L:\s*\$?([+-]?[\d.]+)', line)
        elif "[CLOSE] Trade closed:" in line:
            pnl_match = re.search(r'P&L=\$?([+-]?[\d.]+)', line)
        
        if pnl_match:
            pnl = float(pnl_match.group(1))
            # Update most recent trade that's still OPEN
            for trade in reversed(self.recent_trades):
                if trade.get('pnl') == 'OPEN':
                    trade['pnl'] = f"${pnl:+.2f}"
                    # NOTE: Don't increment win/loss here - database is source of truth
                    # This prevents jumpy numbers from log+database conflicts
                    break
    
    def _parse_hmm_lstm(self, line: str):
        """Parse HMM regime and LSTM predictions"""
        # HMM Regime
        if "[HMM] Regime:" in line or "[HMM] Market Regime:" in line:
            match = re.search(r'(?:Regime|Market Regime):\s*([^,\(]+)', line)
            if match:
                self.signal.hmm_regime = match.group(1).strip()
            
            conf_match = re.search(r'[Cc]onfidence:\s*([\d.]+)%', line)
            if conf_match:
                self.signal.hmm_confidence = float(conf_match.group(1))
        
        # LSTM predictions
        if "[LSTM]" in line and ("direction=" in line or ":" in line):
            # Format: [LSTM] 15min: direction=UP, return=+1.5%, confidence=25%
            match = re.search(
                r'\[LSTM\]\s+([^:]+):\s+direction=(\w+),\s+return=([+-]?[\d.]+)%,\s+confidence=([\d.]+)%',
                line
            )
            if match:
                self.signal.lstm_predictions[match.group(1).strip()] = {
                    'direction': match.group(2),
                    'return': float(match.group(3)),
                    'confidence': float(match.group(4))
                }
    
    def _parse_liquidity(self, line: str):
        """Parse liquidity metrics"""
        if "Spread OK:" in line or "Spread=" in line:
            match = re.search(r'Spread[=\s:OK]*\s*([\d.]+)%', line)
            if match:
                self.liquidity.spread = float(match.group(1))
        
        if "Volume" in line and "[LIQUIDITY]" in line:
            match = re.search(r'Volume[=\s:OK]*\s*(\d+)', line)
            if match:
                self.liquidity.volume = int(match.group(1))
    
    def _archive_and_reset_session(self, new_run_id: Optional[str] = None):
        """Archive current session data and reset for new training run"""
        # Only archive if we have meaningful data (at least some trades or cycles)
        if self.metrics.trades > 0 or self.metrics.cycles > 10:
            archived = {
                'run_id': self.current_run_id or 'unknown',
                'trades': list(self.recent_trades),
                'total_trades': self.metrics.trades,
                'final_balance': self.account.current_balance,
                'pnl': self.account.pnl,
                'pnl_pct': self.account.pnl_pct,
                'wins': self.account.winning_trades,
                'losses': self.account.losing_trades,
                'cycles': self.metrics.cycles,
                'archived_at': datetime.now().strftime('%H:%M:%S')
            }
            self.archived_sessions.append(archived)
            # Keep only last 5 archived sessions
            if len(self.archived_sessions) > 5:
                self.archived_sessions.pop(0)
        
        # Update current run ID
        self.current_run_id = new_run_id
        
        # Reset all metrics for new session
        self.metrics = TradingMetrics()
        self.account = AccountState(initial_balance=load_config_balance())
        self.account.current_balance = self.account.initial_balance
        self.signal = SignalState()
        self.recent_trades.clear()
    
    def _reset_session(self):
        """Reset all metrics for new session (backward compatible)"""
        self._archive_and_reset_session(None)
    
    def _update_pnl(self):
        """Recalculate P&L from balance"""
        self.account.pnl = self.account.current_balance - self.account.initial_balance
        if self.account.initial_balance > 0:
            self.account.pnl_pct = (self.account.pnl / self.account.initial_balance) * 100
    
    # ========================================================================
    # DATABASE LOADING
    # ========================================================================
    
    def _load_from_database(self):
        """Load account stats from database as fallback/supplement to log parsing.
        
        Win/loss counts: Always loaded (database is authoritative)
        Balance/P&L: Loaded as fallback if not set from logs
        Open positions: Load to calculate unrealized P&L
        """
        try:
            conn = sqlite3.connect(DB_PATH)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT balance, winning_trades, losing_trades, total_profit_loss 
                FROM account_state ORDER BY updated_at DESC LIMIT 1
            ''')
            row = cursor.fetchone()
            if row:
                # Win/loss counts - database is authoritative
                self.account.winning_trades = row[1] or 0
                self.account.losing_trades = row[2] or 0
                
                # Balance - only update if we haven't seen any from logs recently
                # (indicated by balance still at initial value)
                db_balance = row[0] or 0
                if db_balance > 0 and abs(self.account.current_balance - self.account.initial_balance) < 0.01:
                    self.account.current_balance = db_balance
                    self.account.pnl = row[3] or 0
                    self._update_pnl()
            
            # Get open positions to calculate unrealized P&L
            cursor.execute('''
                SELECT option_type, premium_paid, current_price, quantity
                FROM trades 
                WHERE status IN ('FILLED', 'OPEN')
            ''')
            open_trades = cursor.fetchall()
            
            total_invested = 0.0
            total_current_value = 0.0
            for trade in open_trades:
                option_type, premium_paid, current_price, quantity = trade
                if premium_paid and quantity:
                    total_invested += premium_paid * quantity
                    if current_price:
                        total_current_value += current_price * quantity
            
            # Store unrealized P&L for open positions
            if total_invested > 0:
                self.open_positions_invested = total_invested
                self.open_positions_value = total_current_value
                self.unrealized_pnl = total_current_value - total_invested
            else:
                self.open_positions_invested = 0
                self.open_positions_value = 0
                self.unrealized_pnl = 0
            
            conn.close()
        except Exception:
            self.unrealized_pnl = 0
            self.open_positions_invested = 0
            self.open_positions_value = 0
    
    # ========================================================================
    # LAYOUT GENERATION
    # ========================================================================
    
    def generate_layout(self) -> Layout:
        """Generate the dashboard layout"""
        layout = Layout()
        
        layout.split(
            Layout(name="header", size=4),
            Layout(name="body", minimum_size=30),
            Layout(name="footer", size=3)
        )
        
        layout["body"].split_row(
            Layout(name="left", minimum_size=45),
            Layout(name="right", minimum_size=45)
        )
        
        layout["left"].split(
            Layout(name="metrics", size=16),
            Layout(name="account", minimum_size=12)
        )
        
        layout["right"].split(
            Layout(name="signals", minimum_size=18),
            Layout(name="trades", minimum_size=12)
        )
        
        layout["header"].update(self._build_header())
        layout["metrics"].update(self._build_metrics())
        layout["account"].update(self._build_account())
        layout["signals"].update(self._build_signals())
        layout["trades"].update(self._build_trades())
        layout["footer"].update(self._build_footer())
        
        return layout
    
    def _build_header(self) -> Panel:
        """Build header panel"""
        elapsed = time.time() - self.start_time
        hours, remainder = divmod(int(elapsed), 3600)
        minutes, seconds = divmod(remainder, 60)
        
        text = Text()
        text.append("🎓 TRAINING DASHBOARD ", style="bold cyan")
        text.append(f" | Running: {hours:02d}:{minutes:02d}:{seconds:02d}", style="bold white")
        
        text.append("\n📅 Simulated Time: ", style="bold magenta")
        if self.simulated_date != "Not started" and self.simulated_time:
            text.append(f"{self.simulated_date} {self.simulated_time}", style="bold green")
        else:
            text.append("Waiting for first timestamp...", style="dim yellow")
        
        # Phase indicator
        phase_style = "bold yellow" if self.phase == "Training" else "bold green"
        phase_emoji = "📚" if self.phase == "Training" else "📝"
        text.append(f" | {phase_emoji} Phase: {self.phase}", style=phase_style)
        
        return Panel(text, border_style="cyan")
    
    def _build_metrics(self) -> Panel:
        """Build metrics panel"""
        table = Table(box=box.SIMPLE, show_header=False, padding=(0, 1), expand=False)
        table.add_column("Metric", style="cyan", width=18, no_wrap=True)
        table.add_column("Value", style="bold white", width=12, no_wrap=True)
        
        table.add_row("🔄 Cycles", f"{self.metrics.cycles:,}")
        table.add_row("📊 Signals", f"{self.metrics.signals:,}")
        table.add_row("💼 Trades", f"{self.metrics.trades:,}")
        
        # Signal rate
        signal_rate = (self.metrics.signals / max(self.metrics.cycles, 1)) * 100
        table.add_row("📈 Signal Rate", f"{signal_rate:.1f}%")
        
        # Win rate
        total_closed = self.account.winning_trades + self.account.losing_trades
        win_rate = (self.account.winning_trades / max(total_closed, 1)) * 100
        win_style = get_style_for_value(win_rate, (60, 40))
        table.add_row("🏆 Win Rate", f"{win_rate:.1f}%", style=win_style)
        
        table.add_row("🧠 RL Updates", f"{self.metrics.rl_updates:,}")
        
        pos_style = "bold red" if self.metrics.current_positions >= self.metrics.max_positions else "bold white"
        table.add_row("📍 Positions", f"{self.metrics.current_positions}/{self.metrics.max_positions}", style=pos_style)
        
        # Cooldown indicator (losing streak protection)
        if self.metrics.cooldown_remaining > 0:
            table.add_row("⏸️ COOLDOWN", f"{self.metrics.cooldown_remaining} cycles", style="bold yellow")
        elif self.metrics.consecutive_losses >= 3:
            table.add_row("⚠️ Losses", f"{self.metrics.consecutive_losses} in a row", style="bold red")
        
        # Liquidity metrics
        if self.liquidity.spread > 0:
            table.add_row("↔️ Spread", f"{self.liquidity.spread:.1f}%")
        
        if self.liquidity.volume > 0:
            table.add_row("📊 Volume", f"{self.liquidity.volume:,}")
        
        return Panel(table, title="[bold]📊 Training Metrics", border_style="blue")
    
    def _build_account(self) -> Panel:
        """Build account panel"""
        text = Text()
        
        # Balance
        text.append("💰 Balance: ", style="bold cyan")
        text.append(f"${self.account.current_balance:,.2f}\n", style="bold white")
        
        # P&L
        pnl_text, pnl_style = format_pnl(self.account.pnl)
        text.append("📈 P&L: ", style="bold cyan")
        text.append(f"{pnl_text} ({self.account.pnl_pct:+.1f}%)\n", style=pnl_style)
        
        # Unrealized P&L for open positions (explains the disconnect!)
        if self.metrics.current_positions > 0 and self.open_positions_invested > 0:
            unrealized_text, unrealized_style = format_pnl(self.unrealized_pnl)
            text.append("\n📊 Open Positions: ", style="bold yellow")
            text.append(f"{self.metrics.current_positions} open\n", style="white")
            text.append("   Invested: ", style="dim white")
            text.append(f"${self.open_positions_invested:,.2f}\n", style="white")
            text.append("   Unrealized: ", style="dim white")
            text.append(f"{unrealized_text}\n", style=unrealized_style)
        
        # Win/Loss breakdown (closed trades only)
        text.append("\n🏆 Closed Wins: ", style="bold green")
        text.append(f"{self.account.winning_trades}", style="bold white")
        text.append("  ❌ Losses: ", style="bold red")
        text.append(f"{self.account.losing_trades}\n", style="bold white")
        
        # Auto-replenish stats (if any)
        if self.account.replenish_count > 0:
            text.append(f"\n💸 Fed during training: ", style="dim yellow")
            text.append(f"${self.account.replenish_total:,.2f}", style="bold yellow")
            text.append(f" ({self.account.replenish_count}x)", style="dim white")
            
            # True P&L = current balance - initial - total fed
            # This shows what would have happened without bailouts
            true_pnl = self.account.current_balance - self.account.initial_balance - self.account.replenish_total
            true_pnl_text, true_pnl_style = format_pnl(true_pnl)
            text.append(f"\n📉 True P&L (no bailouts): ", style="dim white")
            text.append(f"{true_pnl_text}", style=true_pnl_style)
        
        return Panel(text, title="[bold]💵 Training Account", border_style="green")
    
    def _build_signals(self) -> Panel:
        """Build signals panel"""
        text = Text()
        
        # Market data
        if self.market.spy > 0:
            text.append("📈 SPY: ", style="bold cyan")
            text.append(f"${self.market.spy:.2f}\n", style="bold white")
        if self.market.vix > 0:
            vix_style = "bold red" if self.market.vix > 25 else "bold yellow" if self.market.vix > 15 else "bold green"
            text.append("📉 VIX: ", style="bold cyan")
            text.append(f"{self.market.vix:.2f}\n", style=vix_style)
        
        # HMM Regime
        if self.signal.hmm_regime != "Unknown":
            text.append("\n🎰 Regime: ", style="bold magenta")
            text.append(f"{self.signal.hmm_regime}\n", style="cyan")
            text.append(f"   Confidence: {self.signal.hmm_confidence:.1f}%\n", style="dim white")
        
        # LSTM predictions (show top 3)
        if self.signal.lstm_predictions:
            text.append("\n🧠 TCN:\n", style="bold magenta")
            for i, (tf, pred) in enumerate(sorted(self.signal.lstm_predictions.items())[:3]):
                dir_style = "bold green" if pred['direction'] == "UP" else "bold red"
                text.append(f"  {tf}: ", style="cyan")
                text.append(f"{pred['direction']} ", style=dir_style)
                text.append(f"{pred['return']:+.1f}%\n", style="white")
        
        # Current signal
        text.append("\n📡 Signal: ", style="bold white")
        text.append(f"{self.signal.last_signal}\n", style="yellow")
        text.append("   Confidence: ", style="bold white")
        style = get_style_for_value(self.signal.confidence, (60, 40))
        text.append(f"{self.signal.confidence:.1f}%", style=style)
        
        return Panel(text, title="[bold]📡 Market & Signals", border_style="yellow")
    
    def _build_trades(self) -> Panel:
        """Build trades panel"""
        table = Table(box=box.SIMPLE, show_header=True, padding=(0, 1), expand=False)
        table.add_column("Time", style="cyan", width=8, no_wrap=True)
        table.add_column("Type", style="bold yellow", width=10, no_wrap=True)
        table.add_column("Entry", style="green", width=8, no_wrap=True)
        table.add_column("P&L", style="white", width=10, no_wrap=True)
        
        if self.recent_trades:
            for trade in reversed(self.recent_trades):
                action = trade['action']
                action_style = "bold green" if "CALL" in action else "bold red" if "PUT" in action else "white"
                
                pnl = trade.get('pnl', '')
                if pnl == 'OPEN':
                    pnl_text = Text("⏳ OPEN", style="cyan")
                elif '+' in str(pnl):
                    pnl_text = Text(pnl, style="bold green")
                elif '-' in str(pnl):
                    pnl_text = Text(pnl, style="bold red")
                else:
                    pnl_text = Text(pnl, style="white")
                
                table.add_row(
                    trade['time'],
                    Text(action, style=action_style),
                    trade['price'],
                    pnl_text
                )
        else:
            table.add_row("-", "No trades", "-", "-")
        
        return Panel(table, title="[bold]🎯 Recent Trades", border_style="magenta")
    
    def _build_footer(self) -> Panel:
        """Build footer panel"""
        text = Text()
        
        # Current run info
        if self.current_run_id:
            text.append("🏃 Run: ", style="white")
            text.append(f"{self.current_run_id}", style="bold cyan")
            text.append(" | ", style="white")
        
        # Archived sessions summary
        if self.archived_sessions:
            text.append(f"📦 {len(self.archived_sessions)} archived", style="dim yellow")
            # Show last archived session P&L
            last = self.archived_sessions[-1]
            pnl_style = "green" if last['pnl'] >= 0 else "red"
            text.append(f" (last: ${last['pnl']:+.0f})", style=pnl_style)
            text.append(" | ", style="white")
        
        # Log file
        text.append("📁 ", style="white")
        log_name = Path(self.log_file.name).name if self.log_file else "Searching..."
        text.append(f"{log_name}", style="cyan")
        text.append(" | ", style="white")
        text.append("Ctrl+C", style="bold red")
        text.append(" to stop", style="white")
        return Panel(text, border_style="cyan")


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Run the training dashboard"""
    console.clear()
    console.print()
    console.print("[bold cyan]🎓 Training Dashboard[/bold cyan]")
    console.print("[dim]Watching training log for progress...[/dim]")
    console.print()
    
    # Check for log files
    log_files = glob.glob(LOG_PATTERN) or glob.glob(FALLBACK_LOG_PATTERN)
    
    if not log_files:
        console.print("[bold red]❌ No log file found![/bold red]")
        console.print()
        console.print("[yellow]Please start training first:[/yellow]")
        console.print("[bold cyan]python train_then_go_live.py[/bold cyan]")
        console.print()
        console.print("[dim]This dashboard will automatically find the log file.[/dim]")
        return
    
    console.print("[bold green]✅ Found log file! Starting dashboard...[/bold green]")
    console.print()
    time.sleep(1)
    
    dashboard = TrainingDashboard()
    
    try:
        with Live(dashboard.generate_layout(), refresh_per_second=2, screen=True, console=console) as live:
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
