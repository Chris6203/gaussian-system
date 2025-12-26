#!/usr/bin/env python3
"""
Log Parser
==========

Parses training log files to extract state information.
Extracted from training_dashboard_server.py for modularity.
"""

import re
import glob
import logging
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
import os

from backend.dashboard.state import (
    TrainingState, LogTrade, LSTMPrediction, ValidatedPrediction
)

logger = logging.getLogger(__name__)

# Market timezone for consistent timestamp handling
try:
    import pytz
    MARKET_TZ = pytz.timezone('US/Eastern')
    HAS_PYTZ = True
except ImportError:
    MARKET_TZ = None
    HAS_PYTZ = False

def get_market_time() -> datetime:
    """Get current time in market timezone (US/Eastern) as naive datetime."""
    if HAS_PYTZ:
        market_dt = pytz.UTC.localize(datetime.utcnow()).astimezone(MARKET_TZ)
        return market_dt.replace(tzinfo=None)
    else:
        # Fallback: assume UTC-5 for Eastern (doesn't account for DST)
        return datetime.utcnow() - timedelta(hours=5)

def local_to_market_time(dt: datetime) -> datetime:
    """Convert a datetime to market time (Eastern)."""
    if HAS_PYTZ:
        if dt.tzinfo is None:
            # Assume local time, localize and convert
            local_tz = datetime.now().astimezone().tzinfo
            local_dt = dt.replace(tzinfo=local_tz)
            market_dt = local_dt.astimezone(MARKET_TZ)
            return market_dt.replace(tzinfo=None)
        else:
            # Already has timezone, just convert
            return dt.astimezone(MARKET_TZ).replace(tzinfo=None)
    else:
        # Fallback: no conversion (assume already in market time)
        return dt

def local_timestamp_to_market(timestamp_str: str) -> str:
    """Convert a local timestamp string to market time string."""
    try:
        dt = datetime.fromisoformat(timestamp_str.replace(' ', 'T'))
        et_dt = local_to_market_time(dt)
        return et_dt.strftime('%Y-%m-%dT%H:%M:%S')
    except Exception:
        return timestamp_str


class LogParser:
    """
    Parses training log files and updates TrainingState.
    
    Handles incremental parsing for efficiency.
    """
    
    def __init__(
        self,
        log_pattern: str = 'logs/real_bot_simulation*.log',
        fallback_pattern: str = '../logs/real_bot_simulation*.log',
        initial_lines: int = 5000
    ):
        """
        Initialize log parser.
        
        Args:
            log_pattern: Primary glob pattern for log files
            fallback_pattern: Fallback pattern if primary finds nothing
            initial_lines: Number of lines to parse on initial load
        """
        self.log_pattern = log_pattern
        self.fallback_pattern = fallback_pattern
        self.initial_lines = initial_lines
        
        self._last_file_position = 0
        self._current_log_path: Optional[str] = None
    
    def find_latest_log(self) -> Optional[str]:
        """Find the most recent log file.

        Priority order:
        1. Primary pattern with content (>0 bytes)
        2. Environment variable override
        3. Fallback pattern with content
        4. Models directory logs (for training dashboard)
        5. Any matching file (even empty)
        """
        def get_valid_files(pattern: str, require_content: bool = True) -> list:
            """Get files matching pattern, optionally requiring content."""
            result = []
            for p in glob.glob(pattern):
                try:
                    path = Path(p)
                    if not path.exists():
                        continue
                    if require_content and path.stat().st_size == 0:
                        continue
                    result.append(p)
                except Exception:
                    continue
            return result

        # 1) Primary pattern - prefer files with content
        primary_files = get_valid_files(self.log_pattern, require_content=True)
        if primary_files:
            return max(primary_files, key=lambda f: Path(f).stat().st_mtime)

        # 2) Explicit override (useful if training logs are redirected elsewhere)
        env_log = os.environ.get("TRAINING_LOG_FILE")
        if env_log:
            path = Path(env_log)
            if path.exists() and path.stat().st_size > 0:
                return env_log

        # 3) Fallback pattern - for live dashboard, this catches real_bot_simulation*.log
        fallback_files = get_valid_files(self.fallback_pattern, require_content=True)
        if fallback_files:
            return max(fallback_files, key=lambda f: Path(f).stat().st_mtime)

        # 4) Models directory logs (for training dashboard)
        model_patterns = [
            "models/*/run*.log",
            "models/*/*run*.log"
        ]
        for pattern in model_patterns:
            model_files = get_valid_files(pattern, require_content=True)
            if model_files:
                return max(model_files, key=lambda f: Path(f).stat().st_mtime)

        # 5) Last resort - recursive search under models/
        deep_files = get_valid_files("models/**/run*.log", require_content=True)
        if deep_files:
            return max(deep_files, key=lambda f: Path(f).stat().st_mtime)

        # 6) If nothing has content, try primary pattern even if empty
        primary_any = get_valid_files(self.log_pattern, require_content=False)
        if primary_any:
            return max(primary_any, key=lambda f: Path(f).stat().st_mtime)

        return None
    
    def parse_log_file(self, log_path: str, state: TrainingState) -> None:
        """
        Parse log file and update state.
        
        Args:
            log_path: Path to log file
            state: TrainingState to update
        """
        try:
            # Reset position if log file changed
            if log_path != self._current_log_path:
                self._last_file_position = 0
                self._current_log_path = log_path
            
            # Auto-detect encoding (PowerShell redirection often produces UTF-16 LE).
            enc = "utf-8"
            try:
                with open(log_path, "rb") as bf:
                    head = bf.read(4096)
                # BOM checks
                if head.startswith(b"\xff\xfe") or head.startswith(b"\xfe\xff"):
                    enc = "utf-16"
                elif b"\x00" in head:
                    # Heuristic: lots of NULs usually means UTF-16
                    enc = "utf-16"
            except Exception:
                enc = "utf-8"

            with open(log_path, 'r', encoding=enc, errors='ignore') as f:
                if self._last_file_position == 0:
                    # Initial load - parse last N lines
                    lines = f.readlines()
                    for line in lines[-self.initial_lines:]:
                        self._parse_line(line, state)
                    self._last_file_position = f.tell()
                else:
                    # Incremental read
                    f.seek(self._last_file_position)
                    for line in f.readlines():
                        self._parse_line(line, state)
                    self._last_file_position = f.tell()
            
            state.session.log_file = Path(log_path).name
            
        except Exception as e:
            logger.error(f"Error parsing log: {e}")
    
    def _parse_line(self, line: str, state: TrainingState) -> None:
        """Parse a single log line."""
        try:
            self._parse_dashboard_update(line, state)
            self._parse_session_info(line, state)
            self._parse_prices(line, state)
            self._parse_signals(line, state)
            self._parse_metrics(line, state)
            self._parse_trades_from_log(line, state)
            self._parse_hmm_lstm(line, state)
            self._parse_liquidity(line, state)
            self._parse_rl_stats(line, state)
        except Exception:
            pass

    def _parse_dashboard_update(self, line: str, state: TrainingState) -> None:
        """
        Parse a compact dashboard update line emitted by training scripts.
        Format:
          [DASH] cycle=123 signals=45 trades=6 rl_updates=7 balance=4999.12 spy=685.50 vix=17.18 qqq=619.53 btc=88268.54
        """
        if "[DASH]" not in line:
            return
        try:
            # key=value pairs
            kv = dict(re.findall(r'(\w+)=([-\d.]+)', line))

            if "cycle" in kv:
                state.progress.cycles = int(float(kv["cycle"]))
            if "signals" in kv:
                state.progress.signals = int(float(kv["signals"]))
            if "trades" in kv:
                state.progress.trades = int(float(kv["trades"]))
            if "rl_updates" in kv:
                state.progress.rl_updates = int(float(kv["rl_updates"]))

            if "balance" in kv:
                bal = float(kv["balance"])
                if bal > 0:
                    state.account.current_balance = bal
                    state.account.update_pnl()

            if "spy" in kv:
                state.market.spy_price = float(kv["spy"])
            if "vix" in kv:
                v = float(kv["vix"])
                if v < 100:
                    state.market.vix_value = v
            if "qqq" in kv:
                state.market.qqq_price = float(kv["qqq"])
            if "btc" in kv:
                state.market.btc_price = float(kv["btc"])
        except Exception:
            return
    
    def _parse_session_info(self, line: str, state: TrainingState) -> None:
        """Parse session and timestamp info."""
        # Detect new training run
        new_run_detected = False
        new_run_id = None
        
        if "FULL RESET" in line or "[RESET] Clearing all old data" in line:
            new_run_detected = True
        
        if "[NN] Using timestamped model directory:" in line or "MODEL_RUN_DIR" in line:
            match = re.search(r'run_(\d{8}_\d{6})', line)
            if match:
                new_run_id = match.group(1)
                if new_run_id != state.session.current_run_id:
                    new_run_detected = True
        
        if "TIME-TRAVEL TRAINING V2:" in line:
            new_run_detected = True
        
        if new_run_detected:
            state.reset_for_new_session(new_run_id, state.account.initial_balance)
        
        # Cycle header: "[CYCLE 55] | 2025-09-29 17:31:00"
        # NOTE: These timestamps are ALREADY in market time (ET) from historical data
        # Do NOT convert them - they're already correct!
        if "[CYCLE" in line and "|" in line:
            match = re.search(r'\[CYCLE\s+(\d+)\]\s+\|\s+(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2})', line)
            if match:
                state.progress.cycles = int(match.group(1))
                ts = match.group(2).split()
                # Timestamps from training are already in ET (from historical data)
                state.session.simulated_date = ts[0]
                state.session.simulated_time = ts[1]
        
        # Cycle count: "- Completed cycles: 2657"
        if "Completed cycles:" in line:
            match = re.search(r'Completed cycles:\s*(\d+)', line)
            if match:
                state.progress.cycles = int(match.group(1))
        
        # Simulated time: "[TIME] Processing timestamp: 2025-08-19 14:09:00"
        # NOTE: These timestamps are ALREADY in market time (ET) from historical data
        if "[TIME] Processing timestamp:" in line:
            match = re.search(r'\[TIME\] Processing timestamp:\s+(\d{4}-\d{2}-\d{2})\s+(\d{2}:\d{2}:\d{2})', line)
            if match:
                # Timestamps from training are already in ET (from historical data)
                state.session.simulated_date = match.group(1)
                state.session.simulated_time = match.group(2)
        
        # Phase detection
        if "PHASE 2" in line or "LIVE SHADOW TRADING" in line:
            state.session.phase = "Live Paper"
        elif "TIME-TRAVEL TRAINING" in line or "PHASE 1" in line:
            state.session.phase = "Training"
        
        # Auto-replenish tracking
        if "[TRAINING] Auto-replenished" in line:
            state.account.replenish_count += 1
            match = re.search(r'Auto-replenished\s*\$?([\d,]+\.?\d*)', line)
            if match:
                amount = float(match.group(1).replace(',', ''))
                state.account.replenish_total += amount
    
    def _parse_prices(self, line: str, state: TrainingState) -> None:
        """Parse market prices."""
        # SPY
        if "SPY" in line and "$" in line:
            match = re.search(r'(?:\[SPY\]|\[PRICE\]\s*SPY:|SPY(?:\s+Price)?:)\s*\$([\d.]+)', line)
            if match:
                state.market.spy_price = float(match.group(1))
        
        # VIX
        if "VIX" in line and "data" not in line.lower():
            match = re.search(r'(?:\[VIX\]|VIX:)\s*\$?([\d.]+)', line)
            if match:
                val = float(match.group(1))
                if val < 100:
                    state.market.vix_value = val
        
        # QQQ
        if "QQQ" in line and "$" in line:
            match = re.search(r'(?:\[QQQ\]|QQQ(?:\s+Price)?:)\s*\$([\d.]+)', line)
            if match:
                state.market.qqq_price = float(match.group(1))
        
        # BTC-USD
        if "BTC-USD" in line and "$" in line:
            match = re.search(r'(?:\[BTC-USD\]|BTC-USD(?:\s+Price)?:)\s*\$?([\d,]+\.?\d*)', line)
            if match:
                val = match.group(1).replace(',', '')
                if val:
                    state.market.btc_price = float(val)
    
    def _parse_signals(self, line: str, state: TrainingState) -> None:
        """Parse signal information."""
        if "FINAL Signal:" in line:
            if "BUY_CALLS" in line:
                state.signal.last_signal = "BUY_CALLS"
            elif "BUY_PUTS" in line:
                state.signal.last_signal = "BUY_PUTS"
            elif "BUY_STRADDLE" in line:
                state.signal.last_signal = "BUY_STRADDLE"
            elif "HOLD" in line:
                state.signal.last_signal = "HOLD"
            
            match = re.search(r'confidence:\s*([\d.]+)%?\)', line)
            if match:
                conf = float(match.group(1))
                state.signal.signal_confidence = conf * 100 if conf < 1.0 else conf
        
        if "[SIGNAL]" in line:
            if "BUY_CALLS" in line:
                state.signal.last_signal = "BUY_CALLS"
            elif "BUY_PUTS" in line:
                state.signal.last_signal = "BUY_PUTS"
            elif "STRADDLE" in line:
                state.signal.last_signal = "BUY_STRADDLE"
            elif "HOLD" in line:
                state.signal.last_signal = "HOLD"
    
    def _parse_metrics(self, line: str, state: TrainingState) -> None:
        """Parse training metrics and balance."""
        if "Completed cycles:" in line:
            match = re.search(r'Completed cycles:\s*([\d,]+)', line)
            if match:
                state.progress.cycles = int(match.group(1).replace(',', ''))
        
        if "Signals generated:" in line:
            match = re.search(r'Signals generated:\s*([\d,]+)', line)
            if match:
                state.progress.signals = int(match.group(1).replace(',', ''))
        
        if "Trades made:" in line:
            match = re.search(r'Trades made:\s*([\d,]+)', line)
            if match:
                state.progress.trades = int(match.group(1).replace(',', ''))
        
        if "RL updates:" in line:
            match = re.search(r'RL updates:\s*([\d,]+)', line)
            if match:
                state.progress.rl_updates = int(match.group(1).replace(',', ''))
        
        # Balance
        if "balance:" in line.lower() and "$" in line:
            match = re.search(r'\$([\d,]+\.?\d*)', line)
            if match:
                balance_str = match.group(1).replace(',', '')
                try:
                    new_balance = float(balance_str)
                    if new_balance > 0:
                        state.account.current_balance = new_balance
                        state.account.update_pnl()
                except ValueError:
                    pass
        
        # Positions
        if "Active:" in line or "Active Positions:" in line:
            match = re.search(r'Active(?:\s+Positions)?:\s*(\d+)/(\d+)', line)
            if match:
                state.current_positions = int(match.group(1))
                state.max_positions = int(match.group(2))
        
        # Cooldown
        if "[COOLDOWN]" in line:
            match = re.search(r'(\d+)\s*cycles?\s*remaining', line)
            if match:
                state.progress.cooldown_remaining = int(match.group(1))
            match = re.search(r'Losing streak of (\d+)', line)
            if match:
                state.progress.consecutive_losses = int(match.group(1))
        
        if "Consecutive losses:" in line:
            match = re.search(r'Consecutive losses:\s*(\d+)/(\d+)', line)
            if match:
                state.progress.consecutive_losses = int(match.group(1))
    
    def _parse_trades_from_log(self, line: str, state: TrainingState, max_trades: int = 50) -> None:
        """Parse trade entries and exits from log lines."""
        # Parse [TRADE] Closed format
        if "[TRADE] Closed" in line or ("[TRADE]" in line and "Closed" in line):
            # Full format: Closed CALL | $entry→$exit | entry_time→exit_time
            trade_match = re.search(
                r'\[TRADE\]\s*Closed\s+(\w+)\s*\|\s*\$?([\d.]+)→\$?([\d.]+)\s*\|\s*(\d{2}:\d{2}:\d{2})→(\d{2}:\d{2}:\d{2})',
                line
            )
            if trade_match:
                option_type = trade_match.group(1)
                entry_price = float(trade_match.group(2))
                exit_price = float(trade_match.group(3))
                entry_time_str = trade_match.group(4)
                exit_time_str = trade_match.group(5)
                
                # Build timestamps
                entry_timestamp = None
                exit_timestamp = None
                if state.session.simulated_date and state.session.simulated_date != "Not started":
                    entry_timestamp = f"{state.session.simulated_date}T{entry_time_str}"
                    exit_timestamp = f"{state.session.simulated_date}T{exit_time_str}"
                
                trade = LogTrade(
                    entry_time=entry_timestamp,
                    exit_time=exit_timestamp,
                    type=f"BUY_{option_type}S" if option_type in ['CALL', 'PUT'] else option_type,
                    entry_price=entry_price,
                    exit_price=exit_price,
                    status='PENDING_PNL'
                )
                state.log_trades.append(trade)
            else:
                # Simple format: "[TRADE] Closed CALL trade"
                simple_match = re.search(r'\[TRADE\]\s*Closed\s+(CALL|PUT)\s+trade', line, re.IGNORECASE)
                if simple_match:
                    option_type = simple_match.group(1).upper()
                    exit_timestamp = None
                    if state.session.simulated_datetime:
                        exit_timestamp = state.session.simulated_datetime
                    
                    trade = LogTrade(
                        exit_time=exit_timestamp,
                        type=f"BUY_{option_type}S",
                        status='PENDING_PNL'
                    )
                    state.log_trades.append(trade)
            
            # Trim to max
            if len(state.log_trades) > max_trades:
                state.log_trades = state.log_trades[-max_trades:]
        
        # Parse Net P&L for closed trades (win/loss counted in [CLOSE] section below)
        if "Net P&L:" in line and "$" in line:
            pnl_match = re.search(r'Net P&L:\s*\$([+-]?[\d.]+)', line)
            if pnl_match:
                pnl_value = float(pnl_match.group(1))

                # NOTE: Don't count wins/losses here - [CLOSE] section handles it

                # Update most recent pending trade
                for trade in reversed(state.log_trades):
                    if trade.status in ['OPEN', 'PENDING_PNL']:
                        trade.pnl = pnl_value
                        trade.status = 'CLOSED'
                        if not trade.exit_time and state.session.simulated_datetime:
                            trade.exit_time = state.session.simulated_datetime
                        break
        
        # Parse [CLOSE] format
        if "[CLOSE] Trade closed:" in line:
            pnl_match = re.search(r'P&L=.?([+-]?[\d.]+)', line)
            if pnl_match:
                pnl_value = float(pnl_match.group(1))
                
                if pnl_value > 0:
                    state.account.winning_trades += 1
                elif pnl_value < 0:
                    state.account.losing_trades += 1
                
                for trade in reversed(state.log_trades):
                    if trade.status in ['OPEN', 'PENDING_PNL']:
                        trade.pnl = pnl_value
                        trade.status = 'CLOSED'
                        if not trade.exit_time and state.session.simulated_datetime:
                            trade.exit_time = state.session.simulated_datetime
                        break
    
    def _parse_hmm_lstm(self, line: str, state: TrainingState) -> None:
        """Parse HMM regime and TCN/temporal predictions (log tags use [LSTM] for compatibility)."""
        # HMM regime
        if "[HMM] Regime:" in line or "[HMM] Market Regime:" in line:
            full_match = re.search(
                r'\[HMM\] Regime:\s*([^,]+),\s*Confidence:\s*([\d.]+)%,\s*Trend:\s*([^,]+),\s*Vol:\s*(.+)', 
                line
            )
            if full_match:
                state.signal.hmm_regime = full_match.group(1).strip()
                state.signal.hmm_confidence = float(full_match.group(2))
                state.signal.hmm_trend = full_match.group(3).strip()
                state.signal.hmm_volatility = full_match.group(4).strip()
            else:
                match = re.search(r'(?:Regime|Market Regime):\s*([^,\(]+)', line)
                if match:
                    state.signal.hmm_regime = match.group(1).strip()
                conf_match = re.search(r'[Cc]onfidence:\s*([\d.]+)%', line)
                if conf_match:
                    state.signal.hmm_confidence = float(conf_match.group(1))
        
        # TCN/temporal predictions (log tags use [LSTM] for compatibility)
        if "[LSTM]" in line and "direction=" in line and "return=" in line:
            match = re.search(
                r'\[LSTM\]\s+([^:]+):\s+direction=(\w+),\s+return=([+-]?[\d.]+)%,\s+confidence=([\d.]+)%',
                line
            )
            if match:
                direction = match.group(2).strip()
                conf = float(match.group(4))
                
                if direction != 'N/A' and conf > 0:
                    # Clear warmup state
                    if 'warmup' in state.signal.lstm_predictions:
                        state.signal.lstm_predictions = {}
                    
                    timeframe = match.group(1).strip()
                    pred_return = float(match.group(3))
                    
                    state.signal.lstm_predictions[timeframe] = {
                        'direction': direction,
                        'return': pred_return,
                        'confidence': conf
                    }
                    
                    # Store prediction with timestamp
                    if state.session.simulated_datetime:
                        timestamp = state.session.simulated_datetime
                        
                        # Check if already have prediction at this timestamp
                        existing_idx = None
                        for i, pred in enumerate(state.lstm_predictions_history):
                            if isinstance(pred, dict) and pred.get('timestamp') == timestamp:
                                existing_idx = i
                                break
                            elif hasattr(pred, 'timestamp') and pred.timestamp == timestamp:
                                existing_idx = i
                                break
                        
                        pred_entry = {
                            'timestamp': timestamp,
                            'predictions': dict(state.signal.lstm_predictions),
                            'spy_price': state.market.spy_price
                        }
                        
                        if existing_idx is not None:
                            state.lstm_predictions_history[existing_idx] = pred_entry
                        else:
                            state.lstm_predictions_history.append(pred_entry)
                        
                        # Keep only last N predictions
                        max_predictions = 500
                        if len(state.lstm_predictions_history) > max_predictions:
                            state.lstm_predictions_history = state.lstm_predictions_history[-max_predictions:]
        
        # LSTM warmup
        if "[LSTM] WARMUP:" in line:
            match = re.search(r'WARMUP:\s*(\d+)/(\d+)', line)
            if match:
                current = int(match.group(1))
                needed = int(match.group(2))
                if not state.signal.lstm_predictions or 'warmup' in state.signal.lstm_predictions:
                    state.signal.lstm_predictions = {
                        'warmup': {'current': current, 'needed': needed, 'status': 'warming_up'}
                    }
    
    def _parse_liquidity(self, line: str, state: TrainingState) -> None:
        """Parse liquidity metrics."""
        if "[LIQUIDITY]" in line:
            if "Spread=" in line:
                match = re.search(r'Spread=([\d.]+)%', line)
                if match:
                    state.liquidity.spread = float(match.group(1))
            
            if "Vol=" in line:
                match = re.search(r'Vol=(\d+)', line)
                if match:
                    state.liquidity.volume = int(match.group(1))
            
            if "Quality=" in line:
                match = re.search(r'Quality=(\d+)/100', line)
                if match:
                    state.liquidity.exec_score = float(match.group(1))
    
    def _parse_rl_stats(self, line: str, state: TrainingState) -> None:
        """Parse RL model statistics."""
        if "[RL] Exit Policy:" in line or "Exit Policy:" in line:
            exits_match = re.search(r'(\d+)\s*exits', line)
            online_match = re.search(r'(\d+)\s*online\s*updates', line)
            if exits_match:
                state.rl_stats.exit_policy_exits = int(exits_match.group(1))
            if online_match:
                state.rl_stats.exit_policy_online_updates = int(online_match.group(1))


