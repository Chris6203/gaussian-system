#!/usr/bin/env python3
"""
Dashboard State Management
==========================

Dataclasses for managing dashboard state in a clean, organized structure.
Breaks down the monolithic TrainingState into focused components.
"""

from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional
from datetime import datetime

# Import centralized time utilities for consistent market time
try:
    from backend.time_utils import get_market_time
except ImportError:
    # Fallback if time_utils not available
    def get_market_time():
        try:
            import pytz
            market_tz = pytz.timezone('US/Eastern')
            return pytz.UTC.localize(datetime.utcnow()).astimezone(market_tz).replace(tzinfo=None)
        except ImportError:
            return datetime.now()


@dataclass
class MarketData:
    """Current market prices."""
    spy_price: float = 0.0
    vix_value: float = 0.0
    btc_price: float = 0.0
    qqq_price: float = 0.0


@dataclass
class TrainingProgress:
    """Training progress metrics."""
    cycles: int = 0
    signals: int = 0
    trades: int = 0
    rl_updates: int = 0
    skipped_hours: int = 0
    cooldown_remaining: int = 0
    consecutive_losses: int = 0


@dataclass
class AccountState:
    """Account balance and P&L tracking."""
    initial_balance: float = 1000.0
    current_balance: float = 1000.0
    pnl: float = 0.0
    pnl_pct: float = 0.0
    winning_trades: int = 0
    losing_trades: int = 0
    
    # Auto-replenish tracking (for training)
    replenish_count: int = 0
    replenish_total: float = 0.0
    true_pnl: float = 0.0  # P&L without bailouts
    
    # Open positions (unrealized)
    unrealized_pnl: float = 0.0
    open_positions_invested: float = 0.0
    open_positions_value: float = 0.0
    
    def update_pnl(self) -> None:
        """Recalculate P&L values."""
        self.pnl = self.current_balance - self.initial_balance
        if self.initial_balance > 0:
            self.pnl_pct = (self.pnl / self.initial_balance) * 100
        self.true_pnl = self.current_balance - self.initial_balance - self.replenish_total
    
    @property
    def win_rate(self) -> float:
        """Calculate win rate percentage."""
        total = self.winning_trades + self.losing_trades
        return (self.winning_trades / max(total, 1)) * 100


@dataclass
class SignalState:
    """Current signal and prediction state."""
    last_signal: str = "Waiting..."
    signal_confidence: float = 0.0
    
    # HMM state
    hmm_regime: str = "Unknown"
    hmm_confidence: float = 0.0
    hmm_trend: str = ""
    hmm_volatility: str = ""
    
    # TCN/temporal predictions (current)
    lstm_predictions: Dict = field(default_factory=dict)  # Keep name for API compatibility


@dataclass
class LiquidityMetrics:
    """Liquidity tracking."""
    spread: float = 0.0
    volume: int = 0
    exec_score: float = 0.0


@dataclass
class RLStats:
    """RL model statistics."""
    exit_policy_exits: int = 0
    exit_policy_online_updates: int = 0
    threshold_policy_updates: int = 0


@dataclass
class SessionInfo:
    """Session and timing information."""
    simulated_date: str = "Not started"
    simulated_time: str = ""
    phase: str = "Training"  # "Training" or "Live Paper"
    current_run_id: str = ""
    start_time: float = 0.0
    last_update: str = ""
    log_file: str = ""
    
    @property
    def simulated_datetime(self) -> Optional[str]:
        """Get combined datetime string if available."""
        if self.simulated_date and self.simulated_date != "Not started" and self.simulated_time:
            return f"{self.simulated_date}T{self.simulated_time}"
        return None


@dataclass
class PositionDetail:
    """Detailed open position information."""
    id: str = ""
    entry_date: str = "--"
    entry_time: str = "--:--"
    entry_timestamp: Optional[str] = None
    option_type: str = "UNKNOWN"
    strike_price: float = 0.0
    entry_price: float = 0.0
    quantity: int = 1
    invested: float = 0.0
    current_value: float = 0.0
    current_premium: float = 0.0
    unrealized_pnl: float = 0.0
    unrealized_pnl_pct: float = 0.0
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    days_to_expiry: float = 1.0


@dataclass
class TradeRecord:
    """Record of a trade for display."""
    date: str = "--"
    time: str = "00:00"
    action: str = "UNKNOWN"
    strike: float = 0.0
    entry_price: float = 0.0
    exit_price: float = 0.0
    exit_time: str = "--"
    pnl: float = 0.0
    status: str = "UNKNOWN"
    actual_return_pct: float = 0.0
    confidence: float = 0.0


@dataclass
class LogTrade:
    """Trade parsed from log file."""
    entry_time: Optional[str] = None
    exit_time: Optional[str] = None
    type: str = "UNKNOWN"
    entry_price: float = 0.0
    exit_price: float = 0.0
    strike: Optional[float] = None
    pnl: Optional[float] = None
    status: str = "OPEN"


@dataclass
class LSTMPrediction:
    """TCN/temporal prediction record (named LSTM for API compatibility)."""
    timestamp: str = ""
    predictions: Dict = field(default_factory=dict)
    spy_price: float = 0.0
    validated: bool = False
    
    # Validation results per timeframe
    validation_results: Dict = field(default_factory=dict)

# Alias for cleaner code
TemporalPrediction = LSTMPrediction


@dataclass
class ValidatedPrediction:
    """Validated temporal prediction with accuracy info."""
    timestamp: str = ""
    validation_time: str = ""
    timeframe: str = ""
    predicted_direction: str = ""
    predicted_return: float = 0.0
    confidence: float = 0.0
    actual_direction: str = ""
    actual_move_pct: float = 0.0
    is_correct: bool = False
    entry_price: float = 0.0
    exit_price: float = 0.0


@dataclass
class ArchivedSession:
    """Archived training session summary."""
    run_id: str = "unknown"
    total_trades: int = 0
    final_balance: float = 0.0
    pnl: float = 0.0
    pnl_pct: float = 0.0
    wins: int = 0
    losses: int = 0
    cycles: int = 0
    replenish_count: int = 0
    replenish_total: float = 0.0
    archived_at: str = ""


@dataclass
class TrainingState:
    """
    Complete training dashboard state.
    
    Composes smaller dataclasses for better organization.
    """
    # Component states
    market: MarketData = field(default_factory=MarketData)
    progress: TrainingProgress = field(default_factory=TrainingProgress)
    account: AccountState = field(default_factory=AccountState)
    signal: SignalState = field(default_factory=SignalState)
    liquidity: LiquidityMetrics = field(default_factory=LiquidityMetrics)
    rl_stats: RLStats = field(default_factory=RLStats)
    session: SessionInfo = field(default_factory=SessionInfo)
    
    # Positions tracking
    current_positions: int = 0
    max_positions: int = 2
    open_positions_details: List[PositionDetail] = field(default_factory=list)
    
    # Trade records
    recent_trades: List[TradeRecord] = field(default_factory=list)
    log_trades: List[LogTrade] = field(default_factory=list)
    
    # TCN/temporal prediction tracking (named lstm_* for API compatibility)
    lstm_predictions_history: List[LSTMPrediction] = field(default_factory=list)
    lstm_validated_predictions: List[ValidatedPrediction] = field(default_factory=list)
    
    # Archived sessions
    archived_sessions: List[ArchivedSession] = field(default_factory=list)
    
    def to_api_dict(self) -> Dict:
        """
        Convert to flat dictionary for API response.
        
        Maintains backward compatibility with original API format.
        """
        # Start with flat structure expected by frontend
        data = {
            # Market data
            'spy_price': self.market.spy_price,
            'vix_value': self.market.vix_value,
            'btc_price': self.market.btc_price,
            'qqq_price': self.market.qqq_price,
            
            # Training progress
            'cycles': self.progress.cycles,
            'signals': self.progress.signals,
            'trades': self.progress.trades,
            'rl_updates': self.progress.rl_updates,
            'skipped_hours': self.progress.skipped_hours,
            'cooldown_remaining': self.progress.cooldown_remaining,
            'consecutive_losses': self.progress.consecutive_losses,
            
            # Account state
            'initial_balance': self.account.initial_balance,
            'current_balance': self.account.current_balance,
            'pnl': self.account.pnl,
            'pnl_pct': self.account.pnl_pct,
            'winning_trades': self.account.winning_trades,
            'losing_trades': self.account.losing_trades,
            'replenish_count': self.account.replenish_count,
            'replenish_total': self.account.replenish_total,
            'true_pnl': self.account.true_pnl,
            'unrealized_pnl': self.account.unrealized_pnl,
            'open_positions_invested': self.account.open_positions_invested,
            'open_positions_value': self.account.open_positions_value,
            
            # Signal state
            'last_signal': self.signal.last_signal,
            'signal_confidence': self.signal.signal_confidence,
            'hmm_regime': self.signal.hmm_regime,
            'hmm_confidence': self.signal.hmm_confidence,
            'hmm_trend': self.signal.hmm_trend,
            'hmm_volatility': self.signal.hmm_volatility,
            'lstm_predictions': self.signal.lstm_predictions,
            
            # Liquidity
            'spread': self.liquidity.spread,
            'volume': self.liquidity.volume,
            'exec_score': self.liquidity.exec_score,
            
            # RL stats
            'exit_policy_exits': self.rl_stats.exit_policy_exits,
            'exit_policy_online_updates': self.rl_stats.exit_policy_online_updates,
            'threshold_policy_updates': self.rl_stats.threshold_policy_updates,
            
            # Session info
            'simulated_date': self.session.simulated_date,
            'simulated_time': self.session.simulated_time,
            'phase': self.session.phase,
            'current_run_id': self.session.current_run_id,
            'start_time': self.session.start_time,
            'last_update': self.session.last_update,
            'log_file': self.session.log_file,
            
            # Positions
            'current_positions': self.current_positions,
            'max_positions': self.max_positions,
            'open_positions_details': [asdict(p) for p in self.open_positions_details],
            
            # Trades
            'recent_trades': [asdict(t) if hasattr(t, '__dataclass_fields__') else t for t in self.recent_trades],
            'log_trades': [asdict(t) if hasattr(t, '__dataclass_fields__') else t for t in self.log_trades],
            
            # TCN prediction history (named lstm_* for API compatibility)
            'lstm_predictions_history': [asdict(p) if hasattr(p, '__dataclass_fields__') else p for p in self.lstm_predictions_history],
            'lstm_validated_predictions': [asdict(p) if hasattr(p, '__dataclass_fields__') else p for p in self.lstm_validated_predictions],
            
            # Archived
            'archived_sessions': [asdict(s) if hasattr(s, '__dataclass_fields__') else s for s in self.archived_sessions],
        }
        
        return data
    
    def reset_for_new_session(self, new_run_id: Optional[str] = None, initial_balance: float = 1000.0) -> None:
        """Reset state for a new training session, archiving current if needed."""
        # Archive current session if it has meaningful data
        if self.progress.trades > 0 or self.progress.cycles > 10:
            archived = ArchivedSession(
                run_id=self.session.current_run_id or 'unknown',
                total_trades=self.progress.trades,
                final_balance=self.account.current_balance,
                pnl=self.account.pnl,
                pnl_pct=self.account.pnl_pct,
                wins=self.account.winning_trades,
                losses=self.account.losing_trades,
                cycles=self.progress.cycles,
                replenish_count=self.account.replenish_count,
                replenish_total=self.account.replenish_total,
                archived_at=get_market_time().strftime('%H:%M:%S')
            )
            self.archived_sessions.append(archived)
            
            # Keep only last N sessions
            if len(self.archived_sessions) > 5:
                self.archived_sessions.pop(0)
        
        # Reset components
        self.session.current_run_id = new_run_id or ""
        self.progress = TrainingProgress()
        self.account = AccountState(
            initial_balance=initial_balance,
            current_balance=initial_balance
        )
        self.signal = SignalState()
        self.recent_trades = []
        self.log_trades = []
        self.lstm_predictions_history = []
        self.lstm_validated_predictions = []
        self.open_positions_details = []


