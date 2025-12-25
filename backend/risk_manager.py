#!/usr/bin/env python3
"""
Centralized Risk Manager
========================

Issue 6 Fix: Single source of truth for all risk limits.

The Problem:
- Risk limits scattered across multiple files
- Per-trade: 2% risk, 5% position size
- Order execution: various formulas
- Portfolio: daily/weekly limits, max positions
- Easy for these to diverge when updating one place

The Solution:
- All risk calculations centralized here
- Config defines the limits, this module enforces them
- All other modules call risk_manager, don't reimplement

Also addresses Issue 4 (Cold Start):
- Reduces position sizes when calibration data is limited
- Conservative thresholds during warm-up period

Usage:
    from backend.risk_manager import RiskManager
    
    rm = RiskManager(config)
    
    # Check if trade is allowed
    can_trade, reason = rm.check_trade_allowed(
        account_balance=5000,
        current_positions=2,
        daily_pnl=-200,
        weekly_pnl=-500
    )
    
    # Get position size for a trade
    size = rm.calculate_position_size(
        account_balance=5000,
        option_price=2.50,
        calibrated_confidence=0.65,
        calibration_samples=75,
        volatility_regime='NORMAL_VOL'
    )
"""

import logging
import json
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from enum import Enum

logger = logging.getLogger(__name__)


class RiskCheckResult(Enum):
    """Risk check outcomes."""
    ALLOWED = "allowed"
    BLOCKED_DAILY_LOSS = "blocked_daily_loss"
    BLOCKED_WEEKLY_LOSS = "blocked_weekly_loss"
    BLOCKED_MAX_POSITIONS = "blocked_max_positions"
    BLOCKED_POSITION_SIZE = "blocked_position_size"
    BLOCKED_INSUFFICIENT_BALANCE = "blocked_insufficient_balance"
    BLOCKED_COLD_START = "blocked_cold_start"
    BLOCKED_TIME_CONSTRAINT = "blocked_time_constraint"


@dataclass
class RiskLimits:
    """Centralized risk limit configuration."""
    # Per-trade limits
    max_risk_per_trade_pct: float = 0.02  # 2% max loss per trade
    max_position_size_pct: float = 0.05   # 5% max position size
    
    # Portfolio limits
    max_daily_loss_pct_warn: float = 0.05   # 5% warning
    max_daily_loss_pct_stop: float = 0.10   # 10% stop trading
    max_weekly_loss_pct_stop: float = 0.15  # 15% weekly stop
    max_concurrent_positions: int = 5
    
    # Cold start limits
    cold_start_enabled: bool = True
    min_calibration_samples: int = 50
    min_calibration_samples_full_size: int = 200
    cold_start_confidence_threshold: float = 0.60
    cold_start_position_scale: float = 0.25
    
    # Position sizing
    confidence_clamp_min: float = 0.5
    confidence_clamp_max: float = 0.7
    
    @classmethod
    def from_config(cls, config: Dict) -> 'RiskLimits':
        """Create RiskLimits from config dict."""
        risk_cfg = config.get('risk', {})
        cold_cfg = config.get('cold_start', {})
        threshold_cfg = config.get('threshold_management', {})
        
        return cls(
            # Risk limits
            max_risk_per_trade_pct=risk_cfg.get('max_risk_per_trade_pct', 0.02),
            max_position_size_pct=risk_cfg.get('max_position_size_pct', 0.05),
            max_daily_loss_pct_warn=risk_cfg.get('max_daily_loss_pct_warn', 0.05),
            max_daily_loss_pct_stop=risk_cfg.get('max_daily_loss_pct_stop', 0.10),
            max_weekly_loss_pct_stop=risk_cfg.get('max_weekly_loss_pct_stop', 0.15),
            max_concurrent_positions=risk_cfg.get('max_concurrent_positions', 5),
            
            # Cold start
            cold_start_enabled=cold_cfg.get('enabled', True),
            min_calibration_samples=cold_cfg.get('min_calibration_samples', 50),
            min_calibration_samples_full_size=cold_cfg.get('min_calibration_samples_for_full_size', 200),
            cold_start_confidence_threshold=cold_cfg.get('cold_start_confidence_threshold', 0.60),
            cold_start_position_scale=cold_cfg.get('cold_start_position_scale', 0.25),
            
            # Confidence clamping
            confidence_clamp_min=risk_cfg.get('position_size_confidence_clamp', {}).get('min', 0.5),
            confidence_clamp_max=risk_cfg.get('position_size_confidence_clamp', {}).get('max', 0.7),
        )


@dataclass
class TradeRiskAssessment:
    """Assessment of risk for a proposed trade."""
    allowed: bool
    result: RiskCheckResult
    reason: str
    
    # Position sizing (if allowed)
    contracts: int = 0
    dollar_risk: float = 0.0
    position_value: float = 0.0
    
    # Cold start info
    is_cold_start: bool = False
    cold_start_scale: float = 1.0
    
    # Limits applied
    max_dollar_risk: float = 0.0
    max_position_value: float = 0.0


class RiskManager:
    """
    Centralized risk management for all trading decisions.
    
    Enforces:
    1. Per-trade risk limits (max loss, position size)
    2. Portfolio limits (daily/weekly loss, concurrent positions)
    3. Cold start constraints (reduced size during warm-up)
    4. Position sizing with confidence clamping
    """
    
    def __init__(self, config: Optional[Dict] = None, config_path: str = 'config.json'):
        """
        Initialize risk manager.
        
        Args:
            config: Config dict (if None, loads from config_path)
            config_path: Path to config.json
        """
        if config is None:
            config = self._load_config(config_path)
        
        self.limits = RiskLimits.from_config(config)
        self.config = config
        
        # Track daily/weekly P&L
        self._daily_pnl: float = 0.0
        self._weekly_pnl: float = 0.0
        self._daily_reset_time: datetime = datetime.now().replace(hour=0, minute=0, second=0)
        self._weekly_reset_time: datetime = self._get_week_start()
        
        # Track positions
        self._current_positions: int = 0
        
        logger.info("✅ RiskManager initialized")
        logger.info(f"   Max risk/trade: {self.limits.max_risk_per_trade_pct:.1%}")
        logger.info(f"   Max position: {self.limits.max_position_size_pct:.1%}")
        logger.info(f"   Daily stop: {self.limits.max_daily_loss_pct_stop:.1%}")
        logger.info(f"   Weekly stop: {self.limits.max_weekly_loss_pct_stop:.1%}")
        logger.info(f"   Cold start: {'enabled' if self.limits.cold_start_enabled else 'disabled'}")
    
    def _load_config(self, config_path: str) -> Dict:
        """Load config from file."""
        try:
            with open(config_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Could not load config from {config_path}: {e}")
            return {}
    
    def _get_week_start(self) -> datetime:
        """Get start of current week (Monday 00:00)."""
        now = datetime.now()
        days_since_monday = now.weekday()
        return (now - timedelta(days=days_since_monday)).replace(hour=0, minute=0, second=0)
    
    def check_trade_allowed(
        self,
        account_balance: float,
        current_positions: int,
        daily_pnl: float = 0.0,
        weekly_pnl: float = 0.0,
        calibration_samples: int = 0,
        minutes_to_close: Optional[int] = None,
        dte: int = 1
    ) -> Tuple[bool, RiskCheckResult, str]:
        """
        Check if a new trade is allowed given current risk state.
        
        Args:
            account_balance: Current account balance
            current_positions: Number of open positions
            daily_pnl: Today's realized P&L
            weekly_pnl: This week's realized P&L
            calibration_samples: Number of calibration samples (for cold start)
            minutes_to_close: Minutes until market close
            dte: Days to expiration of proposed contract
            
        Returns:
            (allowed, result, reason) tuple
        """
        # Update tracked values
        self._current_positions = current_positions
        self._check_reset_periods()
        self._daily_pnl = daily_pnl
        self._weekly_pnl = weekly_pnl
        
        # Check 1: Max concurrent positions
        if current_positions >= self.limits.max_concurrent_positions:
            return False, RiskCheckResult.BLOCKED_MAX_POSITIONS, \
                f"Max positions reached ({current_positions}/{self.limits.max_concurrent_positions})"
        
        # Check 2: Daily loss limit
        daily_loss_pct = abs(min(0, daily_pnl)) / account_balance if account_balance > 0 else 0
        if daily_loss_pct >= self.limits.max_daily_loss_pct_stop:
            return False, RiskCheckResult.BLOCKED_DAILY_LOSS, \
                f"Daily loss limit reached ({daily_loss_pct:.1%} >= {self.limits.max_daily_loss_pct_stop:.1%})"
        
        # Check 3: Weekly loss limit
        weekly_loss_pct = abs(min(0, weekly_pnl)) / account_balance if account_balance > 0 else 0
        if weekly_loss_pct >= self.limits.max_weekly_loss_pct_stop:
            return False, RiskCheckResult.BLOCKED_WEEKLY_LOSS, \
                f"Weekly loss limit reached ({weekly_loss_pct:.1%} >= {self.limits.max_weekly_loss_pct_stop:.1%})"
        
        # Check 4: Issue 5 - Time to close constraint for 0DTE
        if minutes_to_close is not None and dte == 0:
            min_time = self.config.get('options', {}).get(
                'time_to_close_constraints', {}
            ).get('min_time_to_close_for_0dte_entry', 90)
            
            if minutes_to_close < min_time:
                return False, RiskCheckResult.BLOCKED_TIME_CONSTRAINT, \
                    f"0DTE entry blocked: {minutes_to_close}min to close (min: {min_time})"
        
        # Check 5: Cold start (warning, not blocking unless very few samples)
        if self.limits.cold_start_enabled:
            if calibration_samples < 20:  # Very few samples
                return False, RiskCheckResult.BLOCKED_COLD_START, \
                    f"Cold start: need at least 20 calibration samples (have {calibration_samples})"
        
        return True, RiskCheckResult.ALLOWED, "Trade allowed"
    
    def _check_reset_periods(self) -> None:
        """Check if daily/weekly periods need reset."""
        now = datetime.now()
        
        # Daily reset at midnight
        if now.date() > self._daily_reset_time.date():
            self._daily_pnl = 0.0
            self._daily_reset_time = now.replace(hour=0, minute=0, second=0)
            logger.info("[RISK] Daily P&L reset")
        
        # Weekly reset on Monday
        current_week_start = self._get_week_start()
        if current_week_start > self._weekly_reset_time:
            self._weekly_pnl = 0.0
            self._weekly_reset_time = current_week_start
            logger.info("[RISK] Weekly P&L reset")
    
    def calculate_position_size(
        self,
        account_balance: float,
        option_price: float,
        calibrated_confidence: float,
        calibration_samples: int = 200,
        volatility_regime: str = 'NORMAL_VOL',
        regime_position_scale: float = 1.0
    ) -> TradeRiskAssessment:
        """
        Calculate position size for a trade.
        
        Issue 4 fix: Reduces size during cold start.
        Issue 6 fix: Centralized calculation with confidence clamping.
        
        Args:
            account_balance: Current account balance
            option_price: Option premium per contract (in dollars)
            calibrated_confidence: Calibrated confidence (0-1)
            calibration_samples: Number of calibration samples
            volatility_regime: Current vol regime (affects position scale)
            regime_position_scale: Position scale from regime (0.3-1.2)
            
        Returns:
            TradeRiskAssessment with sizing details
        """
        # Calculate hard limits
        max_dollar_risk = account_balance * self.limits.max_risk_per_trade_pct
        max_position_value = account_balance * self.limits.max_position_size_pct
        
        # Issue 4: Cold start scaling
        is_cold_start = calibration_samples < self.limits.min_calibration_samples_full_size
        if is_cold_start and self.limits.cold_start_enabled:
            if calibration_samples < self.limits.min_calibration_samples:
                cold_start_scale = self.limits.cold_start_position_scale
            else:
                # Gradual ramp from cold_start_scale to 1.0
                progress = (calibration_samples - self.limits.min_calibration_samples) / \
                          (self.limits.min_calibration_samples_full_size - self.limits.min_calibration_samples)
                cold_start_scale = self.limits.cold_start_position_scale + \
                                  progress * (1.0 - self.limits.cold_start_position_scale)
        else:
            cold_start_scale = 1.0
        
        # Issue 6: Confidence clamping (prevents over-concentration)
        # Clamp confidence contribution to narrow range
        clamped_conf = min(max(calibrated_confidence, self.limits.confidence_clamp_min), 
                          self.limits.confidence_clamp_max)
        # Normalize to get multiplier between 0.83 and 1.17
        confidence_adj = clamped_conf / 0.6  # 0.5/0.6 = 0.83, 0.7/0.6 = 1.17
        
        # Combine all scaling factors
        total_scale = regime_position_scale * cold_start_scale * confidence_adj
        
        # Calculate target position value
        target_value = max_position_value * total_scale
        
        # Don't exceed max dollar risk
        target_value = min(target_value, max_dollar_risk)
        
        # Calculate contracts (option price is per share, contracts = 100 shares)
        contract_cost = option_price * 100  # Cost per contract
        if contract_cost <= 0:
            contracts = 0
        else:
            contracts = int(target_value / contract_cost)
            contracts = max(1, contracts)  # At least 1 contract if we're trading
        
        actual_value = contracts * contract_cost
        
        # Final check: position value can't exceed max
        if actual_value > max_position_value:
            contracts = int(max_position_value / contract_cost)
            actual_value = contracts * contract_cost
        
        return TradeRiskAssessment(
            allowed=contracts > 0,
            result=RiskCheckResult.ALLOWED if contracts > 0 else RiskCheckResult.BLOCKED_POSITION_SIZE,
            reason=f"{contracts} contracts @ ${option_price:.2f}" if contracts > 0 else "Position too small",
            contracts=contracts,
            dollar_risk=actual_value,  # Premium paid = max loss for long options
            position_value=actual_value,
            is_cold_start=is_cold_start,
            cold_start_scale=cold_start_scale,
            max_dollar_risk=max_dollar_risk,
            max_position_value=max_position_value
        )
    
    def get_effective_max_hold(
        self,
        config_max_hold_minutes: int,
        minutes_to_close: int,
        safety_margin_minutes: int = 20
    ) -> int:
        """
        Calculate effective max hold time.
        
        Issue 5 fix: max_hold must respect time-to-close.
        
        Args:
            config_max_hold_minutes: Max hold from config/regime
            minutes_to_close: Minutes until market close
            safety_margin_minutes: Buffer before close
            
        Returns:
            Effective max hold in minutes
        """
        # Formula from Issue 5: min(config_max_hold, minutes_to_close - safety_margin)
        time_constrained = max(0, minutes_to_close - safety_margin_minutes)
        effective = min(config_max_hold_minutes, time_constrained)
        
        if effective < config_max_hold_minutes:
            logger.debug(f"[RISK] Max hold constrained: {config_max_hold_minutes}min -> {effective}min (close in {minutes_to_close}min)")
        
        return effective
    
    def record_trade_pnl(self, pnl: float) -> None:
        """Record P&L from a closed trade."""
        self._daily_pnl += pnl
        self._weekly_pnl += pnl
    
    def get_risk_status(self, account_balance: float) -> Dict:
        """Get current risk status summary."""
        daily_loss_pct = abs(min(0, self._daily_pnl)) / account_balance if account_balance > 0 else 0
        weekly_loss_pct = abs(min(0, self._weekly_pnl)) / account_balance if account_balance > 0 else 0
        
        return {
            'daily_pnl': self._daily_pnl,
            'weekly_pnl': self._weekly_pnl,
            'daily_loss_pct': daily_loss_pct,
            'weekly_loss_pct': weekly_loss_pct,
            'daily_limit_pct': self.limits.max_daily_loss_pct_stop,
            'weekly_limit_pct': self.limits.max_weekly_loss_pct_stop,
            'daily_headroom': self.limits.max_daily_loss_pct_stop - daily_loss_pct,
            'weekly_headroom': self.limits.max_weekly_loss_pct_stop - weekly_loss_pct,
            'current_positions': self._current_positions,
            'max_positions': self.limits.max_concurrent_positions,
            'is_daily_warning': daily_loss_pct >= self.limits.max_daily_loss_pct_warn,
            'is_daily_stop': daily_loss_pct >= self.limits.max_daily_loss_pct_stop,
            'is_weekly_stop': weekly_loss_pct >= self.limits.max_weekly_loss_pct_stop,
        }


def create_risk_manager(config: Optional[Dict] = None) -> RiskManager:
    """Factory function to create RiskManager."""
    return RiskManager(config=config)


# =============================================================================
# UNIT TEST ASSERTIONS (for tests/)
# =============================================================================

def assert_position_within_limits(
    position_value: float,
    account_balance: float,
    limits: RiskLimits
) -> None:
    """Assert position doesn't exceed limits. For unit tests."""
    max_value = account_balance * limits.max_position_size_pct
    assert position_value <= max_value, \
        f"Position value ${position_value:.2f} exceeds max ${max_value:.2f}"

def assert_risk_within_limits(
    premium_paid: float,
    account_balance: float,
    limits: RiskLimits
) -> None:
    """Assert max potential loss doesn't exceed limits. For unit tests."""
    max_risk = account_balance * limits.max_risk_per_trade_pct
    assert premium_paid <= max_risk, \
        f"Premium ${premium_paid:.2f} exceeds max risk ${max_risk:.2f}"









