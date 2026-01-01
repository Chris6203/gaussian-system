#!/usr/bin/env python3
"""
Unified Exit Manager
=====================

Central exit decision manager that enforces a single, clear exit policy
with hard safety rules always applied first.

Exit Priority Order:
1. HARD SAFETY RULES (always checked first, cannot be overridden)
   - Max loss threshold
   - Max hold time exceeded
   - Near expiry safety
   - Portfolio max drawdown
   - Trailing stop triggered

2. MODEL-BASED EXIT (configured type: xgboost_exit or nn_exit)
   - Only one model makes exit decisions
   - The other can be used for offline analysis only

Usage:
    from backend.unified_exit_manager import UnifiedExitManager
    
    exit_manager = UnifiedExitManager.from_config(arch_config)
    
    decision = exit_manager.should_exit(
        position=position,
        current_price=current_price,
        market_state=market_state
    )
    
    if decision.should_exit:
        # Execute exit
        logger.info(f"EXIT: {decision.reason}")
"""

import logging
from dataclasses import dataclass
from typing import Dict, Optional, Any, Tuple
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class ExitDecision:
    """Result of exit evaluation."""
    should_exit: bool
    exit_score: float  # 0-1, higher = more urgent
    reason: str
    rule_type: str  # "hard_rule" | "model_based" | "hold"
    details: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.details is None:
            self.details = {}


@dataclass
class PositionInfo:
    """Information about an open position for exit evaluation."""
    trade_id: str
    entry_price: float
    current_price: float
    entry_time: datetime
    option_type: str  # "CALL" or "PUT"
    days_to_expiry: float
    entry_confidence: float
    predicted_move_pct: float
    high_water_mark_pct: float = 0.0  # Best P&L seen
    
    @property
    def pnl_pct(self) -> float:
        """Calculate current P&L percentage."""
        if self.entry_price <= 0:
            return 0.0
        return (self.current_price - self.entry_price) / self.entry_price * 100
    
    @property
    def minutes_held(self) -> float:
        """Calculate minutes position has been held."""
        return (datetime.now() - self.entry_time).total_seconds() / 60
    
    @property
    def drawdown_from_high(self) -> float:
        """Calculate drawdown from high water mark."""
        return self.high_water_mark_pct - self.pnl_pct


@dataclass
class MarketState:
    """Current market state for exit evaluation."""
    vix_level: float = 18.0
    live_predicted_move_pct: Optional[float] = None
    live_confidence: Optional[float] = None
    hmm_trend: float = 0.5
    hmm_vol: float = 0.5
    hmm_liq: float = 0.5
    bullish_timeframe_ratio: float = 0.5
    bearish_timeframe_ratio: float = 0.5


class UnifiedExitManager:
    """
    Unified exit decision manager.
    
    Enforces a single, clear exit policy with hard safety rules
    always applied first, followed by a single configured model-based exit.
    """
    
    def __init__(
        self,
        exit_policy_type: str = "xgboost_exit",
        # Hard safety rules
        hard_stop_loss_pct: float = -15.0,
        hard_take_profit_pct: float = 40.0,
        hard_max_hold_minutes: int = 240,
        hard_min_expiry_minutes: int = 30,
        # Trailing stop
        trailing_stop_activation_pct: float = 10.0,
        trailing_stop_distance_pct: float = 5.0,
        # Model thresholds
        xgb_exit_threshold: float = 0.55,
        nn_exit_threshold: float = 0.60,
    ):
        """
        Initialize the unified exit manager.
        
        Args:
            exit_policy_type: "xgboost_exit" or "nn_exit"
            hard_stop_loss_pct: Hard stop loss (negative, e.g., -15.0)
            hard_take_profit_pct: Hard take profit (positive, e.g., 40.0)
            hard_max_hold_minutes: Maximum hold time in minutes
            hard_min_expiry_minutes: Minimum time to expiry before forced exit
            trailing_stop_activation_pct: P&L % to activate trailing stop
            trailing_stop_distance_pct: Distance for trailing stop
            xgb_exit_threshold: Exit threshold for XGBoost model
            nn_exit_threshold: Exit threshold for NN model
        """
        self.exit_policy_type = exit_policy_type
        self.hard_stop_loss_pct = hard_stop_loss_pct
        self.hard_take_profit_pct = hard_take_profit_pct
        self.hard_max_hold_minutes = hard_max_hold_minutes
        self.hard_min_expiry_minutes = hard_min_expiry_minutes
        self.trailing_stop_activation_pct = trailing_stop_activation_pct
        self.trailing_stop_distance_pct = trailing_stop_distance_pct
        self.xgb_exit_threshold = xgb_exit_threshold
        self.nn_exit_threshold = nn_exit_threshold
        
        # Model instances (lazy loaded)
        self._xgb_exit_policy = None
        self._nn_exit_policy = None
        
        # Stats
        self.stats = {
            'total_evaluations': 0,
            'hard_rule_exits': 0,
            'model_exits': 0,
            'holds': 0,
            'exit_reasons': {},
        }
        
        logger.info(f"🚪 Unified Exit Manager initialized")
        logger.info(f"   Exit policy type: {exit_policy_type}")
        logger.info(f"   Hard rules: stop={hard_stop_loss_pct}%, profit={hard_take_profit_pct}%, "
                   f"max_hold={hard_max_hold_minutes}m, min_expiry={hard_min_expiry_minutes}m")
        logger.info(f"   Trailing stop: activation={trailing_stop_activation_pct}%, "
                   f"distance={trailing_stop_distance_pct}%")
    
    @classmethod
    def from_config(cls, arch_config) -> "UnifiedExitManager":
        """Create from ArchConfig."""
        exit_cfg = arch_config.exit_policy
        return cls(
            exit_policy_type=exit_cfg.type,
            hard_stop_loss_pct=exit_cfg.hard_stop_loss_pct,
            hard_take_profit_pct=exit_cfg.hard_take_profit_pct,
            hard_max_hold_minutes=exit_cfg.hard_max_hold_minutes,
            hard_min_expiry_minutes=exit_cfg.hard_min_expiry_minutes,
            trailing_stop_activation_pct=exit_cfg.trailing_stop_activation_pct,
            trailing_stop_distance_pct=exit_cfg.trailing_stop_distance_pct,
            xgb_exit_threshold=exit_cfg.xgb_exit_threshold,
            nn_exit_threshold=exit_cfg.nn_exit_threshold,
        )
    
    @property
    def xgb_exit_policy(self):
        """Lazy load XGBoost exit policy."""
        if self._xgb_exit_policy is None:
            try:
                from backend.xgboost_exit_policy import XGBoostExitPolicy, XGBoostExitConfig
                config = XGBoostExitConfig(
                    exit_probability_threshold=self.xgb_exit_threshold,
                    fallback_take_profit=self.hard_take_profit_pct / 2,
                    fallback_stop_loss=self.hard_stop_loss_pct / 2,
                    fallback_max_hold_minutes=self.hard_max_hold_minutes / 2,
                )
                self._xgb_exit_policy = XGBoostExitPolicy(config)
                # Try to load saved model
                try:
                    self._xgb_exit_policy.load("models/xgboost_exit_policy.pkl")
                except Exception:
                    pass
            except ImportError:
                logger.warning("XGBoost exit policy not available")
        return self._xgb_exit_policy
    
    @property
    def nn_exit_policy(self):
        """Lazy load NN exit policy."""
        if self._nn_exit_policy is None:
            try:
                from backend.rl_exit_policy import RLExitPolicy
                self._nn_exit_policy = RLExitPolicy(
                    learning_rate=0.001,
                    use_expanded_inputs=True
                )
                # Try to load saved model
                try:
                    self._nn_exit_policy.load("models/rl_exit_policy.pt")
                except Exception:
                    pass
            except ImportError:
                logger.warning("NN exit policy not available")
        return self._nn_exit_policy
    
    def should_exit(
        self,
        position: PositionInfo,
        market_state: MarketState,
    ) -> ExitDecision:
        """
        Evaluate whether to exit a position.
        
        This is the SINGLE entry point for all exit decisions.
        Hard safety rules are checked first, then model-based exit.
        
        Args:
            position: Current position information
            market_state: Current market state
            
        Returns:
            ExitDecision with verdict and reasoning
        """
        self.stats['total_evaluations'] += 1
        
        pnl = position.pnl_pct
        minutes_held = position.minutes_held
        days_to_expiry = position.days_to_expiry
        
        # =================================================================
        # PHASE 1: HARD SAFETY RULES (always checked first)
        # =================================================================
        
        # Rule 1: Hard Stop Loss
        if pnl <= self.hard_stop_loss_pct:
            return self._hard_exit(
                f"STOP LOSS: P&L {pnl:.1f}% <= {self.hard_stop_loss_pct}%",
                pnl, position.trade_id
            )
        
        # Rule 2: Hard Take Profit
        if pnl >= self.hard_take_profit_pct:
            return self._hard_exit(
                f"TAKE PROFIT: P&L {pnl:.1f}% >= {self.hard_take_profit_pct}%",
                pnl, position.trade_id
            )
        
        # Rule 3: Maximum Hold Time
        if minutes_held >= self.hard_max_hold_minutes:
            return self._hard_exit(
                f"MAX HOLD TIME: {minutes_held:.0f}m >= {self.hard_max_hold_minutes}m",
                pnl, position.trade_id
            )
        
        # Rule 4: Near Expiry
        expiry_minutes = days_to_expiry * 24 * 60  # Convert to minutes
        if expiry_minutes < self.hard_min_expiry_minutes:
            return self._hard_exit(
                f"NEAR EXPIRY: {expiry_minutes:.0f}m < {self.hard_min_expiry_minutes}m",
                pnl, position.trade_id
            )

        # Rule 4b: Stochastic Exit Timing (Jerry's Quantor-MTFuzz A.30)
        # Exit at 50-75% of expected trade duration to capture most gains
        # and avoid theta decay / gamma risk near max hold
        import os
        stochastic_exit_enabled = os.environ.get('STOCHASTIC_EXIT_ENABLED', '0') == '1'
        if stochastic_exit_enabled and pnl > 0:
            try:
                # Optimal exit zone: [0.5, 0.75] of max hold time
                hold_ratio = minutes_held / self.hard_max_hold_minutes
                exit_zone_start = float(os.environ.get('STOCHASTIC_EXIT_START', '0.50'))
                exit_zone_end = float(os.environ.get('STOCHASTIC_EXIT_END', '0.75'))
                min_profit_to_exit = float(os.environ.get('STOCHASTIC_EXIT_MIN_PROFIT', '3.0'))

                if exit_zone_start <= hold_ratio <= exit_zone_end and pnl >= min_profit_to_exit:
                    return self._hard_exit(
                        f"STOCHASTIC EXIT: P&L {pnl:.1f}% at {hold_ratio*100:.0f}% of max hold "
                        f"(zone: {exit_zone_start*100:.0f}-{exit_zone_end*100:.0f}%)",
                        pnl, position.trade_id
                    )
            except Exception:
                pass

        # Rule 5: Trailing Stop
        if position.high_water_mark_pct >= self.trailing_stop_activation_pct:
            trailing_level = position.high_water_mark_pct - self.trailing_stop_distance_pct
            if pnl <= trailing_level:
                return self._hard_exit(
                    f"TRAILING STOP: P&L {pnl:.1f}% fell below {trailing_level:.1f}% "
                    f"(high was {position.high_water_mark_pct:.1f}%)",
                    pnl, position.trade_id
                )
        
        # =================================================================
        # PHASE 2: MODEL-BASED EXIT (single configured model)
        # =================================================================
        
        model_decision = self._evaluate_model_exit(position, market_state)
        if model_decision.should_exit:
            self.stats['model_exits'] += 1
            self._record_reason(model_decision.reason)
            return model_decision
        
        # =================================================================
        # DEFAULT: HOLD
        # =================================================================
        
        self.stats['holds'] += 1
        return ExitDecision(
            should_exit=False,
            exit_score=model_decision.exit_score if model_decision else 0.3,
            reason=f"HOLD: P&L {pnl:.1f}%, held {minutes_held:.0f}m",
            rule_type="hold",
            details={
                'pnl_pct': pnl,
                'minutes_held': minutes_held,
                'model_score': model_decision.exit_score if model_decision else 0.0,
            }
        )
    
    def _hard_exit(self, reason: str, pnl: float, trade_id: str) -> ExitDecision:
        """Create a hard rule exit decision."""
        self.stats['hard_rule_exits'] += 1
        self._record_reason(reason.split(':')[0])  # Record just the rule name
        
        logger.info(f"🛑 HARD EXIT ({trade_id}): {reason}")
        
        return ExitDecision(
            should_exit=True,
            exit_score=1.0,
            reason=reason,
            rule_type="hard_rule",
            details={'pnl_pct': pnl}
        )
    
    def _evaluate_model_exit(
        self,
        position: PositionInfo,
        market_state: MarketState,
    ) -> ExitDecision:
        """Evaluate exit using the configured model."""
        
        # Common parameters for both models
        common_params = {
            'prediction_timeframe_minutes': 15,  # Use config horizon
            'time_held_minutes': position.minutes_held,
            'current_pnl_pct': position.pnl_pct,
            'days_to_expiration': int(position.days_to_expiry),
            'predicted_move_pct': position.predicted_move_pct,
            'actual_move_pct': position.pnl_pct / 100,  # Rough approximation
            'entry_confidence': position.entry_confidence,
            'trade_id': position.trade_id,
            'vix_level': market_state.vix_level,
            'position_is_call': position.option_type == "CALL",
            'bullish_timeframe_ratio': market_state.bullish_timeframe_ratio,
            'bearish_timeframe_ratio': market_state.bearish_timeframe_ratio,
        }
        
        try:
            if self.exit_policy_type == "xgboost_exit" and self.xgb_exit_policy:
                should_exit, score, details = self.xgb_exit_policy.should_exit(**common_params)
                return ExitDecision(
                    should_exit=should_exit,
                    exit_score=score,
                    reason=details.get('reason', 'XGBoost exit'),
                    rule_type="model_based",
                    details=details
                )
            
            elif self.exit_policy_type == "nn_exit" and self.nn_exit_policy:
                # NN exit policy has additional parameters
                nn_params = {
                    **common_params,
                    'live_predicted_move_pct': market_state.live_predicted_move_pct,
                    'live_confidence': market_state.live_confidence,
                    'hmm_trend': market_state.hmm_trend,
                    'hmm_vol': market_state.hmm_vol,
                    'hmm_liq': market_state.hmm_liq,
                }
                should_exit, score, details = self.nn_exit_policy.should_exit(**nn_params)
                return ExitDecision(
                    should_exit=should_exit,
                    exit_score=score,
                    reason=details.get('reason', 'NN exit'),
                    rule_type="model_based",
                    details=details
                )
            
        except Exception as e:
            logger.warning(f"Model exit evaluation failed: {e}")
        
        # Fallback: return hold with low score
        return ExitDecision(
            should_exit=False,
            exit_score=0.3,
            reason="Model unavailable - defaulting to hold",
            rule_type="model_based"
        )
    
    def _record_reason(self, reason: str):
        """Record exit reason for stats."""
        self.stats['exit_reasons'][reason] = self.stats['exit_reasons'].get(reason, 0) + 1
    
    def store_experience(
        self,
        position: PositionInfo,
        final_pnl_pct: float,
        exited_early: bool,
    ):
        """
        Store exit experience for model learning.
        
        Call this when a trade closes to let the model learn.
        """
        try:
            params = {
                'prediction_timeframe_minutes': 15,
                'time_held_minutes': position.minutes_held,
                'exit_pnl_pct': final_pnl_pct,
                'days_to_expiration': int(position.days_to_expiry),
                'predicted_move_pct': position.predicted_move_pct,
                'actual_move_pct': final_pnl_pct / 100,
                'entry_confidence': position.entry_confidence,
                'exited_early': exited_early,
                'trade_id': position.trade_id,
            }
            
            if self.exit_policy_type == "xgboost_exit" and self.xgb_exit_policy:
                self.xgb_exit_policy.store_exit_experience(**params)
            elif self.exit_policy_type == "nn_exit" and self.nn_exit_policy:
                self.nn_exit_policy.store_exit_experience(**params)
                
        except Exception as e:
            logger.warning(f"Failed to store exit experience: {e}")
    
    def train_model(self) -> Optional[Dict]:
        """Train the active exit model."""
        try:
            if self.exit_policy_type == "xgboost_exit" and self.xgb_exit_policy:
                return self.xgb_exit_policy.train()
            elif self.exit_policy_type == "nn_exit" and self.nn_exit_policy:
                return self.nn_exit_policy.train_from_experiences()
        except Exception as e:
            logger.warning(f"Exit model training failed: {e}")
        return None
    
    def save(self, base_path: str = "models"):
        """Save the active exit model."""
        try:
            if self.exit_policy_type == "xgboost_exit" and self.xgb_exit_policy:
                self.xgb_exit_policy.save(f"{base_path}/xgboost_exit_policy.pkl")
            elif self.exit_policy_type == "nn_exit" and self.nn_exit_policy:
                self.nn_exit_policy.save(f"{base_path}/rl_exit_policy.pt")
        except Exception as e:
            logger.warning(f"Failed to save exit model: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get exit manager statistics."""
        total = self.stats['total_evaluations']
        return {
            'total_evaluations': total,
            'hard_rule_exits': self.stats['hard_rule_exits'],
            'hard_rule_rate': self.stats['hard_rule_exits'] / max(1, total),
            'model_exits': self.stats['model_exits'],
            'model_exit_rate': self.stats['model_exits'] / max(1, total),
            'holds': self.stats['holds'],
            'hold_rate': self.stats['holds'] / max(1, total),
            'top_exit_reasons': sorted(
                self.stats['exit_reasons'].items(),
                key=lambda x: -x[1]
            )[:10],
            'exit_policy_type': self.exit_policy_type,
        }


# Export
__all__ = [
    'UnifiedExitManager',
    'ExitDecision',
    'PositionInfo',
    'MarketState',
]
