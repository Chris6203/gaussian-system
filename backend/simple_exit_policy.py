#!/usr/bin/env python3
"""
SIMPLE RULE-BASED EXIT POLICY

Replaces the broken RL exit policy with simple, sensible rules that actually work.

Rules:
1. Take profit at +8% or higher (options move fast)
2. Stop loss at -10% or lower (cut losers)
3. Time exit after 30 minutes (theta decay)
4. Trailing stop: if we hit +5%, don't let it drop below +2%
5. Quick scalp: if +3% in first 5 minutes, take it

No neural networks, no learning - just rules that make sense for options.
"""

import logging
from typing import Dict, Tuple, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class ExitConfig:
    """Configurable exit thresholds"""
    # Profit targets
    take_profit_pct: float = 8.0      # Exit at +8%
    quick_scalp_pct: float = 3.0      # Quick profit in first 5 min
    quick_scalp_minutes: float = 5.0
    
    # Stop loss
    stop_loss_pct: float = -10.0      # Exit at -10%
    
    # Trailing stop (activates after hitting trail_activate_pct)
    trail_activate_pct: float = 5.0   # Start trailing after +5%
    trail_stop_pct: float = 2.0       # Don't let it drop below +2% once trailing
    
    # Time-based exits
    max_hold_minutes: float = 45.0    # Hard exit after 45 min
    time_pressure_start: float = 20.0 # Start feeling pressure at 20 min
    
    # Near-expiry protection
    min_days_to_expiry: int = 1       # Exit if < 1 day to expiry


class SimpleExitPolicy:
    """
    Simple rule-based exit policy for options trades.
    
    No ML, no RL - just straightforward rules that work.
    """
    
    def __init__(self, config: ExitConfig = None):
        self.config = config or ExitConfig()
        self.high_water_marks: Dict[str, float] = {}  # Track best P&L per trade
        self.stats = {
            'total_exits': 0,
            'profit_exits': 0,
            'stop_loss_exits': 0,
            'time_exits': 0,
            'trail_stop_exits': 0,
            'scalp_exits': 0,
        }
        logger.info("📋 Simple Exit Policy initialized (rule-based, no RL)")
        logger.info(f"   Take profit: +{self.config.take_profit_pct}%")
        logger.info(f"   Stop loss: {self.config.stop_loss_pct}%")
        logger.info(f"   Max hold: {self.config.max_hold_minutes} min")
    
    def should_exit(self,
                   prediction_timeframe_minutes: int,
                   time_held_minutes: float,
                   current_pnl_pct: float,
                   days_to_expiration: int,
                   predicted_move_pct: float = 0.0,
                   actual_move_pct: float = 0.0,
                   entry_confidence: float = 0.5,
                   trade_id: Optional[str] = None,
                   # Accept but mostly ignore these (kept for compatibility)
                   live_predicted_move_pct: Optional[float] = None,
                   live_confidence: Optional[float] = None,
                   live_hmm_state: Optional[str] = None,
                   entry_hmm_state: Optional[str] = None,
                   vix_level: float = 18.0,
                   bid_ask_spread_pct: float = 0.0,
                   hmm_regime_info: Optional[Dict] = None,
                   position_is_call: Optional[bool] = None,
                   bullish_timeframe_ratio: float = 0.5,
                   bearish_timeframe_ratio: float = 0.5,
                   hmm_trend: float = 0.5,
                   hmm_vol: float = 0.5,
                   hmm_liq: float = 0.5,
                   vix_bb_pos: float = 0.5,
                   vix_roc: float = 0.0,
                   vix_percentile: float = 0.5,
                   **kwargs  # Accept any other args
                   ) -> Tuple[bool, float, Dict]:
        """
        Decide if we should exit based on simple rules.
        
        Returns:
            should_exit: Boolean decision
            exit_score: 0-1 score (for compatibility, higher = more urgent)
            details: Dictionary with reasoning
        """
        cfg = self.config
        trade_key = trade_id or f"trade_{id(self)}"
        
        # Track high water mark for trailing stop
        if trade_key not in self.high_water_marks:
            self.high_water_marks[trade_key] = current_pnl_pct
        else:
            self.high_water_marks[trade_key] = max(
                self.high_water_marks[trade_key], 
                current_pnl_pct
            )
        high_water = self.high_water_marks[trade_key]
        
        # ============= RULE 1: TAKE PROFIT =============
        if current_pnl_pct >= cfg.take_profit_pct:
            self.stats['total_exits'] += 1
            self.stats['profit_exits'] += 1
            return True, 0.95, {
                'should_exit': True,
                'exit_score': 0.95,
                'reason': f'💰 TAKE PROFIT: +{current_pnl_pct:.1f}% >= +{cfg.take_profit_pct}%',
                'rule': 'take_profit',
                'pnl_pct': current_pnl_pct,
                'time_held': time_held_minutes,
            }
        
        # ============= RULE 2: STOP LOSS =============
        if current_pnl_pct <= cfg.stop_loss_pct:
            self.stats['total_exits'] += 1
            self.stats['stop_loss_exits'] += 1
            return True, 0.99, {
                'should_exit': True,
                'exit_score': 0.99,
                'reason': f'🛑 STOP LOSS: {current_pnl_pct:.1f}% <= {cfg.stop_loss_pct}%',
                'rule': 'stop_loss',
                'pnl_pct': current_pnl_pct,
                'time_held': time_held_minutes,
            }
        
        # ============= RULE 3: QUICK SCALP =============
        if (time_held_minutes <= cfg.quick_scalp_minutes and 
            current_pnl_pct >= cfg.quick_scalp_pct):
            self.stats['total_exits'] += 1
            self.stats['scalp_exits'] += 1
            return True, 0.90, {
                'should_exit': True,
                'exit_score': 0.90,
                'reason': f'⚡ QUICK SCALP: +{current_pnl_pct:.1f}% in {time_held_minutes:.0f}m',
                'rule': 'quick_scalp',
                'pnl_pct': current_pnl_pct,
                'time_held': time_held_minutes,
            }
        
        # ============= RULE 4: TRAILING STOP =============
        if high_water >= cfg.trail_activate_pct:
            # Trailing stop is active
            trail_floor = cfg.trail_stop_pct
            if current_pnl_pct < trail_floor:
                self.stats['total_exits'] += 1
                self.stats['trail_stop_exits'] += 1
                return True, 0.85, {
                    'should_exit': True,
                    'exit_score': 0.85,
                    'reason': f'📉 TRAIL STOP: Was +{high_water:.1f}%, now +{current_pnl_pct:.1f}% < floor +{trail_floor}%',
                    'rule': 'trailing_stop',
                    'pnl_pct': current_pnl_pct,
                    'high_water': high_water,
                    'time_held': time_held_minutes,
                }
        
        # ============= RULE 5: TIME EXIT =============
        if time_held_minutes >= cfg.max_hold_minutes:
            self.stats['total_exits'] += 1
            self.stats['time_exits'] += 1
            return True, 0.80, {
                'should_exit': True,
                'exit_score': 0.80,
                'reason': f'⏰ TIME EXIT: {time_held_minutes:.0f}m >= {cfg.max_hold_minutes}m max',
                'rule': 'time_exit',
                'pnl_pct': current_pnl_pct,
                'time_held': time_held_minutes,
            }
        
        # ============= RULE 6: NEAR EXPIRY =============
        if days_to_expiration < cfg.min_days_to_expiry:
            self.stats['total_exits'] += 1
            return True, 0.90, {
                'should_exit': True,
                'exit_score': 0.90,
                'reason': f'📅 EXPIRY EXIT: {days_to_expiration} days left < {cfg.min_days_to_expiry} min',
                'rule': 'expiry_exit',
                'pnl_pct': current_pnl_pct,
                'days_to_expiration': days_to_expiration,
            }
        
        # ============= RULE 7: TIME PRESSURE (graduated) =============
        if time_held_minutes >= cfg.time_pressure_start:
            # After 20+ minutes, start getting antsy
            time_over = time_held_minutes - cfg.time_pressure_start
            time_remaining = cfg.max_hold_minutes - cfg.time_pressure_start
            pressure = min(1.0, time_over / time_remaining) if time_remaining > 0 else 1.0
            
            # Lower the profit threshold as time increases
            # At 20 min: need +8% to exit
            # At 30 min: need +4% to exit
            # At 40 min: need +1% to exit
            adjusted_target = cfg.take_profit_pct * (1 - pressure * 0.8)
            
            if current_pnl_pct >= adjusted_target and current_pnl_pct > 0:
                self.stats['total_exits'] += 1
                self.stats['profit_exits'] += 1
                return True, 0.70 + pressure * 0.2, {
                    'should_exit': True,
                    'exit_score': 0.70 + pressure * 0.2,
                    'reason': f'⏳ TIME PRESSURE: +{current_pnl_pct:.1f}% >= adjusted +{adjusted_target:.1f}% at {time_held_minutes:.0f}m',
                    'rule': 'time_pressure_profit',
                    'pnl_pct': current_pnl_pct,
                    'time_held': time_held_minutes,
                    'pressure': pressure,
                }
        
        # ============= NO EXIT - HOLD =============
        # Calculate a "hold score" for logging (lower = more confident in hold)
        hold_score = 0.3
        if current_pnl_pct > 0:
            hold_score = 0.2  # More confident holding winners
        if time_held_minutes < 10:
            hold_score = 0.1  # Very confident holding young trades
            
        return False, hold_score, {
            'should_exit': False,
            'exit_score': hold_score,
            'reason': f'✋ HOLD: P&L {current_pnl_pct:+.1f}%, {time_held_minutes:.0f}m held, waiting...',
            'rule': 'hold',
            'pnl_pct': current_pnl_pct,
            'time_held': time_held_minutes,
            'high_water': high_water,
        }
    
    def cleanup_trade(self, trade_id: str):
        """Clean up tracking for a closed trade"""
        if trade_id in self.high_water_marks:
            del self.high_water_marks[trade_id]
    
    def get_stats(self) -> Dict:
        """Get exit statistics"""
        return self.stats.copy()
    
    # Compatibility methods (no-ops for simple policy)
    def store_exit_experience(self, *args, **kwargs):
        """No-op: Simple policy doesn't learn"""
        pass
    
    def train_from_experiences(self, *args, **kwargs):
        """No-op: Simple policy doesn't train"""
        return None
    
    def save(self, path: str):
        """No-op: Nothing to save"""
        logger.info(f"📋 Simple Exit Policy has no model to save")
    
    def load(self, path: str):
        """No-op: Nothing to load"""
        logger.info(f"📋 Simple Exit Policy has no model to load")








