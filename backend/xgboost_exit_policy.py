#!/usr/bin/env python3
"""
XGBOOST EXIT POLICY

Uses XGBoost classifier to learn optimal exit timing from trade history.
Much more reliable than neural networks for this task:
- Trains in seconds, not hours
- Works well with small datasets (50+ trades)
- Doesn't collapse to constant outputs
- Interpretable (can see feature importance)

Falls back to simple rules when not enough training data.
"""

import logging
import pickle
import os
import numpy as np
from typing import Dict, Tuple, Optional, List
from dataclasses import dataclass, field
from datetime import datetime

logger = logging.getLogger(__name__)

# Try to import XGBoost
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    logger.warning("⚠️ XGBoost not installed. Run: pip install xgboost")


@dataclass
class ExitExperience:
    """A single exit decision experience for training"""
    # Features
    pnl_pct: float
    time_held_minutes: float
    days_to_expiry: int
    vix_level: float
    predicted_move_pct: float
    actual_move_pct: float
    entry_confidence: float
    high_water_mark: float  # Best P&L seen so far
    drawdown_from_high: float  # How far below high water mark
    time_ratio: float  # time_held / prediction_timeframe
    is_call: int  # 1 for call, 0 for put
    signal_alignment: float  # How aligned current signals are with position
    
    # Target (set after trade closes)
    should_have_exited: int = 0  # 1 if exiting here was right, 0 if holding was better
    final_pnl: float = 0.0  # What the trade ended up at
    
    timestamp: str = ""


@dataclass
class XGBoostExitConfig:
    """Configuration for XGBoost exit policy"""
    # Minimum trades before using XGBoost (use simple rules until then)
    min_trades_for_ml: int = 30

    # XGBoost hyperparameters (tuned for small datasets)
    n_estimators: int = 50
    max_depth: int = 4
    learning_rate: float = 0.1
    min_child_weight: int = 3
    subsample: float = 0.8
    colsample_bytree: float = 0.8

    # Decision threshold
    exit_probability_threshold: float = 0.55  # Exit if P(exit) > 55%

    # Fallback simple rules (used when not enough data)
    fallback_take_profit: float = 8.0
    fallback_stop_loss: float = -10.0
    fallback_max_hold_minutes: float = 45.0

    # Retraining
    retrain_every_n_trades: int = 10

    # ANTI-PREMATURE EXIT GATES (prevent exiting at tiny profits)
    min_profit_for_exit: float = 3.0  # Don't consider exit below +3% profit
    min_hold_minutes_for_exit: float = 10.0  # Don't consider exit before 10 min
    min_hold_for_profit_exit: float = 5.0  # Min hold even for profits

    # FASTER LOSS CUTTING (Phase 26 improvement)
    # Analysis showed losers held 8x longer than winners (240min vs 30min)
    # These rules ensure losing trades are cut faster
    fast_loss_cut_enabled: bool = True
    fast_loss_cut_minutes: float = 30.0  # If losing + held > 30 min, consider exiting
    fast_loss_threshold_pct: float = -2.0  # If down 2%+ after 30min, exit
    deep_loss_cut_minutes: float = 15.0  # If down 5%+, cut even faster
    deep_loss_threshold_pct: float = -5.0


def _load_exit_config_from_file() -> Dict:
    """Load exit config from config.json file."""
    import json
    try:
        config_paths = ['config.json', '../config.json', 'E:/gaussian/output3/config.json']
        for path in config_paths:
            if os.path.exists(path):
                with open(path, 'r') as f:
                    cfg = json.load(f)
                return cfg.get('entry_controller', {}).get('xgboost_exit', {})
        return {}
    except Exception:
        return {}


class XGBoostExitPolicy:
    """
    XGBoost-based exit policy for options trades.

    Learns from historical trades which conditions lead to good exits.
    Falls back to simple rules when not enough training data.

    Configuration Priority:
    1. Environment variables (for quick testing overrides)
    2. config.json entry_controller.xgboost_exit section
    3. Default values
    """

    def __init__(self, config: XGBoostExitConfig = None):
        self.config = config or XGBoostExitConfig()
        self.model: Optional[xgb.XGBClassifier] = None
        self.is_trained = False

        # Load from config.json
        file_cfg = _load_exit_config_from_file()
        if file_cfg:
            if 'min_profit_for_exit' in file_cfg:
                self.config.min_profit_for_exit = float(file_cfg['min_profit_for_exit'])
            if 'min_hold_minutes_for_exit' in file_cfg:
                self.config.min_hold_minutes_for_exit = float(file_cfg['min_hold_minutes_for_exit'])
            if 'min_hold_for_profit_exit' in file_cfg:
                self.config.min_hold_for_profit_exit = float(file_cfg['min_hold_for_profit_exit'])
            logger.info(f"[XGBOOST] Loaded exit config from config.json")
            logger.info(f"   • Min profit for exit: {self.config.min_profit_for_exit:.1f}%")

        # Environment variable overrides for Phase 26 fast loss cutting
        if os.environ.get('FAST_LOSS_CUT_ENABLED'):
            self.config.fast_loss_cut_enabled = os.environ.get('FAST_LOSS_CUT_ENABLED', '1') == '1'
        if os.environ.get('FAST_LOSS_CUT_MINUTES'):
            self.config.fast_loss_cut_minutes = float(os.environ['FAST_LOSS_CUT_MINUTES'])
        if os.environ.get('FAST_LOSS_THRESHOLD_PCT'):
            self.config.fast_loss_threshold_pct = float(os.environ['FAST_LOSS_THRESHOLD_PCT'])
        if os.environ.get('DEEP_LOSS_CUT_MINUTES'):
            self.config.deep_loss_cut_minutes = float(os.environ['DEEP_LOSS_CUT_MINUTES'])
        if os.environ.get('DEEP_LOSS_THRESHOLD_PCT'):
            self.config.deep_loss_threshold_pct = float(os.environ['DEEP_LOSS_THRESHOLD_PCT'])

        if self.config.fast_loss_cut_enabled:
            logger.info(f"[XGBOOST] Fast loss cutting ENABLED")
            logger.info(f"   • Standard cut: {self.config.fast_loss_threshold_pct:.1f}% after {self.config.fast_loss_cut_minutes:.0f}m")
            logger.info(f"   • Deep cut: {self.config.deep_loss_threshold_pct:.1f}% after {self.config.deep_loss_cut_minutes:.0f}m")

        # Experience storage
        self.experiences: List[ExitExperience] = []
        self.pending_experiences: Dict[str, List[ExitExperience]] = {}  # trade_id -> experiences
        
        # High water marks per trade
        self.high_water_marks: Dict[str, float] = {}
        
        # Stats
        self.stats = {
            'total_exits': 0,
            'ml_exits': 0,
            'rule_exits': 0,
            'trades_seen': 0,
            'last_train_trades': 0,
            'model_accuracy': 0.0,
            'feature_importance': {},
        }
        
        # Feature names for interpretability
        self.feature_names = [
            'pnl_pct', 'time_held_minutes', 'days_to_expiry', 'vix_level',
            'predicted_move_pct', 'actual_move_pct', 'entry_confidence',
            'high_water_mark', 'drawdown_from_high', 'time_ratio',
            'is_call', 'signal_alignment'
        ]
        
        if XGBOOST_AVAILABLE:
            logger.info("🌲 XGBoost Exit Policy initialized")
            logger.info(f"   • Will use ML after {self.config.min_trades_for_ml} trades")
            logger.info(f"   • Exit threshold: {self.config.exit_probability_threshold:.0%}")
        else:
            logger.warning("⚠️ XGBoost not available - using simple rules only")
    
    def _build_features(self,
                       pnl_pct: float,
                       time_held_minutes: float,
                       days_to_expiry: int,
                       vix_level: float,
                       predicted_move_pct: float,
                       actual_move_pct: float,
                       entry_confidence: float,
                       high_water_mark: float,
                       prediction_timeframe: int,
                       is_call: bool,
                       signal_alignment: float) -> np.ndarray:
        """Build feature vector for prediction"""
        drawdown = high_water_mark - pnl_pct
        time_ratio = time_held_minutes / max(1, prediction_timeframe)
        
        features = np.array([
            pnl_pct,
            time_held_minutes,
            days_to_expiry,
            vix_level,
            predicted_move_pct,
            actual_move_pct,
            entry_confidence,
            high_water_mark,
            drawdown,
            time_ratio,
            1 if is_call else 0,
            signal_alignment
        ]).reshape(1, -1)
        
        return features
    
    def _simple_rule_exit(self,
                         pnl_pct: float,
                         time_held_minutes: float,
                         days_to_expiry: int,
                         high_water_mark: float) -> Tuple[bool, float, str]:
        """Fallback simple rules when ML not ready"""
        cfg = self.config

        # Take profit
        if pnl_pct >= cfg.fallback_take_profit:
            return True, 0.95, f'💰 TAKE PROFIT (rule): +{pnl_pct:.1f}%'

        # Stop loss
        if pnl_pct <= cfg.fallback_stop_loss:
            return True, 0.99, f'🛑 STOP LOSS (rule): {pnl_pct:.1f}%'

        # === FASTER LOSS CUTTING (Phase 26 improvement) ===
        # Analysis showed losers held 8x longer than winners (240min vs 30min)
        if cfg.fast_loss_cut_enabled:
            # Deep loss cut: If down 5%+ for 15+ minutes, cut immediately
            if pnl_pct <= cfg.deep_loss_threshold_pct and time_held_minutes >= cfg.deep_loss_cut_minutes:
                return True, 0.92, f'🔪 FAST CUT (deep): {pnl_pct:.1f}% after {time_held_minutes:.0f}m'

            # Standard loss cut: If down 2%+ for 30+ minutes, cut
            if pnl_pct <= cfg.fast_loss_threshold_pct and time_held_minutes >= cfg.fast_loss_cut_minutes:
                return True, 0.88, f'🔪 FAST CUT: {pnl_pct:.1f}% after {time_held_minutes:.0f}m'

        # Trailing stop (if was +5%, don't let it go below +2%)
        if high_water_mark >= 5.0 and pnl_pct < 2.0:
            return True, 0.85, f'📉 TRAIL STOP (rule): was +{high_water_mark:.1f}%, now +{pnl_pct:.1f}%'

        # Time exit
        if time_held_minutes >= cfg.fallback_max_hold_minutes:
            return True, 0.80, f'⏰ TIME EXIT (rule): {time_held_minutes:.0f}m'

        # Near expiry
        if days_to_expiry < 1:
            return True, 0.90, f'📅 EXPIRY EXIT (rule): {days_to_expiry} days'

        # Quick scalp
        if time_held_minutes <= 5 and pnl_pct >= 3.0:
            return True, 0.85, f'⚡ SCALP (rule): +{pnl_pct:.1f}% in {time_held_minutes:.0f}m'

        return False, 0.3, f'✋ HOLD (rule): P&L {pnl_pct:+.1f}%'
    
    def should_exit(self,
                   prediction_timeframe_minutes: int,
                   time_held_minutes: float,
                   current_pnl_pct: float,
                   days_to_expiration: int,
                   predicted_move_pct: float = 0.0,
                   actual_move_pct: float = 0.0,
                   entry_confidence: float = 0.5,
                   trade_id: Optional[str] = None,
                   vix_level: float = 18.0,
                   position_is_call: Optional[bool] = None,
                   bullish_timeframe_ratio: float = 0.5,
                   bearish_timeframe_ratio: float = 0.5,
                   # Accept other args for compatibility
                   **kwargs) -> Tuple[bool, float, Dict]:
        """
        Decide whether to exit using XGBoost (or simple rules as fallback).
        """
        trade_key = trade_id or f"trade_{id(self)}"
        
        # Track high water mark
        if trade_key not in self.high_water_marks:
            self.high_water_marks[trade_key] = current_pnl_pct
        else:
            self.high_water_marks[trade_key] = max(
                self.high_water_marks[trade_key],
                current_pnl_pct
            )
        high_water = self.high_water_marks[trade_key]
        
        # Calculate signal alignment
        is_call = position_is_call if position_is_call is not None else True
        if is_call:
            signal_alignment = bullish_timeframe_ratio - bearish_timeframe_ratio
        else:
            signal_alignment = bearish_timeframe_ratio - bullish_timeframe_ratio
        
        # Store experience for later labeling
        exp = ExitExperience(
            pnl_pct=current_pnl_pct,
            time_held_minutes=time_held_minutes,
            days_to_expiry=days_to_expiration,
            vix_level=vix_level,
            predicted_move_pct=predicted_move_pct,
            actual_move_pct=actual_move_pct,
            entry_confidence=entry_confidence,
            high_water_mark=high_water,
            drawdown_from_high=high_water - current_pnl_pct,
            time_ratio=time_held_minutes / max(1, prediction_timeframe_minutes),
            is_call=1 if is_call else 0,
            signal_alignment=signal_alignment,
            timestamp=datetime.now().isoformat()
        )
        
        if trade_key not in self.pending_experiences:
            self.pending_experiences[trade_key] = []
        self.pending_experiences[trade_key].append(exp)
        
        # ALWAYS apply hard safety rules first (regardless of ML)
        if current_pnl_pct <= -15.0:  # Hard stop
            self.stats['total_exits'] += 1
            self.stats['rule_exits'] += 1
            return True, 0.99, {
                'should_exit': True,
                'exit_score': 0.99,
                'reason': f'🛑 HARD STOP: {current_pnl_pct:.1f}% (safety limit)',
                'method': 'hard_rule',
                'pnl_pct': current_pnl_pct,
            }
        
        if current_pnl_pct >= 15.0:  # Hard take profit
            self.stats['total_exits'] += 1
            self.stats['rule_exits'] += 1
            return True, 0.95, {
                'should_exit': True,
                'exit_score': 0.95,
                'reason': f'💰 BIG PROFIT: +{current_pnl_pct:.1f}% (safety lock)',
                'method': 'hard_rule',
                'pnl_pct': current_pnl_pct,
            }

        # === FAST LOSS CUTTING (Phase 26 - HARD RULE, OVERRIDES XGBOOST) ===
        # Analysis showed losers held 8x longer than winners - cut losses faster
        cfg = self.config
        if cfg.fast_loss_cut_enabled and current_pnl_pct < 0:
            # Deep loss cut: If down 5%+ for 15+ minutes, cut immediately
            if current_pnl_pct <= cfg.deep_loss_threshold_pct and time_held_minutes >= cfg.deep_loss_cut_minutes:
                self.stats['total_exits'] += 1
                self.stats['rule_exits'] += 1
                return True, 0.92, {
                    'should_exit': True,
                    'exit_score': 0.92,
                    'reason': f'🔪 FAST CUT (deep): {current_pnl_pct:.1f}% after {time_held_minutes:.0f}m',
                    'method': 'fast_loss_cut',
                    'pnl_pct': current_pnl_pct,
                }

            # Standard loss cut: If down 2%+ for 30+ minutes, cut
            if current_pnl_pct <= cfg.fast_loss_threshold_pct and time_held_minutes >= cfg.fast_loss_cut_minutes:
                self.stats['total_exits'] += 1
                self.stats['rule_exits'] += 1
                return True, 0.88, {
                    'should_exit': True,
                    'exit_score': 0.88,
                    'reason': f'🔪 FAST CUT: {current_pnl_pct:.1f}% after {time_held_minutes:.0f}m',
                    'method': 'fast_loss_cut',
                    'pnl_pct': current_pnl_pct,
                }
        # === END FAST LOSS CUTTING ===

        # ANTI-PREMATURE EXIT GATES: Don't let XGBoost exit too early or at tiny profits
        # Gate 1: Minimum hold time (except for losses)
        if current_pnl_pct >= 0 and time_held_minutes < self.config.min_hold_for_profit_exit:
            return False, 0.2, {
                'should_exit': False,
                'exit_score': 0.2,
                'reason': f'⏳ TOO EARLY: {time_held_minutes:.0f}m < {self.config.min_hold_for_profit_exit:.0f}m min hold',
                'method': 'hold_time_gate',
                'pnl_pct': current_pnl_pct,
            }

        # Gate 2: Minimum profit for positive exit (let winners develop)
        if 0 < current_pnl_pct < self.config.min_profit_for_exit:
            return False, 0.25, {
                'should_exit': False,
                'exit_score': 0.25,
                'reason': f'📈 LET RUN: +{current_pnl_pct:.1f}% < +{self.config.min_profit_for_exit:.0f}% min profit',
                'method': 'min_profit_gate',
                'pnl_pct': current_pnl_pct,
            }

        # Use XGBoost if trained, otherwise simple rules
        if self.is_trained and XGBOOST_AVAILABLE and self.model is not None:
            try:
                features = self._build_features(
                    pnl_pct=current_pnl_pct,
                    time_held_minutes=time_held_minutes,
                    days_to_expiry=days_to_expiration,
                    vix_level=vix_level,
                    predicted_move_pct=predicted_move_pct,
                    actual_move_pct=actual_move_pct,
                    entry_confidence=entry_confidence,
                    high_water_mark=high_water,
                    prediction_timeframe=prediction_timeframe_minutes,
                    is_call=is_call,
                    signal_alignment=signal_alignment
                )
                
                # Get exit probability
                exit_proba = self.model.predict_proba(features)[0][1]
                should_exit = exit_proba >= self.config.exit_probability_threshold
                
                if should_exit:
                    self.stats['total_exits'] += 1
                    self.stats['ml_exits'] += 1
                    return True, float(exit_proba), {
                        'should_exit': True,
                        'exit_score': float(exit_proba),
                        'reason': f'🌲 XGB EXIT: {exit_proba:.1%} prob (P&L: {current_pnl_pct:+.1f}%)',
                        'method': 'xgboost',
                        'pnl_pct': current_pnl_pct,
                        'exit_probability': float(exit_proba),
                    }
                else:
                    return False, float(exit_proba), {
                        'should_exit': False,
                        'exit_score': float(exit_proba),
                        'reason': f'🌲 XGB HOLD: {exit_proba:.1%} < {self.config.exit_probability_threshold:.0%} (P&L: {current_pnl_pct:+.1f}%)',
                        'method': 'xgboost',
                        'pnl_pct': current_pnl_pct,
                        'exit_probability': float(exit_proba),
                    }
            except Exception as e:
                logger.warning(f"XGBoost prediction failed: {e}, falling back to rules")
        
        # Fallback to simple rules
        should_exit, score, reason = self._simple_rule_exit(
            current_pnl_pct, time_held_minutes, days_to_expiration, high_water
        )
        
        if should_exit:
            self.stats['total_exits'] += 1
            self.stats['rule_exits'] += 1
        
        return should_exit, score, {
            'should_exit': should_exit,
            'exit_score': score,
            'reason': reason,
            'method': 'simple_rules',
            'pnl_pct': current_pnl_pct,
            'trades_until_ml': max(0, self.config.min_trades_for_ml - self.stats['trades_seen']),
        }
    
    def store_exit_experience(self,
                             prediction_timeframe_minutes: int,
                             time_held_minutes: float,
                             exit_pnl_pct: float,
                             days_to_expiration: int,
                             predicted_move_pct: float,
                             actual_move_pct: float,
                             entry_confidence: float,
                             exited_early: bool,
                             trade_id: Optional[str] = None,
                             **kwargs):
        """
        Store completed trade experience and label all decision points.
        
        Called when a trade closes - we can now label whether each
        decision point should have been an exit or hold.
        """
        trade_key = trade_id or f"trade_{id(self)}"
        
        # Get all experiences from this trade
        pending = self.pending_experiences.get(trade_key, [])
        
        if not pending:
            return
        
        # Label each experience based on final outcome
        # IMPROVED LABELING: Avoid premature exit bias
        for exp in pending:
            exp.final_pnl = exit_pnl_pct

            # Calculate time ratio (how far into prediction window we are)
            time_ratio = exp.time_ratio  # Already computed in experience
            min_time_ratio = 0.75  # Require 75% of prediction window before allowing exits

            # Calculate profit change
            profit_change = exp.pnl_pct - exit_pnl_pct

            # Should we have exited at this point?
            # NEW STRATEGY: Only label as "should exit" for significant profit loss
            # after sufficient time has passed

            should_exit = False

            # Case 1: Large profit with significant loss
            if exp.pnl_pct > 8.0 and profit_change > 1.0:
                # Had large profit (+8%), lost 1%+ → exit
                should_exit = True

            # Case 2: Moderate profit with large loss (after time threshold)
            elif time_ratio >= min_time_ratio:
                if exp.pnl_pct > 2.0 and profit_change > 2.0:
                    # Had moderate profit (+2%), lost 2%+ AND held long enough → exit
                    should_exit = True
                elif exp.pnl_pct > 5.0 and profit_change > 1.5:
                    # Had good profit (+5%), lost 1.5%+ AND held long enough → exit
                    should_exit = True

            # Case 3: Stop-loss on large losses that grew significantly
            if exp.pnl_pct < -5.0 and profit_change > 2.0:
                # Big loss (-5%) that got 2% worse → should have cut loss
                should_exit = True

            # Case 4: Disaster avoidance (loss > -10%)
            if exp.pnl_pct < -8.0 and profit_change > 1.0:
                # Huge loss that got any worse → exit
                should_exit = True

            exp.should_have_exited = 1 if should_exit else 0
            self.experiences.append(exp)
        
        # Clean up
        del self.pending_experiences[trade_key]
        if trade_key in self.high_water_marks:
            del self.high_water_marks[trade_key]
        
        self.stats['trades_seen'] += 1
        
        # Retrain periodically
        if (self.stats['trades_seen'] >= self.config.min_trades_for_ml and
            self.stats['trades_seen'] - self.stats['last_train_trades'] >= self.config.retrain_every_n_trades):
            self.train()
    
    def train(self) -> Optional[Dict]:
        """Train XGBoost model on collected experiences"""
        if not XGBOOST_AVAILABLE:
            logger.warning("Cannot train: XGBoost not installed")
            return None
        
        if len(self.experiences) < self.config.min_trades_for_ml:
            logger.info(f"Not enough data to train: {len(self.experiences)}/{self.config.min_trades_for_ml}")
            return None
        
        logger.info(f"🌲 Training XGBoost on {len(self.experiences)} experiences...")
        
        # Build training data
        X = []
        y = []
        
        for exp in self.experiences:
            features = [
                exp.pnl_pct,
                exp.time_held_minutes,
                exp.days_to_expiry,
                exp.vix_level,
                exp.predicted_move_pct,
                exp.actual_move_pct,
                exp.entry_confidence,
                exp.high_water_mark,
                exp.drawdown_from_high,
                exp.time_ratio,
                exp.is_call,
                exp.signal_alignment
            ]
            X.append(features)
            y.append(exp.should_have_exited)
        
        X = np.array(X)
        y = np.array(y)
        
        # Check class balance
        n_exit = np.sum(y)
        n_hold = len(y) - n_exit
        logger.info(f"   Class balance: {n_exit} exits, {n_hold} holds")
        
        if n_exit < 5 or n_hold < 5:
            logger.warning("Not enough examples of both classes")
            return None
        
        # Train model
        self.model = xgb.XGBClassifier(
            n_estimators=self.config.n_estimators,
            max_depth=self.config.max_depth,
            learning_rate=self.config.learning_rate,
            min_child_weight=self.config.min_child_weight,
            subsample=self.config.subsample,
            colsample_bytree=self.config.colsample_bytree,
            use_label_encoder=False,
            eval_metric='logloss',
            random_state=42
        )
        
        self.model.fit(X, y)
        self.is_trained = True
        self.stats['last_train_trades'] = self.stats['trades_seen']
        
        # Calculate accuracy
        predictions = self.model.predict(X)
        accuracy = np.mean(predictions == y)
        self.stats['model_accuracy'] = accuracy
        
        # Get feature importance
        importance = dict(zip(self.feature_names, self.model.feature_importances_))
        self.stats['feature_importance'] = importance
        
        # Log results
        logger.info(f"✅ XGBoost trained! Accuracy: {accuracy:.1%}")
        logger.info("   Top features:")
        for feat, imp in sorted(importance.items(), key=lambda x: -x[1])[:5]:
            logger.info(f"      {feat}: {imp:.3f}")
        
        return {
            'accuracy': accuracy,
            'n_samples': len(self.experiences),
            'feature_importance': importance
        }
    
    def get_stats(self) -> Dict:
        """Get policy statistics"""
        return {
            **self.stats,
            'n_experiences': len(self.experiences),
            'is_trained': self.is_trained,
            'using_ml': self.is_trained and XGBOOST_AVAILABLE,
        }
    
    def save(self, path: str):
        """Save model and experiences"""
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
        
        data = {
            'model': self.model,
            'experiences': self.experiences,
            'stats': self.stats,
            'is_trained': self.is_trained,
            'config': self.config,
        }
        
        with open(path, 'wb') as f:
            pickle.dump(data, f)
        
        logger.info(f"💾 XGBoost Exit Policy saved to {path}")
    
    def load(self, path: str):
        """Load model and experiences"""
        if not os.path.exists(path):
            logger.warning(f"No saved model at {path}")
            return False
        
        try:
            with open(path, 'rb') as f:
                data = pickle.load(f)
            
            self.model = data.get('model')
            self.experiences = data.get('experiences', [])
            self.stats = data.get('stats', self.stats)
            self.is_trained = data.get('is_trained', False)
            
            if self.is_trained:
                logger.info(f"✅ Loaded XGBoost model from {path}")
                logger.info(f"   • {len(self.experiences)} experiences")
                logger.info(f"   • Accuracy: {self.stats.get('model_accuracy', 0):.1%}")
            
            return True
        except Exception as e:
            logger.warning(f"Could not load model: {e}")
            return False
    
    def cleanup_trade(self, trade_id: str):
        """Clean up tracking for a trade"""
        if trade_id in self.high_water_marks:
            del self.high_water_marks[trade_id]
        if trade_id in self.pending_experiences:
            del self.pending_experiences[trade_id]
    
    # Compatibility
    def train_from_experiences(self, *args, **kwargs):
        """Compatibility method - training happens automatically"""
        return self.train()








