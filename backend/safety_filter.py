#!/usr/bin/env python3
"""
Entry Safety Filter
====================

Optional safety filter that can APPROVE, DOWNGRADE, or VETO entry proposals
from the main RL policy.

This provides a lightweight constraint layer without replacing the RL decision.

Usage:
    from backend.safety_filter import EntrySafetyFilter, FilterDecision
    
    filter = EntrySafetyFilter(config)
    decision = filter.evaluate(proposed_action, state_features)
    
    if decision.verdict == "VETO":
        # Don't execute trade
    elif decision.verdict == "DOWNGRADE":
        # Execute with reduced size
    else:  # "APPROVE"
        # Execute as proposed
"""

import logging
import os
from dataclasses import dataclass
from typing import Dict, Optional, Any, Tuple
from enum import Enum

logger = logging.getLogger(__name__)


class FilterVerdict(Enum):
    """Possible filter decisions."""
    APPROVE = "APPROVE"      # Trade approved as proposed
    DOWNGRADE = "DOWNGRADE"  # Trade approved but with reduced size/risk
    VETO = "VETO"            # Trade rejected entirely (convert to HOLD)


@dataclass
class FilterDecision:
    """Result of safety filter evaluation."""
    verdict: FilterVerdict
    reason: str
    proposed_action: str
    final_action: str
    proposed_size: int
    final_size: int
    confidence_override: Optional[float] = None
    
    @property
    def was_modified(self) -> bool:
        return self.verdict != FilterVerdict.APPROVE


@dataclass
class SafetyFilterConfig:
    """Configuration for the safety filter."""
    enabled: bool = True
    can_veto: bool = True
    can_downgrade_size: bool = True
    
    # Veto conditions
    veto_min_confidence: float = 0.40
    veto_max_confidence: float = 1.0  # Inverted: veto HIGH confidence (data shows low conf = better)
    veto_max_vix: float = 35.0
    veto_conflicting_regime: bool = True
    veto_max_drawdown_pct: float = 0.10  # Veto if account drawdown > 10%
    veto_max_daily_trades: int = 10
    
    # Downgrade conditions
    downgrade_low_volume_threshold: float = 0.5
    downgrade_near_close_minutes: int = 60
    downgrade_size_multiplier: float = 0.5
    
    # Logging
    log_all_decisions: bool = False

    # Optional ML tradability gate
    tradability_gate_enabled: bool = False
    tradability_gate_checkpoint_path: str = "models/tradability_gate.pt"
    tradability_veto_threshold: float = 0.45
    tradability_downgrade_threshold: float = 0.55

    # Time Zone Filter - avoid first hour and last 30 min
    time_zone_filter_enabled: bool = False
    time_zone_start_minutes: int = 60  # Minutes after open to start trading
    time_zone_end_minutes: int = 30    # Minutes before close to stop trading

    # VIX Goldilocks Zone - only trade in moderate volatility
    vix_goldilocks_enabled: bool = False
    vix_goldilocks_min: float = 15.0
    vix_goldilocks_max: float = 22.0

    # Momentum Confirmation - require price moving in trade direction
    momentum_confirm_enabled: bool = False
    momentum_min_strength: float = 0.001  # 0.1% minimum 5m move

    # Day of Week Filter - skip noisy days
    day_of_week_filter_enabled: bool = False
    skip_monday: bool = True
    skip_friday: bool = True


class EntrySafetyFilter:
    """
    Safety filter for entry decisions.
    
    Acts as a lightweight constraint layer between RL policy and execution.
    Can approve, downgrade (reduce size), or veto (convert to HOLD) entries.
    
    Design principles:
    1. Simple and transparent - each rule is explicit and logged
    2. Conservative by default - when in doubt, downgrade or veto
    3. Non-blocking for learning - RL still sees all its decisions
    """
    
    def __init__(self, config: SafetyFilterConfig = None):
        self.config = config or SafetyFilterConfig()

        # Environment variable overrides for testing
        max_conf_env = os.environ.get('VETO_MAX_CONFIDENCE')
        if max_conf_env:
            self.config.veto_max_confidence = float(max_conf_env)
            logger.info(f"🛡️ VETO_MAX_CONFIDENCE override: {self.config.veto_max_confidence}")

        min_conf_env = os.environ.get('VETO_MIN_CONFIDENCE')
        if min_conf_env:
            self.config.veto_min_confidence = float(min_conf_env)
            logger.info(f"🛡️ VETO_MIN_CONFIDENCE override: {self.config.veto_min_confidence}")

        # Time Zone Filter overrides
        if os.environ.get('TIME_ZONE_FILTER', '0') == '1':
            self.config.time_zone_filter_enabled = True
            self.config.time_zone_start_minutes = int(os.environ.get('TIME_ZONE_START_MINUTES', '60'))
            self.config.time_zone_end_minutes = int(os.environ.get('TIME_ZONE_END_MINUTES', '30'))
            logger.info(f"🛡️ TIME_ZONE_FILTER enabled: skip first {self.config.time_zone_start_minutes}m, last {self.config.time_zone_end_minutes}m")

        # VIX Goldilocks Zone overrides
        if os.environ.get('VIX_GOLDILOCKS_FILTER', '0') == '1':
            self.config.vix_goldilocks_enabled = True
            self.config.vix_goldilocks_min = float(os.environ.get('VIX_GOLDILOCKS_MIN', '15.0'))
            self.config.vix_goldilocks_max = float(os.environ.get('VIX_GOLDILOCKS_MAX', '22.0'))
            logger.info(f"🛡️ VIX_GOLDILOCKS enabled: VIX {self.config.vix_goldilocks_min}-{self.config.vix_goldilocks_max}")

        # Momentum Confirmation overrides
        if os.environ.get('MOMENTUM_CONFIRM_FILTER', '0') == '1':
            self.config.momentum_confirm_enabled = True
            self.config.momentum_min_strength = float(os.environ.get('MOMENTUM_MIN_STRENGTH', '0.001'))
            logger.info(f"🛡️ MOMENTUM_CONFIRM enabled: min {self.config.momentum_min_strength:.4f}")

        # Day of Week Filter overrides
        if os.environ.get('DAY_OF_WEEK_FILTER', '0') == '1':
            self.config.day_of_week_filter_enabled = True
            self.config.skip_monday = os.environ.get('SKIP_MONDAY', '1') == '1'
            self.config.skip_friday = os.environ.get('SKIP_FRIDAY', '1') == '1'
            logger.info(f"🛡️ DAY_OF_WEEK_FILTER enabled: skip_monday={self.config.skip_monday}, skip_friday={self.config.skip_friday}")

        # Lazy-loaded tradability gate model (optional)
        self._tradability_gate = None
        
        # Track decisions for analysis
        self.stats = {
            'total_evaluations': 0,
            'approvals': 0,
            'downgrades': 0,
            'vetos': 0,
            'veto_reasons': {},
            'downgrade_reasons': {},
        }
        
        # Track daily trades
        self.trades_today = 0
        self._last_trade_date = None
        
        if self.config.enabled:
            logger.info("🛡️ Entry Safety Filter initialized")
            logger.info(f"   Can veto: {self.config.can_veto}")
            logger.info(f"   Can downgrade: {self.config.can_downgrade_size}")
            logger.info(f"   Veto thresholds: conf<{self.config.veto_min_confidence}, conf>{self.config.veto_max_confidence}, VIX>{self.config.veto_max_vix}")
            if self.config.tradability_gate_enabled:
                logger.info(f"   Tradability gate: ENABLED (ckpt={self.config.tradability_gate_checkpoint_path}, "
                            f"veto<{self.config.tradability_veto_threshold:.2f}, "
                            f"downgrade<{self.config.tradability_downgrade_threshold:.2f})")

    def _get_tradability_gate(self):
        """Lazy-load tradability gate model."""
        if self._tradability_gate is not None:
            return self._tradability_gate
        try:
            from backend.tradability_gate import TradabilityGateModel
            gate = TradabilityGateModel(device="cpu")
            if gate.load(self.config.tradability_gate_checkpoint_path):
                self._tradability_gate = gate
                return self._tradability_gate
        except Exception as e:
            logger.warning(f"[GATE] Could not load tradability gate: {e}")
        self._tradability_gate = None
        return None
    
    def evaluate(
        self,
        proposed_action: str,
        proposed_size: int,
        confidence: float,
        vix_level: float,
        hmm_trend: float,
        hmm_vol: float,
        volume_spike: float,
        minutes_to_close: int,
        account_drawdown_pct: float = 0.0,
        daily_trade_count: int = 0,
        position_direction: Optional[str] = None,  # "CALL" or "PUT"
        momentum_5m: float = 0.0,  # 5-minute momentum for confirmation
        current_weekday: int = -1,  # 0=Monday, 4=Friday, -1=unknown
    ) -> FilterDecision:
        """
        Evaluate a proposed entry action.
        
        Args:
            proposed_action: Action from RL policy ("BUY_CALLS", "BUY_PUTS", "HOLD")
            proposed_size: Proposed position size (contracts or multiplier)
            confidence: RL policy's confidence in the action
            vix_level: Current VIX level
            hmm_trend: HMM trend state (0=bearish, 0.5=neutral, 1=bullish)
            hmm_vol: HMM volatility state (0=low, 0.5=normal, 1=high)
            volume_spike: Current volume relative to average
            minutes_to_close: Minutes until market close
            account_drawdown_pct: Current account drawdown from high water mark
            daily_trade_count: Number of trades executed today
            position_direction: "CALL" or "PUT" for conflict detection
            
        Returns:
            FilterDecision with verdict and reasoning
        """
        self.stats['total_evaluations'] += 1
        
        # If filter is disabled, always approve
        if not self.config.enabled:
            return FilterDecision(
                verdict=FilterVerdict.APPROVE,
                reason="Filter disabled",
                proposed_action=proposed_action,
                final_action=proposed_action,
                proposed_size=proposed_size,
                final_size=proposed_size,
            )
        
        # HOLD actions always approved (nothing to filter)
        if proposed_action == "HOLD":
            return FilterDecision(
                verdict=FilterVerdict.APPROVE,
                reason="HOLD action - no filter needed",
                proposed_action=proposed_action,
                final_action=proposed_action,
                proposed_size=0,
                final_size=0,
            )
        
        # Determine position direction from action
        if position_direction is None:
            if "CALL" in proposed_action:
                position_direction = "CALL"
            elif "PUT" in proposed_action:
                position_direction = "PUT"

        # =====================================================================
        # TRADABILITY GATE (optional ML gate; runs before other veto/downgrade)
        # =====================================================================
        if self.config.tradability_gate_enabled and proposed_action in ("BUY_CALLS", "BUY_PUTS"):
            gate = self._get_tradability_gate()
            if gate is not None:
                # Build features from the same inputs available to the filter.
                # NOTE: hmm_* inputs here are floats in [0,1] from the caller.
                gate_features = {
                    "predicted_return": 0.0,  # Not always available at this layer; keep 0 unless caller adds it.
                    "prediction_confidence": float(confidence),
                    "vix_level": float(vix_level),
                    "momentum_5m": 0.0,
                    "volume_spike": float(volume_spike),
                    "hmm_trend": float(hmm_trend),
                    "hmm_volatility": float(hmm_vol),
                    "hmm_liquidity": 0.5,  # not provided by current signature
                    "hmm_confidence": 0.5, # not provided by current signature
                }

                p_tradable = float(gate.predict_proba(gate_features))
                if self.config.can_veto and p_tradable < self.config.tradability_veto_threshold:
                    return self._veto(
                        proposed_action, proposed_size,
                        f"Tradability gate veto: p={p_tradable:.2f} < {self.config.tradability_veto_threshold:.2f}"
                    )
                if self.config.can_downgrade_size and p_tradable < self.config.tradability_downgrade_threshold:
                    final_size = max(1, int(proposed_size * self.config.downgrade_size_multiplier))
                    return self._downgrade(
                        proposed_action, proposed_size, final_size,
                        f"Tradability gate downgrade: p={p_tradable:.2f} < {self.config.tradability_downgrade_threshold:.2f}"
                    )
        
        # =====================================================================
        # NEW FILTERS FOR WIN RATE IMPROVEMENT (Phase 41)
        # =====================================================================
        if self.config.can_veto:
            # Time Zone Filter - avoid first hour and last 30 min
            if self.config.time_zone_filter_enabled:
                minutes_since_open = 390 - minutes_to_close  # 390 = 6.5 hours trading day
                if minutes_since_open < self.config.time_zone_start_minutes:
                    return self._veto(
                        proposed_action, proposed_size,
                        f"Time zone: {minutes_since_open}m since open < {self.config.time_zone_start_minutes}m threshold"
                    )
                if minutes_to_close < self.config.time_zone_end_minutes:
                    return self._veto(
                        proposed_action, proposed_size,
                        f"Time zone: {minutes_to_close}m to close < {self.config.time_zone_end_minutes}m threshold"
                    )

            # VIX Goldilocks Zone - only trade in moderate volatility
            if self.config.vix_goldilocks_enabled:
                if vix_level < self.config.vix_goldilocks_min:
                    return self._veto(
                        proposed_action, proposed_size,
                        f"VIX too low: {vix_level:.1f} < {self.config.vix_goldilocks_min:.1f} (no edge)"
                    )
                if vix_level > self.config.vix_goldilocks_max:
                    return self._veto(
                        proposed_action, proposed_size,
                        f"VIX too high: {vix_level:.1f} > {self.config.vix_goldilocks_max:.1f} (too chaotic)"
                    )

            # Momentum Confirmation - price must be moving in trade direction
            if self.config.momentum_confirm_enabled and momentum_5m != 0.0:
                is_bullish_action = "CALL" in proposed_action
                min_strength = self.config.momentum_min_strength
                if is_bullish_action and momentum_5m < min_strength:
                    return self._veto(
                        proposed_action, proposed_size,
                        f"Momentum against CALL: {momentum_5m:.4f} < {min_strength:.4f}"
                    )
                if not is_bullish_action and momentum_5m > -min_strength:
                    return self._veto(
                        proposed_action, proposed_size,
                        f"Momentum against PUT: {momentum_5m:.4f} > {-min_strength:.4f}"
                    )

            # Day of Week Filter - skip Monday and Friday
            if self.config.day_of_week_filter_enabled and current_weekday >= 0:
                if self.config.skip_monday and current_weekday == 0:
                    return self._veto(
                        proposed_action, proposed_size,
                        "Monday skip - weekend gap risk"
                    )
                if self.config.skip_friday and current_weekday == 4:
                    return self._veto(
                        proposed_action, proposed_size,
                        "Friday skip - 0DTE theta acceleration"
                    )

        # =====================================================================
        # VETO CHECKS (if enabled)
        # =====================================================================
        if self.config.can_veto:
            # Check 1: Minimum confidence
            if confidence < self.config.veto_min_confidence:
                return self._veto(
                    proposed_action, proposed_size,
                    f"Confidence too low: {confidence:.1%} < {self.config.veto_min_confidence:.1%}"
                )

            # Check 1b: Maximum confidence (inverted filter - data shows low conf = better WR)
            # Only apply if max_confidence is set below 1.0
            if self.config.veto_max_confidence < 1.0 and confidence > self.config.veto_max_confidence:
                return self._veto(
                    proposed_action, proposed_size,
                    f"Confidence too high: {confidence:.1%} > {self.config.veto_max_confidence:.1%}"
                )

            # Check 2: Maximum VIX
            if vix_level > self.config.veto_max_vix:
                return self._veto(
                    proposed_action, proposed_size,
                    f"VIX too high: {vix_level:.1f} > {self.config.veto_max_vix:.1f}"
                )
            
            # Check 3: Regime conflict
            if self.config.veto_conflicting_regime and position_direction:
                is_bullish_action = position_direction == "CALL"
                hmm_is_bullish = hmm_trend > 0.6
                hmm_is_bearish = hmm_trend < 0.4
                
                # Veto CALL when HMM is bearish, or PUT when HMM is bullish
                if is_bullish_action and hmm_is_bearish:
                    return self._veto(
                        proposed_action, proposed_size,
                        f"CALL conflicts with bearish HMM trend: {hmm_trend:.2f}"
                    )
                elif not is_bullish_action and hmm_is_bullish:
                    return self._veto(
                        proposed_action, proposed_size,
                        f"PUT conflicts with bullish HMM trend: {hmm_trend:.2f}"
                    )
            
            # Check 4: Account drawdown
            if account_drawdown_pct > self.config.veto_max_drawdown_pct:
                return self._veto(
                    proposed_action, proposed_size,
                    f"Account drawdown too high: {account_drawdown_pct:.1%} > {self.config.veto_max_drawdown_pct:.1%}"
                )
            
            # Check 5: Daily trade limit
            if daily_trade_count >= self.config.veto_max_daily_trades:
                return self._veto(
                    proposed_action, proposed_size,
                    f"Daily trade limit reached: {daily_trade_count} >= {self.config.veto_max_daily_trades}"
                )
        
        # =====================================================================
        # DOWNGRADE CHECKS (if enabled)
        # =====================================================================
        final_size = proposed_size
        downgrade_reasons = []
        
        if self.config.can_downgrade_size:
            # Check 1: Low volume
            if volume_spike < self.config.downgrade_low_volume_threshold:
                final_size = max(1, int(final_size * self.config.downgrade_size_multiplier))
                downgrade_reasons.append(f"Low volume ({volume_spike:.2f})")
            
            # Check 2: Near market close
            if minutes_to_close < self.config.downgrade_near_close_minutes:
                final_size = max(1, int(final_size * self.config.downgrade_size_multiplier))
                downgrade_reasons.append(f"Near close ({minutes_to_close}m)")
            
            # Check 3: High volatility regime (HMM)
            if hmm_vol > 0.7:  # High vol regime
                final_size = max(1, int(final_size * 0.75))
                downgrade_reasons.append(f"High vol regime ({hmm_vol:.2f})")
            
            # Check 4: Marginal confidence
            if 0.40 <= confidence < 0.50:
                final_size = max(1, int(final_size * 0.75))
                downgrade_reasons.append(f"Marginal confidence ({confidence:.1%})")
        
        # Return downgrade if size was reduced
        if final_size < proposed_size:
            return self._downgrade(
                proposed_action, proposed_size, final_size,
                "; ".join(downgrade_reasons)
            )
        
        # =====================================================================
        # APPROVE
        # =====================================================================
        self.stats['approvals'] += 1
        
        if self.config.log_all_decisions:
            logger.debug(f"✅ Safety filter APPROVED: {proposed_action} x{proposed_size}")
        
        return FilterDecision(
            verdict=FilterVerdict.APPROVE,
            reason="All safety checks passed",
            proposed_action=proposed_action,
            final_action=proposed_action,
            proposed_size=proposed_size,
            final_size=final_size,
        )
    
    def _veto(self, proposed_action: str, proposed_size: int, reason: str) -> FilterDecision:
        """Create a VETO decision."""
        self.stats['vetos'] += 1
        self.stats['veto_reasons'][reason] = self.stats['veto_reasons'].get(reason, 0) + 1
        
        logger.info(f"🛑 Safety filter VETO: {proposed_action} → HOLD | Reason: {reason}")
        
        return FilterDecision(
            verdict=FilterVerdict.VETO,
            reason=reason,
            proposed_action=proposed_action,
            final_action="HOLD",
            proposed_size=proposed_size,
            final_size=0,
        )
    
    def _downgrade(
        self, 
        proposed_action: str, 
        proposed_size: int, 
        final_size: int, 
        reason: str
    ) -> FilterDecision:
        """Create a DOWNGRADE decision."""
        self.stats['downgrades'] += 1
        self.stats['downgrade_reasons'][reason] = self.stats['downgrade_reasons'].get(reason, 0) + 1
        
        logger.info(f"⚠️ Safety filter DOWNGRADE: {proposed_action} x{proposed_size} → x{final_size} | Reason: {reason}")
        
        return FilterDecision(
            verdict=FilterVerdict.DOWNGRADE,
            reason=reason,
            proposed_action=proposed_action,
            final_action=proposed_action,  # Action stays same, just reduced size
            proposed_size=proposed_size,
            final_size=final_size,
        )
    
    def get_stats(self) -> Dict[str, Any]:
        """Get filter statistics."""
        total = self.stats['total_evaluations']
        return {
            'total_evaluations': total,
            'approvals': self.stats['approvals'],
            'approval_rate': self.stats['approvals'] / max(1, total),
            'downgrades': self.stats['downgrades'],
            'downgrade_rate': self.stats['downgrades'] / max(1, total),
            'vetos': self.stats['vetos'],
            'veto_rate': self.stats['vetos'] / max(1, total),
            'top_veto_reasons': sorted(
                self.stats['veto_reasons'].items(),
                key=lambda x: -x[1]
            )[:5],
            'top_downgrade_reasons': sorted(
                self.stats['downgrade_reasons'].items(),
                key=lambda x: -x[1]
            )[:5],
        }
    
    def reset_daily_count(self):
        """Reset daily trade counter (call at start of trading day)."""
        self.trades_today = 0
    
    def record_trade(self):
        """Record a trade (call after successful execution)."""
        self.trades_today += 1


# Export
__all__ = [
    'EntrySafetyFilter',
    'SafetyFilterConfig',
    'FilterDecision',
    'FilterVerdict',
]
