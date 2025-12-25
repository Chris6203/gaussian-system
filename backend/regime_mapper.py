#!/usr/bin/env python3
"""
Regime Mapper
=============

Issue 3 Fix: Bridge between HMM states and VIX-based regime strategies.

The Problem:
- multi_dimensional_hmm.py discovers 2-7 states per dimension via BIC
- regime_strategies.py uses fixed VIX buckets (ULTRA_LOW, LOW, NORMAL, etc.)
- HMM states can "rotate" when retrained, breaking the mapping silently

The Solution:
- This module provides a CANONICAL regime mapping layer
- Input: {trend_state, vol_state, liq_state, vix, etc.}
- Output: Canonical regime label used for strategy selection
- VIX takes precedence for structural risk controls
- HMM fine-tunes within the VIX-defined bucket

Usage:
    from backend.regime_mapper import RegimeMapper, CanonicalRegime
    
    mapper = RegimeMapper()
    
    # Get canonical regime from HMM + VIX
    regime = mapper.map_regime(
        hmm_trend_state=0,
        hmm_vol_state=1,
        hmm_liq_state=1,
        vix_level=18.5,
        hmm_trend_labels={0: "Uptrend", 1: "Sideways", 2: "Downtrend"},
        hmm_vol_labels={0: "Low Vol", 1: "Normal Vol", 2: "High Vol"}
    )
    
    print(regime.name)  # "NORMAL_VOL"
    print(regime.confidence_multiplier)  # 1.0
"""

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Dict, Optional, Tuple, List

logger = logging.getLogger(__name__)


class CanonicalRegimeType(Enum):
    """Canonical volatility regime types (primary)."""
    ULTRA_LOW_VOL = "ULTRA_LOW_VOL"  # VIX < 12
    LOW_VOL = "LOW_VOL"              # VIX 12-15
    NORMAL_VOL = "NORMAL_VOL"        # VIX 15-20
    ELEVATED_VOL = "ELEVATED_VOL"    # VIX 20-25
    HIGH_VOL = "HIGH_VOL"            # VIX 25-35
    EXTREME_VOL = "EXTREME_VOL"      # VIX > 35


class TrendType(Enum):
    """Canonical trend types (secondary)."""
    STRONG_UPTREND = "STRONG_UPTREND"
    UPTREND = "UPTREND"
    SIDEWAYS = "SIDEWAYS"
    DOWNTREND = "DOWNTREND"
    STRONG_DOWNTREND = "STRONG_DOWNTREND"


class LiquidityType(Enum):
    """Canonical liquidity types (tertiary)."""
    HIGH_LIQUIDITY = "HIGH_LIQUIDITY"
    NORMAL_LIQUIDITY = "NORMAL_LIQUIDITY"
    LOW_LIQUIDITY = "LOW_LIQUIDITY"


@dataclass
class CanonicalRegime:
    """
    Canonical regime output from the mapper.
    
    Contains all information needed for strategy selection:
    - Primary: volatility regime (from VIX, with HMM adjustment)
    - Secondary: trend regime (from HMM)
    - Tertiary: liquidity regime (from HMM)
    - Multipliers for thresholds and position sizing
    """
    # Primary classification
    name: CanonicalRegimeType
    vix_level: float
    
    # Secondary classifications
    trend: TrendType
    liquidity: LiquidityType
    
    # Strategy parameters (derived from regime)
    confidence_multiplier: float  # Apply to base_confidence_threshold
    position_scale: float         # Scale position size
    max_hold_minutes: int         # Max hold time
    stop_loss_pct: float          # Stop loss percentage
    take_profit_pct: float        # Take profit percentage
    
    # Confidence in classification
    classification_confidence: float
    
    # Raw inputs for debugging
    hmm_trend_state: Optional[int] = None
    hmm_vol_state: Optional[int] = None
    hmm_liq_state: Optional[int] = None
    
    # Override flags
    vix_override_applied: bool = False
    
    def get_effective_threshold(self, base_threshold: float) -> float:
        """Get effective threshold after applying regime multiplier."""
        return base_threshold * self.confidence_multiplier


# =============================================================================
# REGIME PARAMETERS
# =============================================================================

REGIME_PARAMETERS = {
    CanonicalRegimeType.ULTRA_LOW_VOL: {
        'confidence_multiplier': 0.82,  # Lower threshold (more trades)
        'position_scale': 1.2,
        'max_hold_minutes': 90,
        'stop_loss_pct': 0.15,
        'take_profit_pct': 0.25,
    },
    CanonicalRegimeType.LOW_VOL: {
        'confidence_multiplier': 0.91,
        'position_scale': 1.1,
        'max_hold_minutes': 100,
        'stop_loss_pct': 0.18,
        'take_profit_pct': 0.30,
    },
    CanonicalRegimeType.NORMAL_VOL: {
        'confidence_multiplier': 1.0,
        'position_scale': 1.0,
        'max_hold_minutes': 120,
        'stop_loss_pct': 0.20,
        'take_profit_pct': 0.40,
    },
    CanonicalRegimeType.ELEVATED_VOL: {
        'confidence_multiplier': 1.09,
        'position_scale': 0.8,
        'max_hold_minutes': 90,
        'stop_loss_pct': 0.25,
        'take_profit_pct': 0.50,
    },
    CanonicalRegimeType.HIGH_VOL: {
        'confidence_multiplier': 1.18,  # Higher threshold (fewer trades)
        'position_scale': 0.5,
        'max_hold_minutes': 60,
        'stop_loss_pct': 0.30,
        'take_profit_pct': 0.60,
    },
    CanonicalRegimeType.EXTREME_VOL: {
        'confidence_multiplier': 1.36,  # Much higher threshold
        'position_scale': 0.3,
        'max_hold_minutes': 45,
        'stop_loss_pct': 0.40,
        'take_profit_pct': 0.80,
    },
}


class RegimeMapper:
    """
    Maps HMM states and VIX to canonical regimes.
    
    Key design decisions:
    1. VIX takes precedence for volatility regime (structural risk control)
    2. HMM vol state can ADJUST within VIX bucket but not override it
    3. HMM trend/liquidity states provide secondary classification
    4. Sanity checks prevent contradictory states
    """
    
    def __init__(
        self,
        vix_thresholds: Tuple[float, ...] = (12, 15, 20, 25, 35),
        hmm_adjustment_range: float = 0.1,  # HMM can shift ±10% within VIX regime
        freeze_hmm_structure: bool = True,  # Lock HMM state count per retrain
        expected_hmm_states: Tuple[int, int, int] = (3, 3, 3)  # trend, vol, liq
    ):
        """
        Args:
            vix_thresholds: VIX levels for regime boundaries
            hmm_adjustment_range: How much HMM can adjust within VIX regime
            freeze_hmm_structure: If True, warn when HMM state count changes
            expected_hmm_states: Expected (trend, vol, liq) state counts
        """
        self.vix_thresholds = vix_thresholds
        self.hmm_adjustment_range = hmm_adjustment_range
        self.freeze_hmm_structure = freeze_hmm_structure
        self.expected_hmm_states = expected_hmm_states
        
        # Track HMM structure for drift detection
        self._last_hmm_structure: Optional[Tuple[int, int, int]] = None
        self._structure_warnings = 0
        
        logger.info(f"✅ RegimeMapper initialized: VIX thresholds={vix_thresholds}")
    
    def map_regime(
        self,
        vix_level: float,
        hmm_trend_state: Optional[int] = None,
        hmm_vol_state: Optional[int] = None,
        hmm_liq_state: Optional[int] = None,
        hmm_trend_labels: Optional[Dict[int, str]] = None,
        hmm_vol_labels: Optional[Dict[int, str]] = None,
        hmm_liq_labels: Optional[Dict[int, str]] = None,
        n_trend_states: int = 3,
        n_vol_states: int = 3,
        n_liq_states: int = 3
    ) -> CanonicalRegime:
        """
        Map inputs to canonical regime.
        
        Args:
            vix_level: Current VIX level (primary)
            hmm_trend_state: HMM trend state index
            hmm_vol_state: HMM volatility state index
            hmm_liq_state: HMM liquidity state index
            hmm_trend_labels: Dict mapping state -> label (for interpretation)
            hmm_vol_labels: Dict mapping state -> label
            hmm_liq_labels: Dict mapping state -> label
            n_trend_states: Number of HMM trend states
            n_vol_states: Number of HMM vol states
            n_liq_states: Number of HMM liq states
            
        Returns:
            CanonicalRegime with all classification info
        """
        # Check HMM structure stability
        current_structure = (n_trend_states, n_vol_states, n_liq_states)
        self._check_hmm_structure(current_structure)
        
        # Step 1: Get base volatility regime from VIX (authoritative)
        vix_regime = self._classify_vix(vix_level)
        
        # Step 2: Apply sanity checks / overrides
        vix_regime, override_applied = self._apply_sanity_checks(
            vix_regime, vix_level, hmm_vol_state, hmm_vol_labels
        )
        
        # Step 3: Classify trend from HMM
        trend = self._classify_trend(hmm_trend_state, hmm_trend_labels, n_trend_states)
        
        # Step 4: Classify liquidity from HMM
        liquidity = self._classify_liquidity(hmm_liq_state, hmm_liq_labels, n_liq_states)
        
        # Step 5: Get regime parameters
        params = REGIME_PARAMETERS[vix_regime]
        
        # Step 6: Calculate classification confidence
        conf = self._calculate_confidence(vix_level, hmm_vol_state, n_vol_states)
        
        regime = CanonicalRegime(
            name=vix_regime,
            vix_level=vix_level,
            trend=trend,
            liquidity=liquidity,
            confidence_multiplier=params['confidence_multiplier'],
            position_scale=params['position_scale'],
            max_hold_minutes=params['max_hold_minutes'],
            stop_loss_pct=params['stop_loss_pct'],
            take_profit_pct=params['take_profit_pct'],
            classification_confidence=conf,
            hmm_trend_state=hmm_trend_state,
            hmm_vol_state=hmm_vol_state,
            hmm_liq_state=hmm_liq_state,
            vix_override_applied=override_applied
        )
        
        logger.debug(
            f"[REGIME] Mapped: VIX={vix_level:.1f} -> {vix_regime.value} | "
            f"Trend={trend.value} | Liq={liquidity.value} | "
            f"Conf mult={params['confidence_multiplier']:.2f}"
        )
        
        return regime
    
    def _classify_vix(self, vix_level: float) -> CanonicalRegimeType:
        """Classify volatility regime from VIX level."""
        if vix_level < self.vix_thresholds[0]:
            return CanonicalRegimeType.ULTRA_LOW_VOL
        elif vix_level < self.vix_thresholds[1]:
            return CanonicalRegimeType.LOW_VOL
        elif vix_level < self.vix_thresholds[2]:
            return CanonicalRegimeType.NORMAL_VOL
        elif vix_level < self.vix_thresholds[3]:
            return CanonicalRegimeType.ELEVATED_VOL
        elif vix_level < self.vix_thresholds[4]:
            return CanonicalRegimeType.HIGH_VOL
        else:
            return CanonicalRegimeType.EXTREME_VOL
    
    def _apply_sanity_checks(
        self,
        vix_regime: CanonicalRegimeType,
        vix_level: float,
        hmm_vol_state: Optional[int],
        hmm_vol_labels: Optional[Dict[int, str]]
    ) -> Tuple[CanonicalRegimeType, bool]:
        """
        Apply sanity checks to prevent contradictory classifications.
        
        Rule: If HMM says "low vol" but VIX > 35, override to EXTREME_VOL.
        VIX is the authoritative source for structural risk controls.
        
        Returns:
            (regime, override_applied) tuple
        """
        override_applied = False
        
        if hmm_vol_state is None or hmm_vol_labels is None:
            return vix_regime, False
        
        hmm_vol_label = hmm_vol_labels.get(hmm_vol_state, "").lower()
        
        # Sanity check: HMM says low vol but VIX is extreme
        if 'low' in hmm_vol_label and vix_level > 35:
            logger.warning(
                f"⚠️ REGIME OVERRIDE: HMM says '{hmm_vol_label}' but VIX={vix_level:.1f}! "
                f"Overriding to EXTREME_VOL for safety."
            )
            return CanonicalRegimeType.EXTREME_VOL, True
        
        # Sanity check: HMM says high vol but VIX is very low
        if ('high' in hmm_vol_label or 'extreme' in hmm_vol_label) and vix_level < 12:
            logger.warning(
                f"⚠️ REGIME OVERRIDE: HMM says '{hmm_vol_label}' but VIX={vix_level:.1f}! "
                f"Trusting VIX (ULTRA_LOW_VOL) but flagging potential HMM drift."
            )
            # Don't override in this case - just warn
            override_applied = False
        
        return vix_regime, override_applied
    
    def _classify_trend(
        self,
        hmm_state: Optional[int],
        labels: Optional[Dict[int, str]],
        n_states: int
    ) -> TrendType:
        """Classify trend from HMM state."""
        if hmm_state is None:
            return TrendType.SIDEWAYS
        
        # If we have labels, use them
        if labels:
            label = labels.get(hmm_state, "").lower()
            if 'strong' in label and 'up' in label:
                return TrendType.STRONG_UPTREND
            elif 'strong' in label and 'down' in label:
                return TrendType.STRONG_DOWNTREND
            elif 'up' in label or 'bull' in label or 'rally' in label:
                return TrendType.UPTREND
            elif 'down' in label or 'bear' in label or 'crash' in label:
                return TrendType.DOWNTREND
            else:
                return TrendType.SIDEWAYS
        
        # Fall back to position-based inference
        # Assuming states are ordered: 0=most bearish, n-1=most bullish
        if n_states >= 3:
            if hmm_state == 0:
                return TrendType.STRONG_DOWNTREND
            elif hmm_state == n_states - 1:
                return TrendType.STRONG_UPTREND
            elif hmm_state < n_states // 2:
                return TrendType.DOWNTREND
            elif hmm_state > n_states // 2:
                return TrendType.UPTREND
        
        return TrendType.SIDEWAYS
    
    def _classify_liquidity(
        self,
        hmm_state: Optional[int],
        labels: Optional[Dict[int, str]],
        n_states: int
    ) -> LiquidityType:
        """Classify liquidity from HMM state."""
        if hmm_state is None:
            return LiquidityType.NORMAL_LIQUIDITY
        
        # If we have labels, use them
        if labels:
            label = labels.get(hmm_state, "").lower()
            if 'high' in label or 'liquid' in label:
                return LiquidityType.HIGH_LIQUIDITY
            elif 'low' in label or 'illiquid' in label:
                return LiquidityType.LOW_LIQUIDITY
            else:
                return LiquidityType.NORMAL_LIQUIDITY
        
        # Fall back to position-based inference
        # Assuming states are ordered: 0=lowest liquidity, n-1=highest
        if n_states >= 3:
            if hmm_state == 0:
                return LiquidityType.LOW_LIQUIDITY
            elif hmm_state == n_states - 1:
                return LiquidityType.HIGH_LIQUIDITY
        
        return LiquidityType.NORMAL_LIQUIDITY
    
    def _calculate_confidence(
        self,
        vix_level: float,
        hmm_vol_state: Optional[int],
        n_vol_states: int
    ) -> float:
        """Calculate confidence in regime classification."""
        # Higher confidence when VIX is clearly in a bucket (not near boundary)
        conf = 1.0
        
        for threshold in self.vix_thresholds:
            distance = abs(vix_level - threshold)
            if distance < 2:  # Within 2 points of boundary
                conf *= 0.8
                break
        
        # Lower confidence if HMM is missing
        if hmm_vol_state is None:
            conf *= 0.9
        
        return conf
    
    def _check_hmm_structure(self, current: Tuple[int, int, int]) -> None:
        """Check if HMM structure has changed (potential drift)."""
        if not self.freeze_hmm_structure:
            return
        
        if self._last_hmm_structure is None:
            self._last_hmm_structure = current
            return
        
        if current != self._last_hmm_structure:
            self._structure_warnings += 1
            if self._structure_warnings <= 3:
                logger.warning(
                    f"⚠️ HMM STRUCTURE CHANGED: {self._last_hmm_structure} -> {current}! "
                    f"State meanings may have shifted. Consider revalidating regime mapping."
                )
            self._last_hmm_structure = current


def create_regime_mapper(config: Optional[Dict] = None) -> RegimeMapper:
    """Factory function to create RegimeMapper from config."""
    defaults = {
        'vix_thresholds': (12, 15, 20, 25, 35),
        'hmm_adjustment_range': 0.1,
        'freeze_hmm_structure': True,
        'expected_hmm_states': (3, 3, 3)
    }
    
    if config:
        defaults.update(config)
    
    return RegimeMapper(**defaults)


# =============================================================================
# INTEGRATION HELPER
# =============================================================================

def get_canonical_regime(
    vix_level: float,
    hmm_regime_info: Optional[Dict] = None,
    mapper: Optional[RegimeMapper] = None
) -> CanonicalRegime:
    """
    Convenience function to get canonical regime from VIX and HMM info.
    
    Args:
        vix_level: Current VIX level
        hmm_regime_info: Dict from MultiDimensionalHMM.predict_current_regime()
        mapper: RegimeMapper instance (creates one if None)
        
    Returns:
        CanonicalRegime
    """
    if mapper is None:
        mapper = RegimeMapper()
    
    if hmm_regime_info is None:
        return mapper.map_regime(vix_level=vix_level)
    
    return mapper.map_regime(
        vix_level=vix_level,
        hmm_trend_state=hmm_regime_info.get('trend_state'),
        hmm_vol_state=hmm_regime_info.get('volatility_state'),
        hmm_liq_state=hmm_regime_info.get('liquidity_state'),
        hmm_trend_labels=hmm_regime_info.get('trend_labels', {}),
        hmm_vol_labels=hmm_regime_info.get('vol_labels', {}),
        hmm_liq_labels=hmm_regime_info.get('liq_labels', {}),
        n_trend_states=hmm_regime_info.get('n_trend_states', 3),
        n_vol_states=hmm_regime_info.get('n_vol_states', 3),
        n_liq_states=hmm_regime_info.get('n_liq_states', 3)
    )


