"""
Distributional Utility Calculator for Options Trading

Calculates expected utility considering:
- P&L distribution (not just mean)
- Execution reality (fill probability, slippage, time-to-fill)
- Tail risk (5th percentile threshold)
- Multiple structure types (single-leg, verticals)

Maximizes: U(a) = E[PnL] - λ_slip*E[slip] - λ_ttf*E[ttf] - λ_reject*(1-p_fill)
"""

import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class StructureType(Enum):
    """Types of option structures"""
    SINGLE_LEG = "single_leg"
    VERTICAL_SPREAD = "vertical_spread"
    SKIP = "skip"


@dataclass
class UtilityWeights:
    """Weights for utility calculation"""
    slippage: float = 1.0      # Weight for slippage penalty
    time_to_fill: float = 0.02  # Weight for TTF penalty  
    rejection: float = 0.5      # Weight for no-fill penalty


@dataclass
class TailGuards:
    """Tail risk thresholds"""
    min_q05: float = -30.0  # Min 5th percentile P&L per contract
    min_p_fill: float = 0.60  # Min fill probability
    min_utility: float = 0.0  # Min utility to trade


@dataclass
class StructureCandidate:
    """A candidate trading structure"""
    structure_type: StructureType
    strike: float
    option_type: str  # 'call' or 'put'
    
    # For verticals
    short_strike: Optional[float] = None
    
    # Market data
    bid: float = 0.0
    ask: float = 0.0
    mid: float = 0.0
    spread_pct: float = 0.0
    bid_size: int = 0
    ask_size: int = 0
    volume: int = 0
    open_interest: int = 0
    
    # For verticals: short leg data
    short_bid: Optional[float] = None
    short_ask: Optional[float] = None
    short_mid: Optional[float] = None
    
    # Greeks
    delta: float = 0.5
    gamma: float = 0.01
    theta: float = -0.05
    vega: float = 0.10
    
    # Additional metadata
    moneyness: float = 0.0
    days_to_expiry: int = 30
    

class DistributionalUtilityCalculator:
    """
    Calculate utility for option structures considering full P&L distribution
    """
    
    def __init__(
        self,
        weights: Optional[UtilityWeights] = None,
        tail_guards: Optional[TailGuards] = None
    ):
        """
        Args:
            weights: Utility weights
            tail_guards: Tail risk thresholds
        """
        self.weights = weights or UtilityWeights()
        self.tail_guards = tail_guards or TailGuards()
        
        logger.info("[UTILITY] Distributional utility calculator initialized")
        logger.info(f"[UTILITY] Weights: slip={self.weights.slippage}, "
                   f"ttf={self.weights.time_to_fill}, reject={self.weights.rejection}")
        logger.info(f"[UTILITY] Guards: q05≥${self.tail_guards.min_q05}, "
                   f"p_fill≥{self.tail_guards.min_p_fill:.0%}, U≥{self.tail_guards.min_utility}")
    
    def calculate_utility(
        self,
        pnl_distribution: Dict[str, float],
        execution_pred: Dict[str, float],
        calibrated_confidence: float
    ) -> Tuple[float, Dict]:
        """
        Calculate expected utility for a structure
        
        Args:
            pnl_distribution: From Monte Carlo simulation
                - mean: Expected P&L
                - std: Standard deviation
                - q05, q25, q50, q75, q95: Percentiles
            execution_pred: From execution model
                - p_fill: Fill probability
                - exp_slip: Expected slippage
                - exp_ttf: Expected time-to-fill
            calibrated_confidence: Calibrated confidence (0-1)
            
        Returns:
            (utility, debug_info)
        """
        # Expected P&L weighted by confidence
        expected_pnl = pnl_distribution['mean'] * calibrated_confidence
        
        # Execution costs
        slippage_cost = self.weights.slippage * execution_pred['exp_slip'] * 100  # Convert to per contract
        ttf_cost = self.weights.time_to_fill * execution_pred['exp_ttf']
        rejection_cost = self.weights.rejection * (1.0 - execution_pred['p_fill']) * 100  # Penalty in $
        
        # Core utility
        utility_core = expected_pnl - slippage_cost - ttf_cost - rejection_cost
        
        # Final utility weighted by fill probability and confidence
        utility = execution_pred['p_fill'] * utility_core
        
        # Debug info
        debug = {
            'expected_pnl': expected_pnl,
            'pnl_mean': pnl_distribution['mean'],
            'confidence': calibrated_confidence,
            'slippage_cost': slippage_cost,
            'ttf_cost': ttf_cost,
            'rejection_cost': rejection_cost,
            'p_fill': execution_pred['p_fill'],
            'utility_core': utility_core,
            'utility': utility,
            'q05': pnl_distribution['q05'],
            'q95': pnl_distribution['q95'],
            'sharpe': pnl_distribution.get('sharpe', 0.0)
        }
        
        logger.debug(f"[UTILITY] U={utility:.2f}: E[PnL]={expected_pnl:.2f} "
                    f"- slip={slippage_cost:.2f} - ttf={ttf_cost:.2f} "
                    f"- reject={rejection_cost:.2f}, p_fill={execution_pred['p_fill']:.1%}")
        
        return utility, debug
    
    def passes_tail_guards(
        self,
        pnl_distribution: Dict[str, float],
        execution_pred: Dict[str, float],
        utility: float
    ) -> Tuple[bool, str]:
        """
        Check if structure passes tail risk guards
        
        Args:
            pnl_distribution: P&L distribution
            execution_pred: Execution prediction
            utility: Calculated utility
            
        Returns:
            (passes, reason)
        """
        # Check 5th percentile
        if pnl_distribution['q05'] < self.tail_guards.min_q05:
            return False, f"Tail risk too high: q05=${pnl_distribution['q05']:.2f} < ${self.tail_guards.min_q05:.2f}"
        
        # Check fill probability
        if execution_pred['p_fill'] < self.tail_guards.min_p_fill:
            return False, f"Fill probability too low: {execution_pred['p_fill']:.1%} < {self.tail_guards.min_p_fill:.0%}"
        
        # Check minimum utility
        if utility < self.tail_guards.min_utility:
            return False, f"Utility too low: {utility:.2f} < {self.tail_guards.min_utility:.2f}"
        
        return True, "Passed all guards"
    
    def select_best_structure(
        self,
        candidates: List[Tuple[StructureCandidate, float, Dict]]
    ) -> Tuple[Optional[StructureCandidate], Optional[float], Optional[Dict]]:
        """
        Select best structure from candidates
        
        Args:
            candidates: List of (candidate, utility, debug_info) tuples
            
        Returns:
            (best_candidate, best_utility, debug_info) or (None, None, None) if none pass
        """
        if not candidates:
            logger.info("[UTILITY] No candidates to evaluate")
            return None, None, None
        
        # Filter by tail guards
        valid_candidates = []
        for candidate, utility, debug in candidates:
            # Check tail guards
            passes, reason = self.passes_tail_guards(
                {'q05': debug['q05'], 'q95': debug['q95']},
                {'p_fill': debug['p_fill']},
                utility
            )
            
            if passes:
                valid_candidates.append((candidate, utility, debug))
            else:
                logger.debug(f"[UTILITY] Rejected {candidate.structure_type.value} "
                           f"${candidate.strike:.2f}: {reason}")
        
        if not valid_candidates:
            logger.info("[UTILITY] No candidates passed tail guards")
            return None, None, None
        
        # Select highest utility
        best = max(valid_candidates, key=lambda x: x[1])
        best_candidate, best_utility, best_debug = best
        
        logger.info(f"[UTILITY] Selected {best_candidate.structure_type.value} "
                   f"${best_candidate.strike:.2f}: U={best_utility:.2f}")
        logger.info(f"[UTILITY]   E[PnL]=${best_debug['expected_pnl']:.2f}, "
                   f"q05=${best_debug['q05']:.2f}, p_fill={best_debug['p_fill']:.1%}")
        
        return best_candidate, best_utility, best_debug
    
    def evaluate_skip_option(self) -> Tuple[float, Dict]:
        """
        Evaluate the "skip" (no trade) option
        
        Returns:
            (utility, debug_info)
        """
        # Skip has 0 P&L, 0 costs, utility = 0
        debug = {
            'expected_pnl': 0.0,
            'confidence': 0.0,
            'slippage_cost': 0.0,
            'ttf_cost': 0.0,
            'rejection_cost': 0.0,
            'p_fill': 1.0,
            'utility_core': 0.0,
            'utility': 0.0,
            'q05': 0.0,
            'q95': 0.0,
            'sharpe': 0.0
        }
        
        return 0.0, debug

