"""
Signal Generation Module
========================

Handles signal generation, combination, and confidence calculation:

- SignalCombiner: Combines signals from multiple sources
- ConfidenceCalculator: Calculates calibrated confidence scores
- SignalState: Manages signal state with hysteresis
- DirectionAnalyzer: Analyzes predicted direction from probabilities

Usage:
    from bot_modules.signals import SignalCombiner, ConfidenceCalculator
    
    combiner = SignalCombiner(config)
    signal = combiner.combine(neural_signal, options_signal, technical_signal)
"""

import logging
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


class SignalAction(Enum):
    """Trading signal actions."""
    HOLD = "HOLD"
    BUY_CALLS = "BUY_CALLS"
    BUY_PUTS = "BUY_PUTS"


class SignalState(Enum):
    """Signal state for hysteresis."""
    NEUTRAL = "NEUTRAL"
    BULLISH = "BULLISH"
    BEARISH = "BEARISH"


@dataclass
class Signal:
    """Trading signal with metadata."""
    action: SignalAction
    confidence: float
    raw_confidence: float
    direction_probabilities: Tuple[float, float, float]  # (up, down, neutral)
    predicted_return: float
    predicted_volatility: float
    reasoning: List[str] = field(default_factory=list)
    metadata: Dict = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'action': self.action.value,
            'confidence': self.confidence,
            'raw_confidence': self.raw_confidence,
            'direction_up': self.direction_probabilities[0],
            'direction_down': self.direction_probabilities[1],
            'direction_neutral': self.direction_probabilities[2],
            'predicted_return': self.predicted_return,
            'predicted_volatility': self.predicted_volatility,
            'reasoning': self.reasoning,
            **self.metadata
        }


@dataclass 
class SignalConfig:
    """Configuration for signal generation - OPTIMIZED FOR HIGH WIN RATE."""
    min_confidence_threshold: float = 0.55  # Minimum confidence to consider a trade
    entry_threshold: float = 0.55           # Threshold to enter new positions
    exit_threshold: float = 0.45            # RAISED from 0.40 - exit earlier if confidence drops
    neutral_zone: float = 0.03              # TIGHTENED from 0.05 - narrower neutral zone
    direction_threshold: float = 0.40       # RAISED from 0.35 - require stronger directional signal
    volatility_scale_factor: float = 1.0
    use_hysteresis: bool = True


class DirectionAnalyzer:
    """
    Analyzes direction probabilities to determine signal direction.
    """
    
    def __init__(self, config: SignalConfig = None):
        self.config = config or SignalConfig()
    
    def analyze(
        self,
        probabilities: Tuple[float, float, float],
        predicted_return: float = 0.0
    ) -> Tuple[SignalAction, float]:
        """
        Analyze direction probabilities - STRICTER for higher win rate.
        
        Args:
            probabilities: (up_prob, down_prob, neutral_prob)
            predicted_return: Predicted return (for tiebreaking)
            
        Returns:
            (action, confidence) tuple
        """
        up_prob, down_prob, neutral_prob = probabilities
        
        # Calculate net directional confidence
        net_direction = up_prob - down_prob
        
        # STRICTER: Require BOTH probability threshold AND meaningful spread
        min_spread = 0.10  # At least 10% difference between up and down
        
        # Determine action based on probabilities
        if up_prob > down_prob and up_prob > neutral_prob:
            if up_prob > self.config.direction_threshold:
                # ADDED: Check spread between up and down
                if (up_prob - down_prob) >= min_spread:
                    return SignalAction.BUY_CALLS, up_prob
        elif down_prob > up_prob and down_prob > neutral_prob:
            if down_prob > self.config.direction_threshold:
                # ADDED: Check spread between down and up
                if (down_prob - up_prob) >= min_spread:
                    return SignalAction.BUY_PUTS, down_prob
        
        # STRICTER: Only use predicted return tiebreaker if it's meaningful
        if abs(net_direction) < self.config.neutral_zone:
            # RAISED threshold from 0.001 to 0.005 (0.5% predicted move)
            if predicted_return > 0.005 and up_prob > 0.35:
                return SignalAction.BUY_CALLS, max(up_prob, 0.5)
            elif predicted_return < -0.005 and down_prob > 0.35:
                return SignalAction.BUY_PUTS, max(down_prob, 0.5)
        
        return SignalAction.HOLD, max(neutral_prob, 0.5)


class ConfidenceCalculator:
    """
    Calculates and adjusts confidence scores.
    
    Combines:
    - Raw model confidence
    - Direction probability strength
    - Return magnitude
    - Volatility adjustment
    - Consistency factor
    """
    
    def __init__(self, config: SignalConfig = None):
        self.config = config or SignalConfig()
    
    def calculate(
        self,
        raw_confidence: float,
        direction_probabilities: Tuple[float, float, float],
        predicted_return: float,
        predicted_volatility: float,
        consistency_factor: float = 1.0
    ) -> float:
        """
        Calculate adjusted confidence.
        
        Args:
            raw_confidence: Raw model confidence (0-1)
            direction_probabilities: (up, down, neutral)
            predicted_return: Predicted return
            predicted_volatility: Predicted volatility
            consistency_factor: How consistent recent predictions are
            
        Returns:
            Adjusted confidence (0-1)
        """
        up_prob, down_prob, neutral_prob = direction_probabilities
        
        # Base confidence from direction strength
        direction_strength = max(up_prob, down_prob) - neutral_prob
        direction_confidence = 0.5 + direction_strength * 0.5
        
        # Return magnitude factor
        return_factor = min(1.0, abs(predicted_return) * 100)  # Scale small returns
        
        # Volatility adjustment (lower confidence in high vol)
        vol_adjustment = 1.0
        if predicted_volatility > 0.02:  # High volatility
            vol_adjustment = 0.9
        elif predicted_volatility < 0.005:  # Very low volatility
            vol_adjustment = 0.95  # Slight penalty for low vol (harder to profit)
        
        # Combine factors
        confidence = (
            0.4 * raw_confidence +
            0.3 * direction_confidence +
            0.2 * return_factor +
            0.1 * consistency_factor
        ) * vol_adjustment
        
        # Clamp to valid range
        return float(np.clip(confidence, 0.0, 1.0))


class SignalStateMachine:
    """
    Manages signal state with hysteresis to prevent rapid switching.
    
    State transitions:
    - NEUTRAL -> BULLISH: When signal is BUY_CALLS with high confidence
    - NEUTRAL -> BEARISH: When signal is BUY_PUTS with high confidence
    - BULLISH -> NEUTRAL: When confidence drops below exit threshold
    - BULLISH -> BEARISH: Hard flip on strong opposing signal
    - BEARISH -> NEUTRAL: When confidence drops below exit threshold
    - BEARISH -> BULLISH: Hard flip on strong opposing signal
    """
    
    def __init__(self, config: SignalConfig = None):
        self.config = config or SignalConfig()
        self._state = SignalState.NEUTRAL
        self._state_entry_time = None
        self._state_entry_confidence = 0.0
    
    @property
    def state(self) -> SignalState:
        return self._state
    
    def update(self, action: SignalAction, confidence: float) -> Tuple[SignalAction, bool]:
        """
        Update state machine with new signal.
        
        Args:
            action: Proposed action
            confidence: Signal confidence
            
        Returns:
            (final_action, state_changed) tuple
        """
        state_changed = False
        final_action = action
        
        entry_thresh = self.config.entry_threshold
        exit_thresh = self.config.exit_threshold
        
        if self._state == SignalState.NEUTRAL:
            if action == SignalAction.BUY_CALLS and confidence >= entry_thresh:
                self._state = SignalState.BULLISH
                state_changed = True
                logger.debug(f"State: NEUTRAL -> BULLISH (conf={confidence:.1%})")
            elif action == SignalAction.BUY_PUTS and confidence >= entry_thresh:
                self._state = SignalState.BEARISH
                state_changed = True
                logger.debug(f"State: NEUTRAL -> BEARISH (conf={confidence:.1%})")
            else:
                final_action = SignalAction.HOLD
        
        elif self._state == SignalState.BULLISH:
            if action == SignalAction.BUY_PUTS and confidence >= entry_thresh:
                # Hard flip
                self._state = SignalState.BEARISH
                state_changed = True
                logger.debug(f"State: BULLISH -> BEARISH (hard flip)")
            elif confidence < exit_thresh:
                # Exit on weak signal
                self._state = SignalState.NEUTRAL
                state_changed = True
                final_action = SignalAction.HOLD
                logger.debug(f"State: BULLISH -> NEUTRAL (conf dropped to {confidence:.1%})")
            elif action == SignalAction.HOLD:
                # Maintain position with weaker signal
                final_action = SignalAction.BUY_CALLS
        
        elif self._state == SignalState.BEARISH:
            if action == SignalAction.BUY_CALLS and confidence >= entry_thresh:
                # Hard flip
                self._state = SignalState.BULLISH
                state_changed = True
                logger.debug(f"State: BEARISH -> BULLISH (hard flip)")
            elif confidence < exit_thresh:
                # Exit on weak signal
                self._state = SignalState.NEUTRAL
                state_changed = True
                final_action = SignalAction.HOLD
                logger.debug(f"State: BEARISH -> NEUTRAL (conf dropped to {confidence:.1%})")
            elif action == SignalAction.HOLD:
                # Maintain position with weaker signal
                final_action = SignalAction.BUY_PUTS
        
        return final_action, state_changed


class SignalCombiner:
    """
    Combines signals from multiple sources with weighted voting.
    """
    
    def __init__(
        self,
        config: SignalConfig = None,
        use_state_machine: bool = True
    ):
        self.config = config or SignalConfig()
        self.direction_analyzer = DirectionAnalyzer(self.config)
        self.confidence_calculator = ConfidenceCalculator(self.config)
        
        self.state_machine = SignalStateMachine(self.config) if use_state_machine else None
    
    def combine(
        self,
        neural_prediction: Dict,
        options_metrics: Dict = None,
        technical_signal: Dict = None,
        multi_timeframe_signal: Dict = None
    ) -> Signal:
        """
        Combine signals from multiple sources.
        
        Args:
            neural_prediction: Neural network prediction dict
            options_metrics: Options metrics dict
            technical_signal: Technical analysis signal
            multi_timeframe_signal: Multi-timeframe signal
            
        Returns:
            Combined Signal object
        """
        reasoning = []
        
        # Extract neural prediction
        predicted_return = neural_prediction.get('predicted_return', 0.0)
        predicted_volatility = neural_prediction.get('predicted_volatility', 0.01)
        direction_probs = (
            neural_prediction.get('direction_up', 0.33),
            neural_prediction.get('direction_down', 0.33),
            neural_prediction.get('direction_neutral', 0.34)
        )
        raw_confidence = neural_prediction.get('confidence', 0.5)
        
        # Analyze direction
        action, direction_confidence = self.direction_analyzer.analyze(
            direction_probs, predicted_return
        )
        
        if action != SignalAction.HOLD:
            reasoning.append(f"Neural: {action.value} (return={predicted_return:.4f})")
        
        # Calculate confidence
        confidence = self.confidence_calculator.calculate(
            raw_confidence=raw_confidence,
            direction_probabilities=direction_probs,
            predicted_return=predicted_return,
            predicted_volatility=predicted_volatility
        )
        
        # Apply multi-timeframe agreement boost
        if multi_timeframe_signal:
            agreement = multi_timeframe_signal.get('agreement_score', 0.5)
            if agreement > 0.7:
                confidence *= 1.1
                reasoning.append(f"Multi-TF agreement boost ({agreement:.1%})")
            elif agreement < 0.3:
                confidence *= 0.9
                reasoning.append(f"Multi-TF disagreement penalty ({agreement:.1%})")
        
        # Apply state machine if enabled
        if self.state_machine and self.config.use_hysteresis:
            action, state_changed = self.state_machine.update(action, confidence)
            if state_changed:
                reasoning.append(f"State: {self.state_machine.state.value}")
        
        # Final threshold check
        if action != SignalAction.HOLD and confidence < self.config.min_confidence_threshold:
            reasoning.append(f"Below threshold ({confidence:.1%} < {self.config.min_confidence_threshold:.1%})")
            action = SignalAction.HOLD
        
        return Signal(
            action=action,
            confidence=confidence,
            raw_confidence=raw_confidence,
            direction_probabilities=direction_probs,
            predicted_return=predicted_return,
            predicted_volatility=predicted_volatility,
            reasoning=reasoning,
            metadata={
                'state': self.state_machine.state.value if self.state_machine else 'N/A'
            }
        )


# Convenience function
def combine_signals(
    neural_prediction: Dict,
    options_metrics: Dict = None,
    config: SignalConfig = None
) -> Dict:
    """Combine signals and return as dictionary."""
    combiner = SignalCombiner(config)
    signal = combiner.combine(neural_prediction, options_metrics)
    return signal.to_dict()




