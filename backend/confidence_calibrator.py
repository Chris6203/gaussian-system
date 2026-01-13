#!/usr/bin/env python3
"""
Confidence Calibrator - The Missing Verses

The neural network poem speaks but never hears the echo.
This module adds the missing stanzas:

1. FEEDBACK LOOP - Confidence learns from P&L outcomes
2. CONSISTENCY CHECK - Direction and confidence must agree
3. CALIBRATION TARGET - Confidence trained against actual win rate
4. TEMPORAL MEMORY - Model remembers recent prediction accuracy

Environment Variables:
- CONFIDENCE_FEEDBACK=1: Enable feedback loop (learn from outcomes)
- CONFIDENCE_CONSISTENCY=1: Enable consistency check (dir/conf agree)
- CONFIDENCE_CALIBRATION=1: Enable calibration (conf = actual win rate)
- CONFIDENCE_MEMORY=1: Enable temporal memory (recent accuracy)
- CONF_CAL_WINDOW=50: Window size for calibration
- CONF_CAL_MIN_SAMPLES=20: Minimum samples before calibrating
"""

import os
import logging
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Tuple, Deque
from collections import deque

logger = logging.getLogger(__name__)

# Configuration
CONFIDENCE_FEEDBACK = os.environ.get('CONFIDENCE_FEEDBACK', '0') == '1'
CONFIDENCE_CONSISTENCY = os.environ.get('CONFIDENCE_CONSISTENCY', '0') == '1'
CONFIDENCE_CALIBRATION = os.environ.get('CONFIDENCE_CALIBRATION', '0') == '1'
CONFIDENCE_MEMORY = os.environ.get('CONFIDENCE_MEMORY', '0') == '1'
CONF_CAL_WINDOW = int(os.environ.get('CONF_CAL_WINDOW', '50'))
CONF_CAL_MIN_SAMPLES = int(os.environ.get('CONF_CAL_MIN_SAMPLES', '20'))


@dataclass
class PredictionRecord:
    """Record of a prediction for calibration."""
    raw_confidence: float
    direction_probs: List[float]  # [DOWN, NEUTRAL, UP]
    predicted_return: float
    timestamp: float
    outcome: Optional[float] = None  # P&L when known
    was_correct: Optional[bool] = None


@dataclass
class CalibrationResult:
    """Result of confidence calibration."""
    calibrated_confidence: float
    raw_confidence: float
    adjustment_reason: str
    consistency_ok: bool
    memory_factor: float  # 0-1, based on recent accuracy
    should_trade: bool


class ConfidenceCalibrator:
    """
    The missing verses - makes the neural network hear its own echo.

    Fixes the broken confidence head by:
    1. Learning from outcomes (feedback)
    2. Checking consistency (direction agrees with confidence)
    3. Calibrating to actual win rate (calibration)
    4. Remembering recent accuracy (memory)
    """

    def __init__(self):
        self.feedback_enabled = CONFIDENCE_FEEDBACK
        self.consistency_enabled = CONFIDENCE_CONSISTENCY
        self.calibration_enabled = CONFIDENCE_CALIBRATION
        self.memory_enabled = CONFIDENCE_MEMORY
        self.window_size = CONF_CAL_WINDOW
        self.min_samples = CONF_CAL_MIN_SAMPLES

        # Prediction history for calibration
        self.predictions: Deque[PredictionRecord] = deque(maxlen=self.window_size * 2)

        # Confidence bucket calibration (10 buckets: 0-10%, 10-20%, etc.)
        self.bucket_outcomes: Dict[int, List[bool]] = {i: [] for i in range(10)}

        # Temporal memory - recent prediction accuracy
        self.recent_correct: Deque[bool] = deque(maxlen=20)

        # Learned calibration curve (maps raw conf to calibrated conf)
        self.calibration_curve: Dict[int, float] = {i: (i + 0.5) / 10 for i in range(10)}

        # Statistics
        self.total_predictions = 0
        self.total_outcomes = 0

        enabled_features = []
        if self.feedback_enabled:
            enabled_features.append("FEEDBACK")
        if self.consistency_enabled:
            enabled_features.append("CONSISTENCY")
        if self.calibration_enabled:
            enabled_features.append("CALIBRATION")
        if self.memory_enabled:
            enabled_features.append("MEMORY")

        if enabled_features:
            logger.info(f"🎵 Confidence Calibrator ENABLED: {', '.join(enabled_features)}")
            logger.info(f"   Window: {self.window_size}, Min samples: {self.min_samples}")

    def _get_bucket(self, confidence: float) -> int:
        """Get confidence bucket (0-9)."""
        bucket = int(confidence * 10)
        return min(bucket, 9)

    def _check_consistency(self, confidence: float, direction_probs: List[float],
                          predicted_return: float) -> Tuple[bool, str]:
        """
        Verse 2: Consistency Check

        The direction head and confidence head should agree.
        High confidence + uncertain direction = inconsistent.
        """
        if not self.consistency_enabled or len(direction_probs) < 3:
            return True, "consistency_disabled"

        # Get direction probabilities
        p_down, p_neutral, p_up = direction_probs[:3]

        # Check 1: High confidence should have clear direction
        max_dir_prob = max(p_down, p_up)
        if confidence > 0.6 and max_dir_prob < 0.4:
            return False, f"high_conf({confidence:.0%})_unclear_dir({max_dir_prob:.0%})"

        # Check 2: Direction should match predicted return sign
        if abs(predicted_return) > 0.001:
            if predicted_return > 0 and p_down > p_up + 0.2:
                return False, f"return_up_but_dir_down"
            if predicted_return < 0 and p_up > p_down + 0.2:
                return False, f"return_down_but_dir_up"

        # Check 3: Neutral probability should inversely correlate with confidence
        if confidence > 0.7 and p_neutral > 0.4:
            return False, f"high_conf({confidence:.0%})_high_neutral({p_neutral:.0%})"

        return True, "consistent"

    def _get_calibrated_confidence(self, raw_confidence: float) -> Tuple[float, str]:
        """
        Verse 3: Calibration

        Map raw confidence to actual observed win rate.
        """
        if not self.calibration_enabled:
            return raw_confidence, "calibration_disabled"

        bucket = self._get_bucket(raw_confidence)
        bucket_outcomes = self.bucket_outcomes.get(bucket, [])

        if len(bucket_outcomes) < self.min_samples:
            # Not enough data - use prior (assume inverted!)
            # Based on observed data: low raw confidence = high win rate
            inverted_conf = 1.0 - raw_confidence
            return inverted_conf, f"inverted_prior(n={len(bucket_outcomes)})"

        # Calculate actual win rate for this bucket
        actual_win_rate = sum(bucket_outcomes) / len(bucket_outcomes)

        # Update calibration curve
        self.calibration_curve[bucket] = actual_win_rate

        return actual_win_rate, f"calibrated(n={len(bucket_outcomes)}, wr={actual_win_rate:.0%})"

    def _get_memory_factor(self) -> Tuple[float, str]:
        """
        Verse 4: Temporal Memory

        How accurate have recent predictions been?
        If we've been wrong a lot, reduce confidence.
        """
        if not self.memory_enabled or len(self.recent_correct) < 5:
            return 1.0, "memory_disabled"

        recent_accuracy = sum(self.recent_correct) / len(self.recent_correct)

        # Scale factor: 0.5 if 0% accurate, 1.5 if 100% accurate, 1.0 if 50%
        memory_factor = 0.5 + recent_accuracy

        return memory_factor, f"memory(acc={recent_accuracy:.0%}, factor={memory_factor:.2f})"

    def calibrate(
        self,
        raw_confidence: float,
        direction_probs: List[float],
        predicted_return: float,
        action: str = 'UNKNOWN'
    ) -> CalibrationResult:
        """
        Calibrate confidence using all four verses.

        Args:
            raw_confidence: Raw model confidence (0-1)
            direction_probs: Direction probabilities [DOWN, NEUTRAL, UP]
            predicted_return: Predicted return
            action: Proposed action (BUY_CALLS, BUY_PUTS, etc.)

        Returns:
            CalibrationResult with calibrated confidence and adjustments
        """
        import time

        # Record prediction
        self.total_predictions += 1
        record = PredictionRecord(
            raw_confidence=raw_confidence,
            direction_probs=direction_probs,
            predicted_return=predicted_return,
            timestamp=time.time()
        )
        self.predictions.append(record)

        reasons = []

        # Verse 2: Consistency check
        consistency_ok, consistency_reason = self._check_consistency(
            raw_confidence, direction_probs, predicted_return
        )
        if not consistency_ok:
            reasons.append(f"INCONSISTENT:{consistency_reason}")

        # Verse 3: Calibration
        calibrated_conf, cal_reason = self._get_calibrated_confidence(raw_confidence)
        reasons.append(cal_reason)

        # Verse 4: Memory factor
        memory_factor, mem_reason = self._get_memory_factor()
        reasons.append(mem_reason)

        # Apply memory factor
        final_confidence = calibrated_conf * memory_factor
        final_confidence = max(0.0, min(1.0, final_confidence))

        # Should we trade?
        should_trade = consistency_ok and final_confidence >= 0.4

        return CalibrationResult(
            calibrated_confidence=final_confidence,
            raw_confidence=raw_confidence,
            adjustment_reason=" | ".join(reasons),
            consistency_ok=consistency_ok,
            memory_factor=memory_factor,
            should_trade=should_trade
        )

    def record_outcome(self, raw_confidence: float, actual_pnl: float,
                      direction: str = 'UNKNOWN'):
        """
        Verse 1: Feedback Loop

        The echo that completes the poem - learn from outcomes.
        """
        if not self.feedback_enabled:
            return

        self.total_outcomes += 1
        was_correct = actual_pnl > 0

        # Update bucket outcomes
        bucket = self._get_bucket(raw_confidence)
        self.bucket_outcomes[bucket].append(was_correct)

        # Keep only recent outcomes per bucket
        if len(self.bucket_outcomes[bucket]) > self.window_size:
            self.bucket_outcomes[bucket] = self.bucket_outcomes[bucket][-self.window_size:]

        # Update temporal memory
        if self.memory_enabled:
            self.recent_correct.append(was_correct)

        # Update most recent prediction record
        for record in reversed(self.predictions):
            if record.outcome is None:
                record.outcome = actual_pnl
                record.was_correct = was_correct
                break

        # Log calibration update
        bucket_wr = (sum(self.bucket_outcomes[bucket]) / len(self.bucket_outcomes[bucket])
                    if self.bucket_outcomes[bucket] else 0)
        logger.debug(f"[CONF_CAL] Bucket {bucket*10}-{(bucket+1)*10}%: "
                    f"n={len(self.bucket_outcomes[bucket])}, WR={bucket_wr:.0%}")

    def get_calibration_curve(self) -> Dict[str, float]:
        """Get the current calibration curve (raw conf -> actual win rate)."""
        curve = {}
        for bucket in range(10):
            conf_range = f"{bucket*10}-{(bucket+1)*10}%"
            outcomes = self.bucket_outcomes.get(bucket, [])
            if outcomes:
                curve[conf_range] = sum(outcomes) / len(outcomes)
            else:
                curve[conf_range] = None
        return curve

    def get_summary(self) -> str:
        """Get human-readable summary."""
        lines = [
            "Confidence Calibrator Summary",
            "============================",
            f"Total predictions: {self.total_predictions}",
            f"Total outcomes: {self.total_outcomes}",
            "",
            "Calibration Curve (raw conf -> actual WR):"
        ]

        for bucket in range(10):
            conf_range = f"{bucket*10}-{(bucket+1)*10}%"
            outcomes = self.bucket_outcomes.get(bucket, [])
            if outcomes:
                wr = sum(outcomes) / len(outcomes)
                lines.append(f"  {conf_range}: n={len(outcomes)}, actual_WR={wr:.1%}")
            else:
                lines.append(f"  {conf_range}: no data")

        if self.memory_enabled and self.recent_correct:
            recent_acc = sum(self.recent_correct) / len(self.recent_correct)
            lines.append("")
            lines.append(f"Recent accuracy (last {len(self.recent_correct)}): {recent_acc:.1%}")

        return "\n".join(lines)


# Global instance
_confidence_calibrator = None


def get_confidence_calibrator() -> ConfidenceCalibrator:
    """Get or create the global confidence calibrator."""
    global _confidence_calibrator
    if _confidence_calibrator is None:
        _confidence_calibrator = ConfidenceCalibrator()
    return _confidence_calibrator


def calibrate_confidence(
    raw_confidence: float,
    direction_probs: List[float],
    predicted_return: float,
    action: str = 'UNKNOWN'
) -> CalibrationResult:
    """
    Convenience function to calibrate confidence.

    Usage:
        result = calibrate_confidence(
            raw_confidence=0.6,
            direction_probs=[0.2, 0.3, 0.5],
            predicted_return=0.001
        )

        if result.should_trade:
            print(f"Trade with calibrated confidence: {result.calibrated_confidence:.1%}")
        else:
            print(f"Skip: {result.adjustment_reason}")
    """
    calibrator = get_confidence_calibrator()
    return calibrator.calibrate(raw_confidence, direction_probs, predicted_return, action)


def record_confidence_outcome(raw_confidence: float, actual_pnl: float,
                             direction: str = 'UNKNOWN'):
    """Record outcome for confidence calibration (feedback loop)."""
    calibrator = get_confidence_calibrator()
    calibrator.record_outcome(raw_confidence, actual_pnl, direction)
