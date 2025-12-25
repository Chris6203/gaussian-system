#!/usr/bin/env python3
"""
Adaptive Timeframe Weight Learner

Dynamically adjusts the importance of each prediction timeframe
based on their actual accuracy over time.

Key Insight: If 30-minute predictions are more accurate than 15-minute,
we should weight them higher! Don't hardcode - learn from data!
"""

import json
import logging
from typing import Dict, List
from datetime import datetime, timedelta
import numpy as np

logger = logging.getLogger(__name__)


class AdaptiveTimeframeWeights:
    """
    Learns which prediction timeframes are most accurate and adjusts their weights.
    
    For each timeframe (15min, 30min, 1hr, 4hr, 12hr, 24hr, 48hr):
    - Tracks predictions made
    - Tracks actual outcomes
    - Calculates accuracy
    - Adjusts weight based on performance
    
    Better performing timeframes → Higher weight
    Worse performing timeframes → Lower weight
    """
    
    def __init__(self, initial_weights: Dict[str, float] = None):
        """
        Initialize with optional starting weights.
        
        Default weights if not provided:
        - 15min: 0.25 (short-term)
        - 30min: 0.20
        - 1hour: 0.15
        - 4hour: 0.15
        - 12hour: 0.10 (medium-term)
        - 24hour: 0.10
        - 48hour: 0.05 (long-term)
        """
        # UPDATED: Support all timeframe naming variations
        self.timeframes = ['5min', '10min', '15min', '20min', '30min', '40min', '45min', '50min', 
                          '1h', '1h20m', '1h40m', '2h', '2hour',
                          '1hour', '4hour', '12hour', '24hour', '48hour']
        
        # Starting weights (will be learned) - short-term weighted higher
        if initial_weights:
            self.weights = initial_weights
        else:
            self.weights = {
                '5min': 0.20,     # Most reliable
                '10min': 0.18,
                '15min': 0.15,
                '20min': 0.12,
                '30min': 0.10,
                '40min': 0.07,
                '50min': 0.05,
                '1h': 0.05,
                '1h20m': 0.03,
                '1h40m': 0.03,
                '2h': 0.02
            }
        
        # Track prediction accuracy for each timeframe
        self.predictions_history = {tf: [] for tf in self.timeframes}
        
        # Performance metrics
        self.accuracy_by_timeframe = {tf: 0.5 for tf in self.timeframes}  # Start at 50%
        self.prediction_counts = {tf: 0 for tf in self.timeframes}
        self.correct_predictions = {tf: 0 for tf in self.timeframes}
        
        # Weight adjustment parameters
        self.min_predictions_for_update = 20  # Need 20+ predictions before adjusting
        self.weight_learning_rate = 0.05  # How fast to adjust weights
        self.min_weight = 0.05  # Minimum 5% weight for any timeframe
        self.max_weight = 0.60  # Maximum 60% weight for any timeframe
        
        logger.info("🎯 Adaptive Timeframe Weights initialized")
        logger.info(f"   Starting weights: {self._format_weights()}")
    
    def _format_weights(self) -> str:
        """Format weights for logging"""
        return ", ".join([f"{tf}={w:.1%}" for tf, w in self.weights.items()])
    
    def store_prediction(self, timeframe: str, predicted_return: float, 
                        predicted_direction: str, confidence: float, timestamp: datetime):
        """
        Store a prediction to be validated later.
        
        Args:
            timeframe: '15min', '30min', '1hour', '4hour', '12hour', '24hour', '48hour'
            predicted_return: What we predicted (e.g., 0.0013 = +0.13%)
            predicted_direction: 'UP', 'DOWN', 'NEUTRAL'
            confidence: Prediction confidence (0-1)
            timestamp: When prediction was made
        """
        if timeframe not in self.timeframes:
            logger.warning(f"Unknown timeframe: {timeframe}")
            return
        
        prediction = {
            'timestamp': timestamp,
            'predicted_return': predicted_return,
            'predicted_direction': predicted_direction,
            'confidence': confidence,
            'validated': False
        }
        
        self.predictions_history[timeframe].append(prediction)
        
        # Keep only recent predictions (last 500 per timeframe)
        if len(self.predictions_history[timeframe]) > 500:
            self.predictions_history[timeframe].pop(0)
    
    def validate_predictions(self, current_price: float, timestamp: datetime):
        """
        Check past predictions to see if they were correct.
        
        For each timeframe, look back and see if the prediction was accurate.
        Update accuracy metrics accordingly.
        
        Args:
            current_price: Current market price
            timestamp: Current time
        """
        for timeframe in self.timeframes:
            # Get time window for this timeframe
            timeframe_minutes = self._get_timeframe_minutes(timeframe)
            lookback_time = timestamp - timedelta(minutes=timeframe_minutes)
            
            # Find predictions that should be validated now
            for prediction in self.predictions_history[timeframe]:
                if prediction['validated']:
                    continue
                
                prediction_time = prediction['timestamp']
                
                # Check if enough time has passed to validate
                time_since_prediction = (timestamp - prediction_time).total_seconds() / 60
                
                if time_since_prediction >= timeframe_minutes:
                    # Time to validate this prediction!
                    # (In reality, we'd need the actual price at prediction time + timeframe)
                    # For now, we'll mark it as ready and validate in the next step
                    prediction['validated'] = True
                    prediction['validation_time'] = timestamp
                    prediction['validation_price'] = current_price
    
    def update_accuracy(self, timeframe: str, was_correct: bool):
        """
        Update accuracy metrics for a timeframe after validating a prediction.
        
        Args:
            timeframe: Which timeframe's prediction
            was_correct: True if prediction was accurate
        """
        if timeframe not in self.timeframes:
            return
        
        self.prediction_counts[timeframe] += 1
        
        if was_correct:
            self.correct_predictions[timeframe] += 1
        
        # Calculate accuracy
        count = self.prediction_counts[timeframe]
        correct = self.correct_predictions[timeframe]
        
        if count > 0:
            accuracy = correct / count
            self.accuracy_by_timeframe[timeframe] = accuracy
            
            logger.debug(f"📊 {timeframe} accuracy: {correct}/{count} = {accuracy:.1%}")
    
    def adjust_weights_based_on_accuracy(self):
        """
        Adjust timeframe weights based on their prediction accuracy.
        
        Key algorithm:
        1. Calculate relative performance vs average
        2. Better than average → increase weight
        3. Worse than average → decrease weight
        4. Normalize so weights sum to 1.0
        """
        # Need minimum data before adjusting
        total_predictions = sum(self.prediction_counts.values())
        if total_predictions < self.min_predictions_for_update * len(self.timeframes):
            logger.debug(f"Not enough predictions yet ({total_predictions}) for weight adjustment")
            return False
        
        logger.info("🎯 Adjusting timeframe weights based on accuracy...")
        logger.info(f"   Current weights: {self._format_weights()}")
        
        # Get current accuracies
        accuracies = self.accuracy_by_timeframe.copy()
        avg_accuracy = np.mean(list(accuracies.values()))
        
        logger.info(f"   Accuracies by timeframe:")
        for tf in self.timeframes:
            count = self.prediction_counts[tf]
            acc = accuracies[tf]
            logger.info(f"     {tf}: {acc:.1%} ({self.correct_predictions[tf]}/{count} predictions)")
        
        logger.info(f"   Average accuracy: {avg_accuracy:.1%}")
        
        # Adjust weights based on relative performance
        new_weights = {}
        for tf in self.timeframes:
            current_weight = self.weights[tf]
            accuracy = accuracies[tf]
            
            # Calculate adjustment
            # If 10% better than average, increase weight by 5%
            # If 10% worse than average, decrease weight by 5%
            performance_vs_avg = (accuracy - avg_accuracy)
            adjustment = performance_vs_avg * self.weight_learning_rate * 10
            
            # Apply adjustment
            new_weight = current_weight + adjustment
            
            # Enforce min/max bounds
            new_weight = max(self.min_weight, min(self.max_weight, new_weight))
            new_weights[tf] = new_weight
        
        # Normalize so weights sum to 1.0
        total_weight = sum(new_weights.values())
        for tf in self.timeframes:
            new_weights[tf] = new_weights[tf] / total_weight
        
        # Log changes
        logger.info(f"   New weights:")
        for tf in self.timeframes:
            old_w = self.weights[tf]
            new_w = new_weights[tf]
            change = new_w - old_w
            logger.info(f"     {tf}: {old_w:.1%} → {new_w:.1%} ({change:+.1%})")
        
        # Update weights
        self.weights = new_weights
        
        logger.info(f"   ✅ Weights adjusted: {self._format_weights()}")
        return True
    
    def get_weights(self) -> Dict[str, float]:
        """Get current weights for each timeframe"""
        return self.weights.copy()
    
    def get_stats(self) -> Dict:
        """Get detailed statistics"""
        return {
            'weights': self.weights.copy(),
            'accuracy_by_timeframe': self.accuracy_by_timeframe.copy(),
            'prediction_counts': self.prediction_counts.copy(),
            'correct_predictions': self.correct_predictions.copy(),
            'total_predictions': sum(self.prediction_counts.values())
        }
    
    def _get_timeframe_minutes(self, timeframe: str) -> int:
        """Convert timeframe string to minutes"""
        mapping = {
            '5min': 5,
            '10min': 10,
            '15min': 15,
            '20min': 20,
            '30min': 30,
            '40min': 40,
            '50min': 50,
            '1h': 60,
            '1hour': 60,  # Legacy
            '1h20m': 80,
            '1h40m': 100,
            '2h': 120,
            '4hour': 240,
            '12hour': 720,
            '24hour': 1440,
            '48hour': 2880
        }
        return mapping.get(timeframe, 15)
    
    def save(self, path: str):
        """Save weights and stats to file"""
        data = {
            'weights': self.weights,
            'accuracy_by_timeframe': self.accuracy_by_timeframe,
            'prediction_counts': self.prediction_counts,
            'correct_predictions': self.correct_predictions,
            'last_update': datetime.now().isoformat()
        }
        
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"💾 Adaptive weights saved to {path}")
    
    def load(self, path: str):
        """Load weights and stats from file"""
        try:
            with open(path, 'r') as f:
                data = json.load(f)
            
            self.weights = data['weights']
            self.accuracy_by_timeframe = data['accuracy_by_timeframe']
            self.prediction_counts = data['prediction_counts']
            self.correct_predictions = data['correct_predictions']
            
            logger.info(f"✅ Adaptive weights loaded from {path}")
            logger.info(f"   Loaded weights: {self._format_weights()}")
            
        except FileNotFoundError:
            logger.info(f"No saved weights found at {path}, using defaults")
        except Exception as e:
            logger.error(f"Error loading adaptive weights: {e}")




