"""
Online Learning Module
======================

Handles online model tuning and continuous learning from predictions.

Provides:
- Prediction storage and validation
- Online model tuning from prediction errors
- Experience replay learning
- Calibration buffer updates

Usage:
    from bot_modules.online_learning import (
        OnlineLearner,
        PredictionValidator,
        ExperienceReplayBuffer
    )
"""

import logging
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Callable, Any

import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------------------
# Prediction Storage
# ------------------------------------------------------------------------------
@dataclass
class StoredPrediction:
    """A prediction stored for later validation."""
    timestamp: datetime
    price_at_prediction: float
    predicted_return: float
    confidence: float
    timeframe_minutes: int
    timeframe_name: str
    features: Optional[np.ndarray] = None
    validated: bool = False
    actual_return: Optional[float] = None
    
    @property
    def is_ready_for_validation(self) -> bool:
        """Check if enough time has passed to validate this prediction."""
        if self.validated:
            return False
        elapsed = (datetime.now() - self.timestamp).total_seconds() / 60
        return elapsed >= self.timeframe_minutes
    
    def validate(self, current_price: float) -> float:
        """Validate the prediction against current price."""
        self.actual_return = (current_price - self.price_at_prediction) / self.price_at_prediction
        self.validated = True
        return self.actual_return


class PredictionBuffer:
    """
    Buffer for storing predictions awaiting validation.
    
    Predictions are stored when made and validated once their
    timeframe has elapsed, enabling online learning.
    """
    
    def __init__(self, max_size: int = 500):
        self.predictions: List[StoredPrediction] = []
        self.max_size = max_size
    
    def store(
        self,
        timestamp: datetime,
        price: float,
        predicted_return: float,
        confidence: float,
        timeframe_minutes: int,
        timeframe_name: str,
        features: np.ndarray = None
    ):
        """Store a prediction for later validation."""
        pred = StoredPrediction(
            timestamp=timestamp,
            price_at_prediction=price,
            predicted_return=predicted_return,
            confidence=confidence,
            timeframe_minutes=timeframe_minutes,
            timeframe_name=timeframe_name,
            features=features.copy() if features is not None else None
        )
        self.predictions.append(pred)
        
        # Trim buffer
        if len(self.predictions) > self.max_size:
            self.predictions = self.predictions[-self.max_size:]
    
    def get_ready_for_validation(self, current_time: datetime = None) -> List[StoredPrediction]:
        """Get predictions that are ready to be validated."""
        if current_time is None:
            current_time = datetime.now()
        
        ready = []
        for pred in self.predictions:
            if not pred.validated:
                elapsed = (current_time - pred.timestamp).total_seconds() / 60
                if elapsed >= pred.timeframe_minutes:
                    ready.append(pred)
        return ready
    
    def remove_validated(self):
        """Remove validated predictions from buffer."""
        self.predictions = [p for p in self.predictions if not p.validated]
    
    def __len__(self) -> int:
        return len(self.predictions)


# ------------------------------------------------------------------------------
# Experience Replay
# ------------------------------------------------------------------------------
@dataclass
class Experience:
    """Training experience for replay learning."""
    features: np.ndarray
    actual_return: float
    predicted_return: float
    actual_volatility: float
    timestamp: datetime
    priority: float = 1.0


class ExperienceReplayBuffer:
    """
    Experience replay buffer for training neural networks.
    
    Stores past (features, return) pairs for diverse batch training.
    Uses priority weighting to focus on important experiences.
    """
    
    def __init__(self, max_size: int = 1000, priority_exponent: float = 0.6):
        self.buffer: deque = deque(maxlen=max_size)
        self.max_size = max_size
        self.priority_exponent = priority_exponent
    
    def add(
        self,
        features: np.ndarray,
        actual_return: float,
        predicted_return: float,
        actual_volatility: float = None,
        timestamp: datetime = None,
        priority: float = None
    ):
        """Add an experience to the buffer."""
        if actual_volatility is None:
            actual_volatility = abs(actual_return)
        if timestamp is None:
            timestamp = datetime.now()
        if priority is None:
            # Higher priority for larger prediction errors
            priority = abs(actual_return - predicted_return) ** self.priority_exponent
        
        exp = Experience(
            features=features.copy(),
            actual_return=actual_return,
            predicted_return=predicted_return,
            actual_volatility=actual_volatility,
            timestamp=timestamp,
            priority=priority
        )
        self.buffer.append(exp)
    
    def sample(self, batch_size: int, weighted: bool = True) -> List[Experience]:
        """Sample experiences from the buffer."""
        if len(self.buffer) < batch_size:
            return list(self.buffer)
        
        if weighted:
            # Priority-weighted sampling
            priorities = np.array([e.priority for e in self.buffer])
            probabilities = priorities / priorities.sum()
            indices = np.random.choice(len(self.buffer), size=batch_size, replace=False, p=probabilities)
            return [self.buffer[i] for i in indices]
        else:
            indices = np.random.choice(len(self.buffer), size=batch_size, replace=False)
            return [self.buffer[i] for i in indices]
    
    def __len__(self) -> int:
        return len(self.buffer)


# ------------------------------------------------------------------------------
# Online Learner
# ------------------------------------------------------------------------------
@dataclass
class OnlineLearningConfig:
    """Configuration for online learning."""
    min_error_threshold: float = 0.0005  # 0.05% - don't tune for smaller errors
    online_lr_scale: float = 0.1         # Use 10% of normal LR for online tuning
    max_grad_norm: float = 0.5           # Tighter gradient clipping for stability
    direction_loss_weight: float = 3.0   # Weight for direction classification loss (INCREASED from 2.0)
    extreme_return_threshold: float = 0.20  # Skip >20% moves (likely data errors)
    primary_horizon_minutes: int = 15    # Primary horizon for calibration


class OnlineLearner:
    """
    Online model tuning from prediction errors.
    
    Key method that makes predictions more accurate over time by:
    1. Storing predictions when made
    2. Validating predictions once timeframe elapses
    3. Tuning the model based on prediction errors
    4. Updating calibration buffer for confidence adjustment
    """
    
    def __init__(
        self,
        config: OnlineLearningConfig = None,
        calibration_callback: Callable = None
    ):
        self.config = config or OnlineLearningConfig()
        self.calibration_callback = calibration_callback
        
        self.prediction_buffer = PredictionBuffer()
        self.experience_buffer = ExperienceReplayBuffer()
        
        # Statistics
        self.tune_count = 0
        self.total_error = 0.0
    
    def store_prediction(
        self,
        timestamp: datetime,
        price: float,
        predicted_return: float,
        confidence: float,
        timeframe_minutes: int,
        timeframe_name: str,
        features: np.ndarray = None
    ):
        """Store a prediction for later validation and tuning."""
        # Sanity check price
        if price < 50 or price > 2000:
            logger.warning(f"Invalid price {price:.2f} for prediction storage - skipping")
            return
        
        self.prediction_buffer.store(
            timestamp=timestamp,
            price=price,
            predicted_return=predicted_return,
            confidence=confidence,
            timeframe_minutes=timeframe_minutes,
            timeframe_name=timeframe_name,
            features=features
        )
    
    def validate_and_tune(
        self,
        current_price: float,
        current_time: datetime,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        feature_buffer: List[np.ndarray],
        sequence_length: int,
        loss_fn: nn.Module,
        base_lr: float
    ) -> Dict:
        """
        Validate past predictions and tune the model.
        
        Args:
            current_price: Current market price
            current_time: Current market time
            model: Neural network model to tune
            optimizer: Model optimizer
            device: PyTorch device
            feature_buffer: Buffer of recent features
            sequence_length: Required sequence length
            loss_fn: Loss function
            base_lr: Base learning rate
            
        Returns:
            Dict with validation statistics
        """
        stats = {
            'validated_count': 0,
            'tuned_count': 0,
            'primary_horizon_count': 0,
            'total_error': 0.0
        }
        
        # Get predictions ready for validation
        ready = self.prediction_buffer.get_ready_for_validation(current_time)
        
        for pred in ready:
            # Sanity check
            if pred.price_at_prediction < 50 or pred.price_at_prediction > 2000:
                pred.validated = True  # Mark invalid to remove
                continue
            
            # Calculate actual return
            actual_return = (current_price - pred.price_at_prediction) / pred.price_at_prediction
            
            # Skip extreme returns (likely data errors)
            if abs(actual_return) > self.config.extreme_return_threshold:
                logger.warning(f"Skipping extreme return {actual_return*100:.1f}%")
                pred.validated = True
                continue
            
            pred.validate(current_price)
            stats['validated_count'] += 1
            stats['total_error'] += abs(actual_return - pred.predicted_return)
            
            # Track calibration for primary horizon
            if pred.timeframe_minutes == self.config.primary_horizon_minutes:
                stats['primary_horizon_count'] += 1
                self._update_calibration(pred, actual_return)
            
            # Tune model if we have features
            if pred.features is not None and model is not None and optimizer is not None:
                tuned = self._tune_from_error(
                    pred=pred,
                    actual_return=actual_return,
                    model=model,
                    optimizer=optimizer,
                    device=device,
                    feature_buffer=feature_buffer,
                    sequence_length=sequence_length,
                    loss_fn=loss_fn,
                    base_lr=base_lr
                )
                if tuned:
                    stats['tuned_count'] += 1
            
            # Store experience for replay
            if pred.features is not None:
                self.experience_buffer.add(
                    features=pred.features,
                    actual_return=actual_return,
                    predicted_return=pred.predicted_return
                )
        
        # Clean up validated predictions
        self.prediction_buffer.remove_validated()
        
        return stats
    
    def _tune_from_error(
        self,
        pred: StoredPrediction,
        actual_return: float,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        feature_buffer: List[np.ndarray],
        sequence_length: int,
        loss_fn: nn.Module,
        base_lr: float
    ) -> bool:
        """
        Tune the model based on a single prediction error.
        
        Returns True if tuning was performed.
        """
        error_magnitude = abs(actual_return - pred.predicted_return)
        
        # Skip small errors
        if error_magnitude < self.config.min_error_threshold:
            return False
        
        if len(feature_buffer) < sequence_length:
            return False
        
        try:
            # Use lower learning rate for online tuning
            online_lr = base_lr * self.config.online_lr_scale
            for param_group in optimizer.param_groups:
                param_group['lr'] = online_lr
            
            model.train()
            optimizer.zero_grad()
            
            # Prepare sequence
            sequence = torch.tensor(
                list(feature_buffer)[-sequence_length:],
                dtype=torch.float32
            ).unsqueeze(0).to(device)
            
            # Current features
            current_features = torch.tensor(
                pred.features,
                dtype=torch.float32
            ).unsqueeze(0).to(device)
            
            # Forward pass
            output = model(current_features, sequence)
            
            # Return loss
            target_return = torch.tensor([actual_return], dtype=torch.float32, device=device)
            return_loss = loss_fn(output["return"].squeeze(), target_return)
            
            # Direction loss
            actual_direction = 2 if actual_return > 0.001 else (0 if actual_return < -0.001 else 1)
            target_direction = torch.tensor([actual_direction], dtype=torch.long, device=device)
            direction_loss = nn.CrossEntropyLoss()(output["direction"], target_direction)
            
            total_loss = return_loss + direction_loss * self.config.direction_loss_weight
            
            # Backward pass
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=self.config.max_grad_norm)
            optimizer.step()
            
            # Restore learning rate
            for param_group in optimizer.param_groups:
                param_group['lr'] = base_lr
            
            # Update stats
            self.tune_count += 1
            self.total_error += error_magnitude
            
            # Log significant corrections
            if error_magnitude > 0.002:
                logger.info(
                    f"🎯 ONLINE TUNE [{pred.timeframe_name}]: "
                    f"pred={pred.predicted_return*100:+.2f}% "
                    f"actual={actual_return*100:+.2f}% "
                    f"error={(actual_return - pred.predicted_return)*100:+.2f}%"
                )
            
            return True
            
        except Exception as e:
            logger.debug(f"Online tuning error: {e}")
            # Restore learning rate on error
            for param_group in optimizer.param_groups:
                param_group['lr'] = base_lr
            return False
    
    def _update_calibration(self, pred: StoredPrediction, actual_return: float):
        """Update calibration buffer with direction accuracy."""
        if self.calibration_callback is None:
            return
        
        # Direction correctness (for Platt scaling)
        predicted_direction = 1 if pred.predicted_return > 0 else -1
        actual_direction = 1 if actual_return > 0 else -1
        direction_correct = predicted_direction == actual_direction
        
        self.calibration_callback(pred.confidence, direction_correct)
    
    def get_stats(self) -> Dict:
        """Get online learning statistics."""
        return {
            'tune_count': self.tune_count,
            'total_error': self.total_error,
            'avg_error': self.total_error / self.tune_count if self.tune_count > 0 else 0,
            'pending_predictions': len(self.prediction_buffer),
            'experience_buffer_size': len(self.experience_buffer)
        }


# ------------------------------------------------------------------------------
# Continuous Learning Manager
# ------------------------------------------------------------------------------
class ContinuousLearningManager:
    """
    Manages continuous learning from historical predictions.
    
    Coordinates:
    - Loading past predictions from database
    - Experience replay training
    - Model checkpointing
    """
    
    def __init__(
        self,
        db_path: str = "data/unified_options_bot.db",
        learning_batch_size: int = 10,
        max_experience_size: int = 1000
    ):
        self.db_path = db_path
        self.learning_batch_size = learning_batch_size
        self.experience_buffer = ExperienceReplayBuffer(max_size=max_experience_size)
    
    def learn_from_database(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        loss_fn: nn.Module,
        get_features_fn: Callable,
        get_price_fn: Callable,
        cutoff_minutes: int = 15
    ) -> Dict:
        """
        Learn from historical predictions stored in database.
        
        Args:
            model: Neural network model
            optimizer: Model optimizer
            device: PyTorch device
            loss_fn: Loss function
            get_features_fn: Function to get features from data
            get_price_fn: Function to get actual price at a time
            cutoff_minutes: Only use predictions older than this
            
        Returns:
            Dict with learning statistics
        """
        import sqlite3
        from datetime import datetime, timedelta
        
        stats = {'processed': 0, 'trained': 0, 'total_loss': 0.0}
        
        try:
            cutoff_time = datetime.now() - timedelta(minutes=cutoff_minutes)
            
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT timestamp, prediction_time, predicted_price_15min, current_price, confidence
                FROM unified_signals 
                WHERE prediction_time IS NOT NULL 
                AND datetime(prediction_time) <= datetime(?)
                AND predicted_price_15min IS NOT NULL
                ORDER BY timestamp DESC
                LIMIT ?
            """, [cutoff_time.strftime('%Y-%m-%d %H:%M:%S'), self.learning_batch_size])
            
            predictions = cursor.fetchall()
            conn.close()
            
            for pred in predictions:
                timestamp, prediction_time, predicted_price, current_price, confidence = pred
                stats['processed'] += 1
                
                actual_price = get_price_fn(prediction_time)
                if actual_price is None:
                    continue
                
                actual_return = (actual_price - current_price) / current_price
                predicted_return = (predicted_price - current_price) / current_price
                
                # Skip extreme returns
                if abs(actual_return) > 0.20:
                    continue
                
                # Get features and store experience
                features = get_features_fn()
                if features is not None:
                    self.experience_buffer.add(
                        features=features,
                        actual_return=actual_return,
                        predicted_return=predicted_return
                    )
                    stats['trained'] += 1
            
            return stats
            
        except Exception as e:
            logger.error(f"Database learning error: {e}")
            return stats
    
    def replay_train(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        loss_fn: nn.Module,
        batch_size: int = 8
    ) -> Optional[float]:
        """
        Train model using experience replay.
        
        Returns average loss if training occurred, None otherwise.
        """
        if len(self.experience_buffer) < batch_size:
            return None
        
        try:
            model.train()
            
            experiences = self.experience_buffer.sample(batch_size)
            total_loss = 0.0
            
            for exp in experiences:
                optimizer.zero_grad()
                
                features = torch.tensor(exp.features, dtype=torch.float32).unsqueeze(0).to(device)
                target_return = torch.tensor([exp.actual_return], dtype=torch.float32).to(device)
                
                # Forward (simplified - assumes model takes single feature vector)
                # In practice, you'd need the full sequence
                output = model.return_head(features)
                
                loss = loss_fn(output.squeeze(), target_return)
                loss.backward()
                
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                
                total_loss += loss.item()
            
            return total_loss / len(experiences)
            
        except Exception as e:
            logger.error(f"Replay training error: {e}")
            return None


__all__ = [
    'StoredPrediction',
    'PredictionBuffer',
    'Experience',
    'ExperienceReplayBuffer',
    'OnlineLearningConfig',
    'OnlineLearner',
    'ContinuousLearningManager',
]








