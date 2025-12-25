#!/usr/bin/env python3
"""
Regime-Specific Model Manager
=============================

Trains and manages separate ML models and RL policies for different
market regimes (identified by Multi-Dimensional HMM).

Key Features:
- Per-regime neural network weights
- Per-regime RL exit/threshold policies
- Automatic regime detection and model switching
- Ensemble predictions combining regime-specific and general models
- Progressive training as regime data accumulates

Usage:
    from backend.regime_models import RegimeModelManager
    
    manager = RegimeModelManager(bot)
    
    # Train models for each regime
    manager.train_regime_models(training_data)
    
    # Get regime-specific prediction
    regime_id = manager.detect_regime(market_data)
    prediction = manager.predict(features, regime_id)

Architecture:
    ┌─────────────────────────────────────────────┐
    │              RegimeModelManager              │
    │                                              │
    │  ┌──────────┐  ┌──────────┐  ┌──────────┐  │
    │  │Low Vol   │  │Normal Vol│  │High Vol  │  │
    │  │Model     │  │Model     │  │Model     │  │
    │  └──────────┘  └──────────┘  └──────────┘  │
    │                                              │
    │  ┌──────────┐  ┌──────────┐  ┌──────────┐  │
    │  │Trending  │  │Sideways  │  │Crisis    │  │
    │  │Policy    │  │Policy    │  │Policy    │  │
    │  └──────────┘  └──────────┘  └──────────┘  │
    │                                              │
    │         ┌──────────────────┐                │
    │         │ General Ensemble │                │
    │         └──────────────────┘                │
    └─────────────────────────────────────────────┘
"""

import logging
import json
import pickle
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from collections import defaultdict
import copy

import numpy as np

logger = logging.getLogger(__name__)

# Optional PyTorch import
try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("PyTorch not available, regime models will be limited")


@dataclass
class RegimeModelConfig:
    """Configuration for a regime-specific model."""
    regime_id: str
    regime_name: str
    min_samples_for_training: int = 100
    retrain_interval_hours: int = 24
    model_weight: float = 0.6  # Weight vs general model
    use_transfer_learning: bool = True


@dataclass
class RegimeModelState:
    """State tracking for a regime-specific model."""
    regime_id: str
    training_samples: int = 0
    last_trained: Optional[datetime] = None
    metrics: Dict = field(default_factory=dict)
    is_trained: bool = False


@dataclass 
class RegimePrediction:
    """Prediction result from regime model."""
    regime_id: str
    direction: float  # -1 to 1
    confidence: float  # 0 to 1
    volatility_forecast: float
    model_contribution: float
    general_contribution: float


class RegimeModelManager:
    """
    Manages regime-specific models for improved prediction accuracy.
    
    Key concepts:
    - Each regime (defined by HMM states) has its own fine-tuned model
    - Models are initialized from the general model (transfer learning)
    - Predictions combine regime-specific and general models
    - Automatic retraining when new regime data arrives
    """
    
    # Predefined regime configurations
    REGIME_CONFIGS = {
        # Volatility regimes
        'low_vol': RegimeModelConfig(
            regime_id='low_vol',
            regime_name='Low Volatility',
            min_samples_for_training=200,
            model_weight=0.7,
        ),
        'normal_vol': RegimeModelConfig(
            regime_id='normal_vol', 
            regime_name='Normal Volatility',
            min_samples_for_training=100,
            model_weight=0.5,
        ),
        'high_vol': RegimeModelConfig(
            regime_id='high_vol',
            regime_name='High Volatility',
            min_samples_for_training=150,
            model_weight=0.65,
        ),
        
        # Trend regimes
        'trending_up': RegimeModelConfig(
            regime_id='trending_up',
            regime_name='Trending Up',
            min_samples_for_training=150,
            model_weight=0.6,
        ),
        'trending_down': RegimeModelConfig(
            regime_id='trending_down',
            regime_name='Trending Down',
            min_samples_for_training=150,
            model_weight=0.6,
        ),
        'sideways': RegimeModelConfig(
            regime_id='sideways',
            regime_name='Sideways/Choppy',
            min_samples_for_training=200,
            model_weight=0.7,  # Higher weight - sideways needs specialized handling
        ),
        
        # Combined regimes (HMM composite states)
        'low_vol_trending': RegimeModelConfig(
            regime_id='low_vol_trending',
            regime_name='Low Vol + Trending',
            min_samples_for_training=250,
            model_weight=0.75,
        ),
        'high_vol_crisis': RegimeModelConfig(
            regime_id='high_vol_crisis',
            regime_name='High Vol Crisis',
            min_samples_for_training=100,  # Less data needed - rare events
            model_weight=0.8,  # Very high weight - crisis needs special handling
        ),
    }
    
    def __init__(
        self,
        bot=None,
        models_dir: str = "models/regime_models",
        enable_transfer_learning: bool = True
    ):
        """
        Args:
            bot: UnifiedOptionsBot instance (for general model access)
            models_dir: Directory to save/load regime models
            enable_transfer_learning: Whether to initialize from general model
        """
        self.bot = bot
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.enable_transfer_learning = enable_transfer_learning
        
        # Model storage
        self._models: Dict[str, Any] = {}  # regime_id -> model
        self._policies: Dict[str, Any] = {}  # regime_id -> RL policy
        self._states: Dict[str, RegimeModelState] = {}  # regime_id -> state
        
        # Training data buffers
        self._regime_data: Dict[str, List[Dict]] = defaultdict(list)
        self._max_buffer_size = 5000
        
        # Current regime tracking
        self._current_regime: Optional[str] = None
        self._regime_history: List[Tuple[datetime, str]] = []
        
        # Initialize configs
        self.configs = {**self.REGIME_CONFIGS}
        
        # Load existing models
        self._load_models()
        
        logger.info(f"✅ RegimeModelManager initialized with {len(self._models)} models")
    
    # =========================================================================
    # REGIME DETECTION
    # =========================================================================
    
    def detect_regime(
        self,
        vix_level: float = None,
        hmm_states: Dict[str, int] = None,
        price_data: np.ndarray = None
    ) -> str:
        """
        Detect current market regime.
        
        Args:
            vix_level: Current VIX level
            hmm_states: Dict with 'trend', 'vol' HMM states
            price_data: Recent prices for trend detection
            
        Returns:
            regime_id string
        """
        # Use HMM states if available (most accurate)
        if hmm_states:
            return self._regime_from_hmm(hmm_states)
        
        # Fall back to VIX + price-based detection
        if vix_level is not None:
            vol_regime = self._classify_vol(vix_level)
            trend_regime = self._classify_trend(price_data) if price_data is not None else 'sideways'
            
            # Combine for composite regime
            if vol_regime == 'low_vol' and trend_regime in ['trending_up', 'trending_down']:
                return 'low_vol_trending'
            elif vol_regime == 'high_vol' and vix_level > 30:
                return 'high_vol_crisis'
            else:
                return vol_regime  # Default to vol regime
        
        return 'normal_vol'  # Default
    
    def _regime_from_hmm(self, hmm_states: Dict[str, int]) -> str:
        """Map HMM states to regime ID."""
        trend_state = hmm_states.get('trend', 1)
        vol_state = hmm_states.get('vol', 1)
        
        # Map states (typically 0=low/bullish, 1=neutral, 2=high/bearish)
        vol_map = {0: 'low_vol', 1: 'normal_vol', 2: 'high_vol'}
        trend_map = {0: 'trending_up', 1: 'sideways', 2: 'trending_down'}
        
        vol_regime = vol_map.get(vol_state, 'normal_vol')
        trend_regime = trend_map.get(trend_state, 'sideways')
        
        # Check for composite regimes
        if vol_regime == 'low_vol' and trend_regime in ['trending_up', 'trending_down']:
            return 'low_vol_trending'
        elif vol_regime == 'high_vol':
            return 'high_vol_crisis'
        
        # Return most relevant single regime
        if trend_regime != 'sideways':
            return trend_regime
        return vol_regime
    
    def _classify_vol(self, vix: float) -> str:
        """Classify volatility regime from VIX."""
        if vix < 15:
            return 'low_vol'
        elif vix < 22:
            return 'normal_vol'
        else:
            return 'high_vol'
    
    def _classify_trend(self, prices: np.ndarray) -> str:
        """Classify trend from price data."""
        if prices is None or len(prices) < 20:
            return 'sideways'
        
        returns = (prices[-1] / prices[-20] - 1) * 100
        
        if returns > 1.5:
            return 'trending_up'
        elif returns < -1.5:
            return 'trending_down'
        else:
            return 'sideways'
    
    def update_regime(self, regime_id: str):
        """Update current regime tracking."""
        if regime_id != self._current_regime:
            self._current_regime = regime_id
            self._regime_history.append((datetime.now(), regime_id))
            
            # Keep history bounded
            if len(self._regime_history) > 1000:
                self._regime_history = self._regime_history[-1000:]
            
            logger.info(f"🔄 Regime switched to: {regime_id}")
    
    # =========================================================================
    # MODEL TRAINING
    # =========================================================================
    
    def add_training_sample(
        self,
        regime_id: str,
        features: np.ndarray,
        target: Dict,
        timestamp: datetime = None
    ):
        """
        Add a training sample for a specific regime.
        
        Args:
            regime_id: Regime this sample belongs to
            features: Feature vector
            target: Target dict with 'direction', 'return', etc.
            timestamp: Sample timestamp
        """
        sample = {
            'features': features,
            'target': target,
            'timestamp': timestamp or datetime.now()
        }
        
        self._regime_data[regime_id].append(sample)
        
        # Maintain buffer size
        if len(self._regime_data[regime_id]) > self._max_buffer_size:
            self._regime_data[regime_id] = self._regime_data[regime_id][-self._max_buffer_size:]
        
        # Update state
        if regime_id not in self._states:
            self._states[regime_id] = RegimeModelState(regime_id=regime_id)
        self._states[regime_id].training_samples = len(self._regime_data[regime_id])
    
    def should_train(self, regime_id: str) -> bool:
        """Check if a regime model should be (re)trained."""
        config = self.configs.get(regime_id)
        if not config:
            return False
        
        state = self._states.get(regime_id)
        if not state:
            return False
        
        # Check minimum samples
        if state.training_samples < config.min_samples_for_training:
            return False
        
        # Check if never trained
        if not state.is_trained:
            return True
        
        # Check retrain interval
        if state.last_trained:
            hours_since = (datetime.now() - state.last_trained).total_seconds() / 3600
            if hours_since < config.retrain_interval_hours:
                return False
        
        return True
    
    def train_regime_model(self, regime_id: str) -> bool:
        """
        Train model for a specific regime.
        
        Args:
            regime_id: Regime to train for
            
        Returns:
            True if training succeeded
        """
        if not TORCH_AVAILABLE:
            logger.warning("PyTorch not available, skipping regime model training")
            return False
        
        config = self.configs.get(regime_id)
        if not config:
            logger.warning(f"No config for regime: {regime_id}")
            return False
        
        data = self._regime_data.get(regime_id, [])
        if len(data) < config.min_samples_for_training:
            logger.info(f"Insufficient data for {regime_id}: {len(data)}/{config.min_samples_for_training}")
            return False
        
        try:
            # Prepare training data
            X = np.array([d['features'] for d in data if d['features'] is not None])
            y = np.array([
                d['target'].get('direction', 0) if isinstance(d['target'], dict) else d['target']
                for d in data
            ])
            
            if len(X) < config.min_samples_for_training:
                return False
            
            # Create or get model
            model = self._get_or_create_model(regime_id, X.shape[1])
            
            # Train
            model = self._train_model(model, X, y, epochs=50)
            
            # Save
            self._models[regime_id] = model
            self._save_model(regime_id, model)
            
            # Update state
            if regime_id not in self._states:
                self._states[regime_id] = RegimeModelState(regime_id=regime_id)
            
            self._states[regime_id].is_trained = True
            self._states[regime_id].last_trained = datetime.now()
            self._states[regime_id].training_samples = len(data)
            
            logger.info(f"✅ Trained regime model '{regime_id}' with {len(X)} samples")
            return True
            
        except Exception as e:
            logger.error(f"Error training regime model {regime_id}: {e}")
            return False
    
    def _get_or_create_model(self, regime_id: str, input_dim: int) -> nn.Module:
        """Get existing model or create new one with transfer learning."""
        if regime_id in self._models:
            return self._models[regime_id]
        
        # Create new model
        model = self._create_regime_model(input_dim)
        
        # Transfer learning from general model if available
        if self.enable_transfer_learning and self.bot:
            try:
                general_model = self.bot.model
                if general_model is not None:
                    # Copy weights from general model (partial)
                    self._transfer_weights(general_model, model)
                    logger.info(f"🔄 Transferred weights to regime model '{regime_id}'")
            except Exception as e:
                logger.debug(f"Could not transfer weights: {e}")
        
        return model
    
    def _create_regime_model(self, input_dim: int) -> nn.Module:
        """Create a regime-specific model architecture."""
        if not TORCH_AVAILABLE:
            return None
        
        # Simpler model for regime-specific predictions
        model = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Tanh()  # Output between -1 and 1
        )
        
        return model
    
    def _transfer_weights(self, source: nn.Module, target: nn.Module):
        """Transfer compatible weights from source to target model."""
        source_state = source.state_dict()
        target_state = target.state_dict()
        
        for name, param in source_state.items():
            if name in target_state:
                if param.shape == target_state[name].shape:
                    target_state[name].copy_(param)
        
        target.load_state_dict(target_state)
    
    def _train_model(
        self,
        model: nn.Module,
        X: np.ndarray,
        y: np.ndarray,
        epochs: int = 50
    ) -> nn.Module:
        """Train the model on data."""
        if not TORCH_AVAILABLE:
            return model
        
        model.train()
        
        # Convert to tensors
        X_tensor = torch.FloatTensor(X)
        y_tensor = torch.FloatTensor(y).unsqueeze(1)
        
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.MSELoss()
        
        for epoch in range(epochs):
            optimizer.zero_grad()
            outputs = model(X_tensor)
            loss = criterion(outputs, y_tensor)
            loss.backward()
            optimizer.step()
            
            if (epoch + 1) % 10 == 0:
                logger.debug(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")
        
        model.eval()
        return model
    
    def train_all_regimes(self):
        """Train all regimes that have sufficient data."""
        trained = 0
        for regime_id in self.configs.keys():
            if self.should_train(regime_id):
                if self.train_regime_model(regime_id):
                    trained += 1
        
        logger.info(f"📚 Trained {trained} regime models")
        return trained
    
    # =========================================================================
    # PREDICTION
    # =========================================================================
    
    def predict(
        self,
        features: np.ndarray,
        regime_id: str = None
    ) -> RegimePrediction:
        """
        Get prediction using regime-specific model.
        
        Args:
            features: Feature vector
            regime_id: Regime to use (auto-detected if None)
            
        Returns:
            RegimePrediction with combined result
        """
        regime_id = regime_id or self._current_regime or 'normal_vol'
        config = self.configs.get(regime_id)
        state = self._states.get(regime_id)
        
        # Get regime model prediction
        regime_pred = None
        regime_conf = 0.0
        
        if state and state.is_trained and regime_id in self._models:
            try:
                regime_pred, regime_conf = self._predict_with_model(
                    self._models[regime_id],
                    features
                )
            except Exception as e:
                logger.debug(f"Regime model prediction error: {e}")
        
        # Get general model prediction (from bot)
        general_pred = None
        general_conf = 0.0
        
        if self.bot:
            try:
                # Use bot's prediction method
                bot_signal = self.bot.generate_unified_signal(
                    features=features,
                    skip_calibration=True  # We'll handle calibration
                )
                if bot_signal:
                    general_pred = bot_signal.get('direction', 0)
                    general_conf = bot_signal.get('confidence', 0.5)
            except Exception as e:
                logger.debug(f"General model prediction error: {e}")
        
        # Combine predictions
        if regime_pred is not None and general_pred is not None:
            # Weighted average based on config
            weight = config.model_weight if config else 0.5
            combined_direction = weight * regime_pred + (1 - weight) * general_pred
            combined_conf = weight * regime_conf + (1 - weight) * general_conf
        elif regime_pred is not None:
            combined_direction = regime_pred
            combined_conf = regime_conf
            weight = 1.0
        elif general_pred is not None:
            combined_direction = general_pred
            combined_conf = general_conf
            weight = 0.0
        else:
            combined_direction = 0.0
            combined_conf = 0.0
            weight = 0.0
        
        return RegimePrediction(
            regime_id=regime_id,
            direction=float(combined_direction),
            confidence=float(combined_conf),
            volatility_forecast=0.0,  # TODO: Add vol forecast
            model_contribution=weight if regime_pred is not None else 0,
            general_contribution=1 - weight if general_pred is not None else 0
        )
    
    def _predict_with_model(
        self,
        model: nn.Module,
        features: np.ndarray
    ) -> Tuple[float, float]:
        """Get prediction from a single model."""
        if not TORCH_AVAILABLE or model is None:
            return None, 0.0
        
        model.eval()
        
        with torch.no_grad():
            X = torch.FloatTensor(features).unsqueeze(0)
            output = model(X)
            direction = output.item()
            
            # Confidence based on distance from neutral
            confidence = min(abs(direction), 1.0)
        
        return direction, confidence
    
    # =========================================================================
    # RL POLICY MANAGEMENT
    # =========================================================================
    
    def get_regime_policy(self, regime_id: str) -> Any:
        """Get RL policy for a regime."""
        return self._policies.get(regime_id)
    
    def set_regime_policy(self, regime_id: str, policy: Any):
        """Set RL policy for a regime."""
        self._policies[regime_id] = policy
        self._save_policy(regime_id, policy)
    
    def should_use_regime_policy(self, regime_id: str) -> bool:
        """Check if should use regime-specific policy vs general."""
        state = self._states.get(regime_id)
        if not state or not state.is_trained:
            return False
        
        # Use regime policy if it has enough training
        return regime_id in self._policies
    
    # =========================================================================
    # PERSISTENCE
    # =========================================================================
    
    def _save_model(self, regime_id: str, model: Any):
        """Save regime model to disk."""
        if not TORCH_AVAILABLE or model is None:
            return
        
        path = self.models_dir / f"{regime_id}_model.pt"
        torch.save(model.state_dict(), path)
        
        # Save state
        state_path = self.models_dir / f"{regime_id}_state.json"
        state = self._states.get(regime_id)
        if state:
            with open(state_path, 'w') as f:
                json.dump({
                    'regime_id': state.regime_id,
                    'training_samples': state.training_samples,
                    'last_trained': state.last_trained.isoformat() if state.last_trained else None,
                    'is_trained': state.is_trained,
                    'metrics': state.metrics
                }, f)
    
    def _save_policy(self, regime_id: str, policy: Any):
        """Save RL policy to disk."""
        path = self.models_dir / f"{regime_id}_policy.pkl"
        with open(path, 'wb') as f:
            pickle.dump(policy, f)
    
    def _load_models(self):
        """Load all saved regime models."""
        if not self.models_dir.exists():
            return
        
        for state_file in self.models_dir.glob("*_state.json"):
            regime_id = state_file.stem.replace("_state", "")
            
            try:
                # Load state
                with open(state_file) as f:
                    state_data = json.load(f)
                
                self._states[regime_id] = RegimeModelState(
                    regime_id=regime_id,
                    training_samples=state_data.get('training_samples', 0),
                    last_trained=datetime.fromisoformat(state_data['last_trained']) 
                                 if state_data.get('last_trained') else None,
                    is_trained=state_data.get('is_trained', False),
                    metrics=state_data.get('metrics', {})
                )
                
                # Load model if PyTorch available
                if TORCH_AVAILABLE:
                    model_path = self.models_dir / f"{regime_id}_model.pt"
                    if model_path.exists():
                        # Need to know input dim - skip for now, load on demand
                        pass
                
                logger.debug(f"Loaded state for regime model '{regime_id}'")
                
            except Exception as e:
                logger.warning(f"Error loading regime model {regime_id}: {e}")
        
        # Load policies
        for policy_file in self.models_dir.glob("*_policy.pkl"):
            regime_id = policy_file.stem.replace("_policy", "")
            try:
                with open(policy_file, 'rb') as f:
                    self._policies[regime_id] = pickle.load(f)
            except Exception as e:
                logger.warning(f"Error loading policy {regime_id}: {e}")
    
    # =========================================================================
    # REPORTING
    # =========================================================================
    
    def get_status(self) -> Dict:
        """Get status of all regime models."""
        return {
            'current_regime': self._current_regime,
            'models_trained': sum(1 for s in self._states.values() if s.is_trained),
            'total_regimes': len(self.configs),
            'regime_states': {
                rid: {
                    'is_trained': state.is_trained,
                    'training_samples': state.training_samples,
                    'last_trained': state.last_trained.isoformat() if state.last_trained else None
                }
                for rid, state in self._states.items()
            },
            'regime_data_counts': {
                rid: len(data) for rid, data in self._regime_data.items()
            },
            'policies_loaded': list(self._policies.keys())
        }
    
    def print_status(self):
        """Print human-readable status."""
        status = self.get_status()
        
        print("=" * 60)
        print("REGIME MODEL STATUS")
        print("=" * 60)
        print(f"Current Regime: {status['current_regime'] or 'Unknown'}")
        print(f"Models Trained: {status['models_trained']}/{status['total_regimes']}")
        print()
        
        print("Regime States:")
        for rid, state in status['regime_states'].items():
            trained = "✅" if state['is_trained'] else "❌"
            samples = state['training_samples']
            config = self.configs.get(rid)
            min_samples = config.min_samples_for_training if config else 0
            print(f"  {trained} {rid:20s} {samples:5d}/{min_samples} samples")
        
        print()
        print("Data Buffers:")
        for rid, count in status['regime_data_counts'].items():
            bar = "█" * int(count / 100)
            print(f"  {rid:20s} {count:5d} {bar}")
        
        print("=" * 60)


# =============================================================================
# INTEGRATION HELPERS
# =============================================================================

def create_regime_model_manager(bot, config: Dict = None) -> RegimeModelManager:
    """Factory function to create RegimeModelManager."""
    models_dir = "models/regime_models"
    if config:
        models_dir = config.get('regime_models_dir', models_dir)
    
    return RegimeModelManager(
        bot=bot,
        models_dir=models_dir,
        enable_transfer_learning=config.get('enable_transfer_learning', True) if config else True
    )


def integrate_regime_models(bot, manager: RegimeModelManager):
    """
    Integrate regime model manager with bot.
    
    Patches bot methods to use regime-specific predictions.
    """
    original_generate_signal = bot.generate_unified_signal
    
    def regime_aware_signal(*args, **kwargs):
        """Wrapper that uses regime-specific model."""
        # Get original signal
        signal = original_generate_signal(*args, **kwargs)
        
        if signal and manager._current_regime:
            # Get regime-enhanced prediction
            features = kwargs.get('features') or args[0] if args else None
            if features is not None:
                try:
                    regime_pred = manager.predict(features, manager._current_regime)
                    
                    # Blend regime prediction with original
                    if regime_pred.model_contribution > 0:
                        orig_direction = signal.get('direction', 0)
                        blended = (
                            orig_direction * 0.4 + 
                            regime_pred.direction * 0.6
                        )
                        signal['direction'] = blended
                        signal['regime_enhanced'] = True
                        signal['regime_id'] = regime_pred.regime_id
                except Exception as e:
                    logger.debug(f"Regime prediction error: {e}")
        
        return signal
    
    # Patch the method
    bot.generate_unified_signal = regime_aware_signal
    bot._regime_model_manager = manager
    
    logger.info("✅ Regime models integrated with bot")









