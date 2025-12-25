#!/usr/bin/env python3
"""
Predictor Manager
==================

Manages the UnifiedOptionsPredictor as a "frozen oracle" during RL training.
Provides clean API for getting predictions with uncertainty estimates.

Key Responsibilities:
1. Load/save predictor checkpoints
2. Freeze predictor weights during RL training
3. Run Monte Carlo sampling for uncertainty estimates
4. Return structured prediction outputs (mean, std, entropy)

IMPORTANT: Understanding "Frozen Predictor"
============================================

The predictor is used in THREE different contexts:

1. SUPERVISED PREDICTOR TRAINING:
   - The predictor IS trained with gradients enabled.
   - Has its own optimizer that updates predictor.parameters().
   - Use: manager.unfreeze() and manager.get_parameters() for optimizer.
   - Example: scripts/train_predictor.py, train_time_travel.py (predictor phase)

2. RL TRAINING (frozen_during_rl = true):
   - The predictor STILL RUNS every step/minute to generate predictions.
   - The predictor weights are NOT updated by RL optimizers.
   - Forward passes are wrapped in torch.no_grad() - no gradient tracking.
   - RL policies have their own separate optimizers.
   - Use: manager.freeze() before RL training begins.

3. LIVE/PAPER TRADING:
   - Pure inference mode - no training happens.
   - The predictor runs every minute just like in RL training.
   - Always uses torch.no_grad() for efficiency.

Usage:
    from backend.predictor_manager import PredictorManager
    
    manager = PredictorManager(config)
    manager.load_checkpoint("models/predictor_v2.pt")
    
    # For RL training: freeze weights (predictor still runs, but no gradients)
    manager.freeze()
    
    # Get prediction with uncertainty (works in all three contexts)
    pred = manager.predict_with_uncertainty(cur_features, seq_features)
    
    # Access uncertainty:
    pred.return_mean, pred.return_std
    pred.dir_probs, pred.dir_entropy
    
    # For supervised predictor training: unfreeze and get parameters
    manager.unfreeze()
    optimizer = torch.optim.Adam(manager.get_parameters(), lr=0.001)
"""

import os
import logging
import torch
import torch.nn as nn
import numpy as np
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, List

logger = logging.getLogger(__name__)


@dataclass
class PredictionOutput:
    """Structured prediction output with uncertainty estimates."""
    # Return prediction
    return_mean: float
    return_std: float  # Epistemic uncertainty from MC sampling
    
    # Direction prediction
    dir_probs: np.ndarray  # [DOWN, NEUTRAL, UP] probabilities
    dir_entropy: float  # Entropy of direction distribution (uncertainty)
    
    # Volatility prediction
    volatility_mean: float
    volatility_std: float
    
    # Confidence (from model)
    raw_confidence: float
    
    # Execution quality predictions
    fillability: float = 0.5
    exp_slippage: float = 0.0
    exp_ttf: float = 30.0  # Expected time to fill in seconds
    
    # Risk-adjusted return (computed)
    @property
    def risk_adjusted_return(self) -> float:
        """Return mean divided by std (Sharpe-like ratio)."""
        eps = 1e-6
        return self.return_mean / (self.return_std + eps)
    
    @property
    def is_high_uncertainty(self) -> bool:
        """Check if prediction has high uncertainty."""
        return self.return_std > 0.3 or self.dir_entropy > 0.9
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for logging/serialization."""
        return {
            'return_mean': self.return_mean,
            'return_std': self.return_std,
            'risk_adjusted_return': self.risk_adjusted_return,
            'dir_probs': self.dir_probs.tolist() if isinstance(self.dir_probs, np.ndarray) else self.dir_probs,
            'dir_entropy': self.dir_entropy,
            'volatility_mean': self.volatility_mean,
            'volatility_std': self.volatility_std,
            'raw_confidence': self.raw_confidence,
            'is_high_uncertainty': self.is_high_uncertainty,
            'fillability': self.fillability,
            'exp_slippage': self.exp_slippage,
            'exp_ttf': self.exp_ttf,
        }


class PredictorManager:
    """
    Manages the predictor model as a frozen oracle.
    
    Key features:
    1. Clean separation from RL training
    2. Monte Carlo sampling for uncertainty estimates
    3. Structured output with all uncertainty metrics
    """
    
    def __init__(
        self,
        feature_dim: int = 50,
        sequence_length: int = 60,
        mc_samples: int = 5,
        predictor_arch: str = "v2_slim_bayesian",
        device: str = "cpu",
        use_gaussian_kernels: bool = True,
    ):
        """
        Initialize the predictor manager.
        
        Args:
            feature_dim: Input feature dimension
            sequence_length: Temporal sequence length
            mc_samples: Number of Monte Carlo samples for uncertainty
            predictor_arch: Architecture variant ("v1_original" or "v2_slim_bayesian")
            device: Torch device
            use_gaussian_kernels: Whether to use RBF kernel features
        """
        self.feature_dim = feature_dim
        self.sequence_length = sequence_length
        self.mc_samples = mc_samples
        self.predictor_arch = predictor_arch
        self.device = torch.device(device)
        self.use_gaussian_kernels = use_gaussian_kernels
        
        # Checkpoint tracking (helps debug "random predictor" regressions)
        self.last_loaded_checkpoint: Optional[str] = None
        self.checkpoint_loaded: bool = False
        
        self.model: Optional[nn.Module] = None
        self.is_frozen = False
        self._initialized = False
        
        logger.info(f"📊 Predictor Manager initialized")
        logger.info(f"   Arch: {predictor_arch}, MC samples: {mc_samples}")
    
    def initialize_model(self, model: Optional[nn.Module] = None):
        """
        Initialize or set the predictor model.
        
        Args:
            model: Optional pre-created model. If None, creates new model.
        """
        if model is not None:
            self.model = model.to(self.device)
        else:
            # Create model based on architecture
            from bot_modules.neural_networks import UnifiedOptionsPredictor
            
            # Set environment variable for encoder type based on arch
            if self.predictor_arch == "v2_slim_bayesian":
                os.environ['TEMPORAL_ENCODER'] = 'tcn'
            else:
                os.environ['TEMPORAL_ENCODER'] = 'lstm'
            
            self.model = UnifiedOptionsPredictor(
                feature_dim=self.feature_dim,
                sequence_length=self.sequence_length,
                use_gaussian_kernels=self.use_gaussian_kernels,
                use_mamba=False  # TCN or LSTM based on env var
            ).to(self.device)
        
        self._initialized = True
        logger.info(f"✅ Predictor model initialized on {self.device}")
    
    def load_checkpoint(self, path: str) -> bool:
        """
        Load predictor from checkpoint.
        
        Args:
            path: Path to checkpoint file
            
        Returns:
            True if loaded successfully
        """
        if not os.path.exists(path):
            logger.warning(f"Checkpoint not found: {path}")
            self.last_loaded_checkpoint = None
            self.checkpoint_loaded = False
            return False
        
        try:
            if not self._initialized:
                self.initialize_model()
            
            checkpoint = torch.load(path, map_location=self.device, weights_only=False)
            
            # Handle different checkpoint formats
            if 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'], strict=False)
            elif 'state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['state_dict'], strict=False)
            else:
                # Assume it's just the state dict
                self.model.load_state_dict(checkpoint, strict=False)
            
            logger.info(f"✅ Loaded predictor checkpoint from {path}")
            self.last_loaded_checkpoint = path
            self.checkpoint_loaded = True
            return True
            
        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}")
            self.last_loaded_checkpoint = None
            self.checkpoint_loaded = False
            return False
    
    def save_checkpoint(self, path: str):
        """Save predictor checkpoint."""
        if not self._initialized or self.model is None:
            logger.warning("Cannot save: model not initialized")
            return
        
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
        
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'config': {
                'feature_dim': self.feature_dim,
                'sequence_length': self.sequence_length,
                'predictor_arch': self.predictor_arch,
                'use_gaussian_kernels': self.use_gaussian_kernels,
            }
        }, path)
        
        logger.info(f"💾 Saved predictor checkpoint to {path}")
    
    def freeze(self):
        """Freeze predictor weights (for RL training)."""
        if self.model is not None:
            for param in self.model.parameters():
                param.requires_grad = False
            self.model.eval()
            self.is_frozen = True
            logger.info("❄️ Predictor weights FROZEN (no gradients)")
    
    def unfreeze(self):
        """Unfreeze predictor weights (for predictor training)."""
        if self.model is not None:
            for param in self.model.parameters():
                param.requires_grad = True
            self.model.train()
            self.is_frozen = False
            logger.info("🔥 Predictor weights UNFROZEN (gradients enabled)")
    
    def predict(
        self,
        cur_features: torch.Tensor,
        seq_features: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Run single forward pass.
        
        Args:
            cur_features: Current features [B, D]
            seq_features: Sequence features [B, T, D]
            
        Returns:
            Dictionary of model outputs
        """
        if not self._initialized or self.model is None:
            raise RuntimeError("Model not initialized")
        
        self.model.eval()
        with torch.no_grad():
            return self.model(cur_features, seq_features)
    
    def predict_with_uncertainty(
        self,
        cur_features: torch.Tensor,
        seq_features: torch.Tensor,
    ) -> PredictionOutput:
        """
        Run Monte Carlo sampling for uncertainty estimates.
        
        This is the main prediction method that should be used.
        Runs K forward passes with Bayesian weight sampling to estimate
        epistemic uncertainty.
        
        Args:
            cur_features: Current features [B, D] or [D]
            seq_features: Sequence features [B, T, D] or [T, D]
            
        Returns:
            PredictionOutput with uncertainty estimates
        """
        if not self._initialized or self.model is None:
            raise RuntimeError("Model not initialized")
        
        # Ensure batch dimension
        if cur_features.dim() == 1:
            cur_features = cur_features.unsqueeze(0)
        if seq_features.dim() == 2:
            seq_features = seq_features.unsqueeze(0)
        
        # Move to device
        cur_features = cur_features.to(self.device)
        seq_features = seq_features.to(self.device)
        
        # Monte Carlo sampling
        self.model.train()  # Enable dropout/Bayesian sampling
        
        return_samples = []
        vol_samples = []
        dir_samples = []
        conf_samples = []
        fill_samples = []
        slip_samples = []
        ttf_samples = []
        
        with torch.no_grad():
            for _ in range(self.mc_samples):
                out = self.model(cur_features, seq_features)
                
                return_samples.append(out['return'].cpu().numpy())
                vol_samples.append(out['volatility'].cpu().numpy())
                dir_samples.append(torch.softmax(out['direction'], dim=-1).cpu().numpy())
                conf_samples.append(out['confidence'].cpu().numpy())
                
                # Execution quality (if available)
                if 'fillability' in out:
                    fill_samples.append(out['fillability'].cpu().numpy())
                if 'exp_slippage' in out:
                    slip_samples.append(out['exp_slippage'].cpu().numpy())
                if 'exp_ttf' in out:
                    ttf_samples.append(out['exp_ttf'].cpu().numpy())
        
        # Return to eval mode
        self.model.eval()
        
        # Compute statistics
        return_arr = np.array(return_samples)  # [K, B, 1]
        vol_arr = np.array(vol_samples)
        dir_arr = np.array(dir_samples)  # [K, B, 3]
        conf_arr = np.array(conf_samples)
        
        # Mean and std across MC samples (squeeze batch for single sample)
        return_mean = float(np.mean(return_arr))
        return_std = float(np.std(return_arr))
        
        vol_mean = float(np.mean(vol_arr))
        vol_std = float(np.std(vol_arr))
        
        # Direction: average probabilities
        dir_probs = np.mean(dir_arr, axis=0).squeeze()  # [3]
        if dir_probs.ndim == 0:
            dir_probs = np.array([0.33, 0.34, 0.33])
        
        # Direction entropy
        eps = 1e-8
        dir_entropy = float(-np.sum(dir_probs * np.log(dir_probs + eps)))
        
        # Confidence
        raw_confidence = float(np.mean(conf_arr))
        
        # Execution quality
        fillability = float(np.mean(fill_samples)) if fill_samples else 0.5
        exp_slippage = float(np.mean(slip_samples)) if slip_samples else 0.0
        exp_ttf = float(np.mean(ttf_samples)) if ttf_samples else 30.0
        
        return PredictionOutput(
            return_mean=return_mean,
            return_std=return_std,
            dir_probs=dir_probs,
            dir_entropy=dir_entropy,
            volatility_mean=vol_mean,
            volatility_std=vol_std,
            raw_confidence=raw_confidence,
            fillability=fillability,
            exp_slippage=exp_slippage,
            exp_ttf=exp_ttf,
        )
    
    def get_state_features(self, prediction: PredictionOutput) -> Dict[str, float]:
        """
        Extract features for RL state from prediction.
        
        Returns features that should be added to RL state vector:
        - return_mean, return_std
        - dir_probs (3 values)
        - dir_entropy
        - risk_adjusted_return
        
        Args:
            prediction: PredictionOutput from predict_with_uncertainty
            
        Returns:
            Dictionary of state features
        """
        return {
            'return_mean': np.clip(prediction.return_mean, -1, 1),
            'return_std': np.clip(prediction.return_std, 0, 1),
            'risk_adjusted_return': np.clip(prediction.risk_adjusted_return, -2, 2),
            'dir_prob_down': prediction.dir_probs[0] if len(prediction.dir_probs) > 0 else 0.33,
            'dir_prob_neutral': prediction.dir_probs[1] if len(prediction.dir_probs) > 1 else 0.34,
            'dir_prob_up': prediction.dir_probs[2] if len(prediction.dir_probs) > 2 else 0.33,
            'dir_entropy': np.clip(prediction.dir_entropy / np.log(3), 0, 1),  # Normalized entropy
            'raw_confidence': prediction.raw_confidence,
            'high_uncertainty': float(prediction.is_high_uncertainty),
        }
    
    def get_parameters(self) -> List[torch.nn.Parameter]:
        """Get model parameters (for optimizer creation during predictor training)."""
        if self.model is None:
            return []
        return list(self.model.parameters())


def build_state_with_uncertainty(
    base_state: Dict[str, float],
    prediction: PredictionOutput,
    inject_uncertainty: bool = True,
) -> np.ndarray:
    """
    Build complete RL state vector with uncertainty features.
    
    Args:
        base_state: Base state features (account, time, market)
        prediction: Prediction output with uncertainty
        inject_uncertainty: Whether to add uncertainty features
        
    Returns:
        Complete state vector as numpy array
    """
    features = []
    
    # Add base state features
    for key in sorted(base_state.keys()):
        features.append(float(base_state[key]))
    
    if inject_uncertainty:
        # Add prediction features with uncertainty
        features.extend([
            np.clip(prediction.return_mean, -1, 1),
            np.clip(prediction.return_std, 0, 1),
            np.clip(prediction.risk_adjusted_return, -2, 2),
            prediction.dir_probs[0] if len(prediction.dir_probs) > 0 else 0.33,
            prediction.dir_probs[1] if len(prediction.dir_probs) > 1 else 0.34,
            prediction.dir_probs[2] if len(prediction.dir_probs) > 2 else 0.33,
            np.clip(prediction.dir_entropy / np.log(3), 0, 1),
            prediction.raw_confidence,
        ])
    else:
        # Just add basic prediction
        features.extend([
            np.clip(prediction.return_mean, -1, 1),
            prediction.raw_confidence,
            prediction.dir_probs[0] if len(prediction.dir_probs) > 0 else 0.33,
            prediction.dir_probs[1] if len(prediction.dir_probs) > 1 else 0.34,
            prediction.dir_probs[2] if len(prediction.dir_probs) > 2 else 0.33,
        ])
    
    return np.array(features, dtype=np.float32)


# Export
__all__ = [
    'PredictorManager',
    'PredictionOutput',
    'build_state_with_uncertainty',
]
