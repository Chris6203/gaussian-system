#!/usr/bin/env python3
"""
Regime Embedding Module
========================

Provides learned embeddings for HMM regime states instead of raw scalar features.
This allows the predictor and policies to learn regime-specific patterns.

Architecture:
- One-hot regime encoding → Embedding layer → Regime vector
- Can be concatenated to feature vectors before prediction heads
- Supports mixture-of-experts gating (optional)

Usage:
    from backend.regime_embedding import RegimeEmbedding
    
    embedder = RegimeEmbedding(embedding_dim=8)
    
    # Get embedding for current regime
    regime_vec = embedder.get_embedding(trend_state=1, vol_state=2, liq_state=1)
    
    # Concatenate to features
    features_with_regime = torch.cat([features, regime_vec], dim=-1)
"""

import logging
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Tuple, Optional, List

logger = logging.getLogger(__name__)


class RegimeEmbedding(nn.Module):
    """
    Learned embedding layer for HMM regime states.
    
    Converts discrete regime states (trend, volatility, liquidity) into
    a continuous embedding vector that captures regime-specific patterns.
    
    Architecture:
    - Input: One-hot encoded regime (n_trend * n_vol * n_liq combinations)
    - Output: embedding_dim vector
    
    Can also operate in "factorized" mode where each dimension (trend, vol, liq)
    has its own embedding that are concatenated.
    """
    
    def __init__(
        self,
        embedding_dim: int = 8,
        n_trend_states: int = 3,
        n_vol_states: int = 3,
        n_liq_states: int = 3,
        factorized: bool = True,
        learnable: bool = True,
    ):
        """
        Args:
            embedding_dim: Output embedding dimension
            n_trend_states: Number of HMM trend states
            n_vol_states: Number of HMM volatility states
            n_liq_states: Number of HMM liquidity states
            factorized: If True, embed each dimension separately and concat.
                        If False, embed the combined regime (n_trend*n_vol*n_liq).
            learnable: If True, embeddings are learned. If False, use fixed one-hot.
        """
        super().__init__()
        
        self.embedding_dim = embedding_dim
        self.n_trend_states = n_trend_states
        self.n_vol_states = n_vol_states
        self.n_liq_states = n_liq_states
        self.factorized = factorized
        self.learnable = learnable
        
        if factorized:
            # Separate embeddings for each dimension
            dim_each = embedding_dim // 3
            self.trend_embed = nn.Embedding(n_trend_states, dim_each) if learnable else None
            self.vol_embed = nn.Embedding(n_vol_states, dim_each) if learnable else None
            self.liq_embed = nn.Embedding(n_liq_states, embedding_dim - 2 * dim_each) if learnable else None
            
            self.output_dim = embedding_dim
        else:
            # Single embedding for combined regime
            n_total_regimes = n_trend_states * n_vol_states * n_liq_states
            self.combined_embed = nn.Embedding(n_total_regimes, embedding_dim) if learnable else None
            self.output_dim = embedding_dim
        
        # Initialize embeddings for interpretability
        if learnable:
            self._init_embeddings()
        
        logger.info(f"📊 Regime Embedding initialized: dim={embedding_dim}, "
                   f"factorized={factorized}, learnable={learnable}")
    
    def _init_embeddings(self):
        """Initialize embeddings with interpretable structure."""
        if self.factorized:
            # Trend: negative → 0 → positive
            if self.trend_embed is not None:
                with torch.no_grad():
                    for i in range(self.n_trend_states):
                        # Map state to [-1, 0, 1] style
                        val = (i - (self.n_trend_states - 1) / 2) / max(1, (self.n_trend_states - 1) / 2)
                        self.trend_embed.weight[i] = val * 0.5
            
            # Volatility: low → normal → high
            if self.vol_embed is not None:
                with torch.no_grad():
                    for i in range(self.n_vol_states):
                        val = (i - (self.n_vol_states - 1) / 2) / max(1, (self.n_vol_states - 1) / 2)
                        self.vol_embed.weight[i] = val * 0.5
            
            # Liquidity: low → normal → high
            if self.liq_embed is not None:
                with torch.no_grad():
                    for i in range(self.n_liq_states):
                        val = (i - (self.n_liq_states - 1) / 2) / max(1, (self.n_liq_states - 1) / 2)
                        self.liq_embed.weight[i] = val * 0.5
        else:
            # For combined, use small random initialization
            if self.combined_embed is not None:
                nn.init.normal_(self.combined_embed.weight, mean=0, std=0.1)
    
    def forward(
        self,
        trend_state: torch.Tensor,
        vol_state: torch.Tensor,
        liq_state: torch.Tensor,
    ) -> torch.Tensor:
        """
        Get regime embedding.
        
        Args:
            trend_state: Tensor of trend state indices [B] or scalar
            vol_state: Tensor of volatility state indices [B] or scalar
            liq_state: Tensor of liquidity state indices [B] or scalar
            
        Returns:
            Regime embedding [B, embedding_dim]
        """
        # Ensure tensors
        if not isinstance(trend_state, torch.Tensor):
            trend_state = torch.tensor([trend_state], dtype=torch.long)
        if not isinstance(vol_state, torch.Tensor):
            vol_state = torch.tensor([vol_state], dtype=torch.long)
        if not isinstance(liq_state, torch.Tensor):
            liq_state = torch.tensor([liq_state], dtype=torch.long)
        
        # Ensure valid range
        trend_state = torch.clamp(trend_state, 0, self.n_trend_states - 1)
        vol_state = torch.clamp(vol_state, 0, self.n_vol_states - 1)
        liq_state = torch.clamp(liq_state, 0, self.n_liq_states - 1)
        
        device = next(self.parameters()).device if self.learnable else trend_state.device
        trend_state = trend_state.to(device)
        vol_state = vol_state.to(device)
        liq_state = liq_state.to(device)
        
        if self.factorized:
            if self.learnable:
                trend_vec = self.trend_embed(trend_state)
                vol_vec = self.vol_embed(vol_state)
                liq_vec = self.liq_embed(liq_state)
                return torch.cat([trend_vec, vol_vec, liq_vec], dim=-1)
            else:
                # Fixed one-hot encoding
                batch_size = trend_state.shape[0]
                trend_oh = torch.zeros(batch_size, self.n_trend_states, device=device)
                vol_oh = torch.zeros(batch_size, self.n_vol_states, device=device)
                liq_oh = torch.zeros(batch_size, self.n_liq_states, device=device)
                trend_oh.scatter_(1, trend_state.unsqueeze(1), 1)
                vol_oh.scatter_(1, vol_state.unsqueeze(1), 1)
                liq_oh.scatter_(1, liq_state.unsqueeze(1), 1)
                return torch.cat([trend_oh, vol_oh, liq_oh], dim=-1)
        else:
            # Combined regime index
            combined_idx = (trend_state * self.n_vol_states * self.n_liq_states +
                           vol_state * self.n_liq_states +
                           liq_state)
            
            if self.learnable:
                return self.combined_embed(combined_idx)
            else:
                # Fixed one-hot
                n_total = self.n_trend_states * self.n_vol_states * self.n_liq_states
                batch_size = combined_idx.shape[0]
                one_hot = torch.zeros(batch_size, n_total, device=device)
                one_hot.scatter_(1, combined_idx.unsqueeze(1), 1)
                return one_hot
    
    def get_embedding(
        self,
        trend_state: int,
        vol_state: int,
        liq_state: int,
    ) -> torch.Tensor:
        """Convenience method for single regime lookup."""
        return self.forward(
            torch.tensor([trend_state]),
            torch.tensor([vol_state]),
            torch.tensor([liq_state]),
        )
    
    def get_embedding_from_hmm_dict(self, hmm_regime: Dict) -> torch.Tensor:
        """
        Get embedding from HMM regime dictionary (as returned by MultiDimensionalHMM).
        
        Args:
            hmm_regime: Dict with keys like 'trend_state', 'volatility_state', 'liquidity_state'
                        or 'trend', 'volatility', 'liquidity' (string names)
        
        Returns:
            Regime embedding tensor
        """
        # Handle numeric state indices
        if 'trend_state' in hmm_regime:
            trend = int(hmm_regime.get('trend_state', 1))
            vol = int(hmm_regime.get('volatility_state', 1))
            liq = int(hmm_regime.get('liquidity_state', 1))
        else:
            # Handle string names
            trend_map = {'Down': 0, 'Downtrend': 0, 'Bearish': 0,
                        'No Trend': 1, 'Neutral': 1, 'Sideways': 1,
                        'Up': 2, 'Uptrend': 2, 'Bullish': 2}
            vol_map = {'Low': 0, 'Normal': 1, 'High': 2}
            liq_map = {'Low': 0, 'Normal': 1, 'High': 2}
            
            trend_name = hmm_regime.get('trend', hmm_regime.get('trend_name', 'Neutral'))
            vol_name = hmm_regime.get('volatility', hmm_regime.get('volatility_name', 'Normal'))
            liq_name = hmm_regime.get('liquidity', hmm_regime.get('liquidity_name', 'Normal'))
            
            trend = trend_map.get(trend_name, 1)
            vol = vol_map.get(vol_name, 1)
            liq = liq_map.get(liq_name, 1)
        
        return self.get_embedding(trend, vol, liq)


class RegimeAwareMLP(nn.Module):
    """
    MLP that incorporates regime embedding.
    
    Architecture:
    - Input features concatenated with regime embedding
    - Standard MLP processing
    - Can optionally use mixture-of-experts with regime-based gating
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        regime_embedding: RegimeEmbedding,
        use_moe: bool = False,
        n_experts: int = 3,
        dropout: float = 0.1,
    ):
        """
        Args:
            input_dim: Input feature dimension (before regime embedding)
            hidden_dim: Hidden layer dimension
            output_dim: Output dimension
            regime_embedding: RegimeEmbedding instance to use
            use_moe: If True, use mixture-of-experts with regime gating
            n_experts: Number of experts (only used if use_moe=True)
            dropout: Dropout rate
        """
        super().__init__()
        
        self.regime_embedding = regime_embedding
        self.use_moe = use_moe
        self.n_experts = n_experts
        
        # Input dimension includes regime embedding
        full_input_dim = input_dim + regime_embedding.output_dim
        
        if use_moe:
            # Multiple expert networks
            self.experts = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(full_input_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dim, output_dim),
                )
                for _ in range(n_experts)
            ])
            
            # Gating network (takes regime embedding as input)
            self.gate = nn.Sequential(
                nn.Linear(regime_embedding.output_dim, n_experts),
                nn.Softmax(dim=-1),
            )
        else:
            # Standard MLP
            self.mlp = nn.Sequential(
                nn.Linear(full_input_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(hidden_dim // 2, output_dim),
            )
    
    def forward(
        self,
        features: torch.Tensor,
        trend_state: torch.Tensor,
        vol_state: torch.Tensor,
        liq_state: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass with regime-aware processing.
        
        Args:
            features: Input features [B, input_dim]
            trend_state, vol_state, liq_state: Regime states
            
        Returns:
            Output tensor [B, output_dim]
        """
        # Get regime embedding
        regime_vec = self.regime_embedding(trend_state, vol_state, liq_state)
        
        # Concatenate features with regime
        x = torch.cat([features, regime_vec], dim=-1)
        
        if self.use_moe:
            # Get expert outputs
            expert_outputs = torch.stack([expert(x) for expert in self.experts], dim=1)  # [B, n_experts, output_dim]
            
            # Get gating weights based on regime
            gate_weights = self.gate(regime_vec).unsqueeze(-1)  # [B, n_experts, 1]
            
            # Weighted combination of experts
            output = (expert_outputs * gate_weights).sum(dim=1)  # [B, output_dim]
            return output
        else:
            return self.mlp(x)


def create_regime_features(
    hmm_regime: Optional[Dict],
    use_embedding: bool = True,
    embedding: Optional[RegimeEmbedding] = None,
) -> Tuple[torch.Tensor, int]:
    """
    Create regime features for state vector.
    
    Args:
        hmm_regime: HMM regime dictionary
        use_embedding: If True, use learned embedding. If False, use scalar features.
        embedding: RegimeEmbedding instance (required if use_embedding=True)
        
    Returns:
        (regime_features, feature_dim) tuple
    """
    if hmm_regime is None:
        # Default regime features
        if use_embedding and embedding is not None:
            return embedding.get_embedding(1, 1, 1), embedding.output_dim
        else:
            return torch.tensor([0.5, 0.5, 0.5, 0.5]), 4  # Default: neutral everything
    
    if use_embedding and embedding is not None:
        return embedding.get_embedding_from_hmm_dict(hmm_regime), embedding.output_dim
    else:
        # Scalar features (legacy behavior)
        trend_map = {'Down': 0.0, 'Downtrend': 0.0, 'Bearish': 0.0,
                    'No Trend': 0.5, 'Neutral': 0.5, 'Sideways': 0.5,
                    'Up': 1.0, 'Uptrend': 1.0, 'Bullish': 1.0}
        vol_map = {'Low': 0.0, 'Normal': 0.5, 'High': 1.0}
        liq_map = {'Low': 0.0, 'Normal': 0.5, 'High': 1.0}
        
        trend_name = hmm_regime.get('trend', hmm_regime.get('trend_name', 'Neutral'))
        vol_name = hmm_regime.get('volatility', hmm_regime.get('volatility_name', 'Normal'))
        liq_name = hmm_regime.get('liquidity', hmm_regime.get('liquidity_name', 'Normal'))
        conf = hmm_regime.get('confidence', hmm_regime.get('combined_confidence', 0.5))
        
        features = torch.tensor([
            trend_map.get(trend_name, 0.5),
            vol_map.get(vol_name, 0.5),
            liq_map.get(liq_name, 0.5),
            float(conf),
        ])
        return features, 4


# Export
__all__ = [
    'RegimeEmbedding',
    'RegimeAwareMLP',
    'create_regime_features',
]
