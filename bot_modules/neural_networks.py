"""
Neural Network Components
=========================

Contains all neural network architectures used by the trading bot:

- BayesianLinear: Bayesian linear layer with learnable weight uncertainty
- RBFKernelLayer: Radial Basis Function kernel layer
- TCNBlock: Temporal Convolutional Network block
- OptionsTCN: Full TCN model for options trading
- OptionsLSTM: LSTM encoder (legacy fallback)
- ResidualBlock: Residual block with Bayesian layers
- UnifiedOptionsPredictor: Main prediction model combining all components

Usage:
    from bot_modules.neural_networks import (
        BayesianLinear,
        OptionsTCN,
        UnifiedOptionsPredictor
    )
"""

import os
import math
import logging
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------------------
# Swappable Component Helpers (T5/Gemma2 Architecture Support)
# ------------------------------------------------------------------------------

def get_norm_layer(dim: int) -> nn.Module:
    """Get normalization layer based on NORM_TYPE env var."""
    norm_type = os.environ.get('NORM_TYPE', 'layernorm').lower()
    if norm_type == 'rmsnorm':
        return RMSNorm(dim)
    else:
        return nn.LayerNorm(dim)


def get_activation(in_dim: int = None, out_dim: int = None) -> nn.Module:
    """Get activation based on ACTIVATION_TYPE env var."""
    act_type = os.environ.get('ACTIVATION_TYPE', 'gelu').lower()
    if act_type == 'geglu' and in_dim and out_dim:
        return GeGLU(in_dim, out_dim)
    elif act_type == 'swiglu' and in_dim and out_dim:
        return SwiGLU(in_dim, out_dim)
    else:
        return nn.GELU()


def get_rbf_layer(input_dim: int, n_centers: int = 25, scales=None):
    """Get RBF layer based on RBF_GATED env var."""
    if os.environ.get('RBF_GATED', '0') == '1':
        return GatedRBFKernelLayer(input_dim, n_centers, scales)
    else:
        return RBFKernelLayer(input_dim, n_centers, scales)


# ------------------------------------------------------------------------------
# Modern Normalization Layers
# ------------------------------------------------------------------------------

class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization (Gemma2/LLaMA style)."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        return self.weight * (x / rms)


# ------------------------------------------------------------------------------
# Gated Activations (T5/Gemma2 Style)
# ------------------------------------------------------------------------------

class GeGLU(nn.Module):
    """Gated GELU activation (T5/Gemma2 style)."""

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.proj = nn.Linear(in_dim, out_dim * 2)
        self.out_dim = out_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, gate = self.proj(x).chunk(2, dim=-1)
        return x * F.gelu(gate)


class SwiGLU(nn.Module):
    """Swish-Gated Linear Unit (LLaMA style)."""

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.proj = nn.Linear(in_dim, out_dim * 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, gate = self.proj(x).chunk(2, dim=-1)
        return x * F.silu(gate)


# ------------------------------------------------------------------------------
# Position Encodings
# ------------------------------------------------------------------------------

class RotaryPositionEmbedding(nn.Module):
    """Rotary Position Embedding (RoPE) - LLaMA/Gemma2 style."""

    def __init__(self, dim: int, max_seq_len: int = 128):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)
        self.max_seq_len = max_seq_len
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        T = x.size(1)
        D = x.size(2)
        pos = torch.arange(T, device=x.device).float()
        sincos = torch.einsum('i,j->ij', pos, self.inv_freq)
        sin_pos, cos_pos = sincos.sin(), sincos.cos()
        rot_dim = min(self.dim * 2, D)
        x_rot = x[..., :rot_dim]
        x_pass = x[..., rot_dim:] if rot_dim < D else None
        x1, x2 = x_rot[..., 0::2], x_rot[..., 1::2]
        sin_pos = sin_pos[:, :x1.size(-1)]
        cos_pos = cos_pos[:, :x1.size(-1)]
        x1_rot = x1 * cos_pos - x2 * sin_pos
        x2_rot = x1 * sin_pos + x2 * cos_pos
        x_rotated = torch.stack([x1_rot, x2_rot], dim=-1).flatten(-2)
        if x_pass is not None:
            return torch.cat([x_rotated, x_pass], dim=-1)
        return x_rotated


class SinusoidalPositionEmbedding(nn.Module):
    """Classic sinusoidal position embedding."""

    def __init__(self, dim: int, max_seq_len: int = 128):
        super().__init__()
        pe = torch.zeros(max_seq_len, dim)
        position = torch.arange(0, max_seq_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        if dim % 2 == 0:
            pe[:, 1::2] = torch.cos(position * div_term)
        else:
            pe[:, 1::2] = torch.cos(position * div_term[:-1])
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, :x.size(1), :x.size(2)]


# ------------------------------------------------------------------------------
# Bayesian Layers
# ------------------------------------------------------------------------------
class BayesianLinear(nn.Module):
    """
    Bayesian linear layer with learnable weight uncertainty.
    
    Uses the reparameterization trick for gradients:
    - weight = weight_mu + weight_std * epsilon
    - Provides epistemic uncertainty estimates
    """
    
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.weight_mu = nn.Parameter(
            torch.randn(out_features, in_features) * 0.1)
        self.weight_logvar = nn.Parameter(
            torch.full((out_features, in_features), -3.0))
        self.bias_mu = nn.Parameter(torch.zeros(out_features))
        self.bias_logvar = nn.Parameter(torch.full((out_features,), -3.0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight_std = torch.exp(0.5 * self.weight_logvar)
        bias_std = torch.exp(0.5 * self.bias_logvar)
        weight = self.weight_mu + weight_std * torch.randn_like(weight_std)
        bias = self.bias_mu + bias_std * torch.randn_like(bias_std)
        return torch.nn.functional.linear(x, weight, bias)


# ------------------------------------------------------------------------------
# RBF Kernel Layer
# ------------------------------------------------------------------------------
class RBFKernelLayer(nn.Module):
    """
    Radial Basis Function kernel layer for non-linear feature expansion.
    
    Computes RBF similarity to learned centers at multiple scales.
    Output dimension: n_centers * len(scales)
    """
    
    def __init__(self, input_dim: int, n_centers: int = 25, scales: List[float] = None):
        super().__init__()
        self.centers = nn.Parameter(torch.randn(n_centers, input_dim) * 0.5)
        self.log_scales = nn.Parameter(
            torch.log(torch.tensor(scales or [0.1, 0.5, 1.0, 2.0, 5.0]))
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outputs = []
        # x: [B, D]; centers: [C, D]
        for log_s in self.log_scales:
            s = torch.exp(log_s)
            # squared Euclidean distance to centers -> [B, C]
            dist2 = torch.sum(
                (x.unsqueeze(1) - self.centers.unsqueeze(0)) ** 2, dim=2)
            outputs.append(torch.exp(-dist2 / (2 * s * s)))

        return torch.cat(outputs, dim=1)  # [B, C * n_scales]


class GatedRBFKernelLayer(nn.Module):
    """
    RBF Kernel with learned gate to suppress irrelevant features.

    Adds a sigmoid gate that can zero out RBF features that
    the model learns are not useful.
    """

    def __init__(self, input_dim: int, n_centers: int = 25, scales: List[float] = None):
        super().__init__()
        self.rbf = RBFKernelLayer(input_dim, n_centers, scales)
        scales = scales or [0.1, 0.5, 1.0, 2.0, 5.0]
        rbf_out_dim = n_centers * len(scales)
        self.gate = nn.Sequential(
            nn.Linear(input_dim, rbf_out_dim),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rbf_features = self.rbf(x)
        gate_weights = self.gate(x)
        return rbf_features * gate_weights


# ------------------------------------------------------------------------------
# TCN Components
# ------------------------------------------------------------------------------
class TCNBlock(nn.Module):
    """
    Temporal Convolutional Network block with dilated causal convolutions.
    
    Key features:
    - Dilated convolutions for exponentially growing receptive field
    - Causal padding (no future information leakage)
    - Residual connections for deep networks
    - Fully parallelizable (much faster than LSTM)
    
    Used by many quantitative finance firms for time series.
    """
    
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, 
                 dilation: int = 1, dropout: float = 0.2):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.dilation = dilation
        
        # Causal padding: pad only on the left side
        self.padding = (kernel_size - 1) * dilation
        
        # Two conv layers per block (standard TCN design)
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size,
                               padding=self.padding, dilation=dilation)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size,
                               padding=self.padding, dilation=dilation)
        
        # Normalization and activation
        self.norm1 = nn.BatchNorm1d(out_channels)
        self.norm2 = nn.BatchNorm1d(out_channels)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()
        
        # Residual connection (with projection if channels differ)
        self.residual = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, C, L] input tensor (batch, channels, sequence length)
        Returns:
            [B, C_out, L] output tensor
        """
        # First conv block
        out = self.conv1(x)
        out = out[:, :, :-self.padding] if self.padding > 0 else out  # Causal: remove right padding
        out = self.norm1(out)
        out = self.activation(out)
        out = self.dropout(out)
        
        # Second conv block
        out = self.conv2(out)
        out = out[:, :, :-self.padding] if self.padding > 0 else out  # Causal: remove right padding
        out = self.norm2(out)
        out = self.activation(out)
        out = self.dropout(out)
        
        # Residual connection
        res = self.residual(x)
        return out + res


class OptionsTCN(nn.Module):
    """
    Temporal Convolutional Network for options trading.
    
    Advantages over LSTM:
    - Fully parallelizable (3-5x faster training)
    - Flexible receptive field via dilations
    - No vanishing gradient issues
    - Proven in quantitative finance
    
    Architecture:
    - Stack of TCN blocks with increasing dilation
    - Dilations: 1, 2, 4, 8, 16 -> receptive field covers 60+ timesteps
    - Attention pooling for sequence aggregation
    """
    
    def __init__(self, input_dim: int, hidden_dim: int = 128, num_layers: int = 5, 
                 kernel_size: int = 3, dropout: float = 0.2):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # Input projection
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.input_norm = nn.LayerNorm(hidden_dim)
        
        # TCN blocks with exponentially increasing dilation
        # Dilation pattern: 1, 2, 4, 8, 16 gives receptive field of 2^5 * kernel_size = ~90 timesteps
        self.tcn_blocks = nn.ModuleList()
        for i in range(num_layers):
            dilation = 2 ** i
            self.tcn_blocks.append(
                TCNBlock(hidden_dim, hidden_dim, kernel_size, dilation, dropout)
            )
        
        # Final layer norm
        self.final_norm = nn.LayerNorm(hidden_dim)
        
        # Attention pooling
        self.attn_pool = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        # Context projection to output dimension
        self.ctx = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 64),
        )
        
        logger.info(f"📊 TCN initialized: {num_layers} layers, receptive field ~{2**num_layers * kernel_size} timesteps")
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, T, D] input sequence
        Returns:
            [B, 64] context vector
        """
        # Handle input shape
        if x.dim() == 2:
            x = x.unsqueeze(0)
        
        B, T, D = x.shape
        
        # Input projection: [B, T, D] -> [B, T, H]
        x = self.input_proj(x)
        x = self.input_norm(x)
        
        # Transpose for conv: [B, T, H] -> [B, H, T]
        x = x.transpose(1, 2)
        
        # Pass through TCN blocks
        for tcn in self.tcn_blocks:
            x = tcn(x)
        
        # Transpose back: [B, H, T] -> [B, T, H]
        x = x.transpose(1, 2)
        x = self.final_norm(x)
        
        # Attention pooling
        attn_weights = torch.softmax(self.attn_pool(x), dim=1)  # [B, T, 1]
        pooled = torch.sum(x * attn_weights, dim=1)  # [B, H]
        
        # Project to context
        return self.ctx(pooled)  # [B, 64]


# ------------------------------------------------------------------------------
# Transformer Components (T5/Gemma2 Style)
# ------------------------------------------------------------------------------

class TransformerBlock(nn.Module):
    """
    Single Transformer block with pre-norm (Gemma2/LLaMA style).

    Uses configurable normalization and activation.
    """

    def __init__(self, dim: int, num_heads: int, dropout: float = 0.15):
        super().__init__()
        self.norm1 = get_norm_layer(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = get_norm_layer(dim)

        # FFN with configurable activation
        act_type = os.environ.get('ACTIVATION_TYPE', 'gelu').lower()
        if act_type in ('geglu', 'swiglu'):
            self.ffn = nn.Sequential(
                get_activation(dim, dim * 4),
                nn.Dropout(dropout),
                nn.Linear(dim * 4, dim),
            )
        else:
            self.ffn = nn.Sequential(
                nn.Linear(dim, dim * 4),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(dim * 4, dim),
            )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        # Pre-norm residual (Gemma2/LLaMA style)
        h = self.norm1(x)
        h, _ = self.attn(h, h, h, attn_mask=mask)
        x = x + self.dropout(h)

        h = self.norm2(x)
        x = x + self.dropout(self.ffn(h))
        return x


class OptionsTransformer(nn.Module):
    """
    Small causal Transformer encoder for options trading.

    Alternative to OptionsTCN with attention mechanism.
    Configurable via TEMPORAL_ENCODER=transformer.

    Features:
    - Pre-norm residual connections (stable training)
    - Configurable position encoding (RoPE, sinusoidal, or none)
    - Causal masking (no future leakage)
    - Same output shape as TCN: [B, 64]
    """

    def __init__(self, input_dim: int, hidden_dim: int = 128,
                 num_layers: int = 2, num_heads: int = 4, dropout: float = 0.15):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        # Input projection
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.input_norm = get_norm_layer(hidden_dim)

        # Position encoding (configurable)
        pos_type = os.environ.get('POS_ENCODING', 'rope').lower()
        if pos_type == 'rope':
            self.pos_emb = RotaryPositionEmbedding(hidden_dim // num_heads)
            self.pos_type = 'rope'
        elif pos_type == 'sinusoidal':
            self.pos_emb = SinusoidalPositionEmbedding(hidden_dim)
            self.pos_type = 'sinusoidal'
        else:
            self.pos_emb = None
            self.pos_type = 'none'

        # Transformer layers
        self.layers = nn.ModuleList([
            TransformerBlock(hidden_dim, num_heads, dropout)
            for _ in range(num_layers)
        ])

        self.final_norm = get_norm_layer(hidden_dim)

        # Attention pooling (same as TCN for drop-in compatibility)
        self.attn_pool = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1)
        )

        # Context projection
        self.ctx = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            get_norm_layer(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 64),
        )

        logger.info(f"Transformer initialized: {num_layers} layers, {num_heads} heads, pos={self.pos_type}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, T, D] input sequence
        Returns:
            [B, 64] context vector
        """
        if x.dim() == 2:
            x = x.unsqueeze(0)

        B, T, D = x.shape

        # Input projection
        x = self.input_proj(x)
        x = self.input_norm(x)

        # Apply position encoding
        if self.pos_emb is not None:
            if self.pos_type == 'rope':
                x = self.pos_emb(x)
            else:  # sinusoidal adds to input
                x = self.pos_emb(x)

        # Causal mask: prevent attending to future positions
        mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()

        # Pass through transformer layers
        for layer in self.layers:
            x = layer(x, mask)

        x = self.final_norm(x)

        # Attention pooling
        attn_weights = torch.softmax(self.attn_pool(x), dim=1)
        pooled = torch.sum(x * attn_weights, dim=1)

        return self.ctx(pooled)  # [B, 64]


# ------------------------------------------------------------------------------
# Mamba2 State Space Model Encoder (Phase 51)
# ------------------------------------------------------------------------------

class Mamba2Block(nn.Module):
    """
    Single Mamba2 block using Structured State-space Duality (SSD).

    Pure PyTorch implementation - no custom CUDA kernels needed.

    Based on "Transformers are SSMs" paper, Mamba2 simplifies the
    selective scan by using matrix operations (SSD formulation):
    - y = SSM(A, B, C)(x) can be computed as y = Mx where M is structured
    - This allows efficient parallel computation

    Key features:
    - Data-dependent gating (selective mechanism)
    - SSD-based parallel scan
    - RMSNorm for stability
    """

    def __init__(self, d_model: int, d_state: int = 64, d_conv: int = 4, expand: int = 2):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = d_model * expand

        # Input projections
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)

        # Convolution for local context
        self.conv1d = nn.Conv1d(
            self.d_inner, self.d_inner,
            kernel_size=d_conv,
            padding=d_conv - 1,
            groups=self.d_inner,
            bias=True
        )

        # SSM parameters - data-dependent (selective)
        self.x_proj = nn.Linear(self.d_inner, d_state * 2, bias=False)  # B and C projections
        self.dt_proj = nn.Linear(self.d_inner, self.d_inner, bias=True)  # Delta (dt)

        # A is log-spaced and learned
        A = torch.arange(1, d_state + 1, dtype=torch.float32)
        self.A_log = nn.Parameter(torch.log(A.unsqueeze(0).expand(self.d_inner, -1)))  # [d_inner, d_state]

        # D is a skip connection
        self.D = nn.Parameter(torch.ones(self.d_inner))

        # Output projection
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

        # Normalization
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, T, d_model]
        Returns:
            [B, T, d_model]
        """
        residual = x
        x = self.norm(x)
        B, T, D = x.shape

        # Input projection with gating
        xz = self.in_proj(x)  # [B, T, d_inner*2]
        x_proj, z = xz.chunk(2, dim=-1)  # Each [B, T, d_inner]

        # 1D convolution for local context
        x_conv = x_proj.transpose(1, 2)  # [B, d_inner, T]
        x_conv = self.conv1d(x_conv)[:, :, :T]  # Truncate padding
        x_conv = x_conv.transpose(1, 2)  # [B, T, d_inner]
        x_conv = F.silu(x_conv)

        # SSM computation using SSD approximation
        # Project to get B, C (data-dependent)
        x_bc = self.x_proj(x_conv)  # [B, T, d_state*2]
        B_proj, C_proj = x_bc.chunk(2, dim=-1)  # Each [B, T, d_state]

        # Delta (timestep) projection
        dt = F.softplus(self.dt_proj(x_conv))  # [B, T, d_inner]

        # Get A from log representation
        A = -torch.exp(self.A_log)  # [d_inner, d_state]

        # Parallel scan using cumulative sum approximation (SSD)
        # This is a simplified version of the selective scan
        # Full Mamba2 uses optimized CUDA kernels, but this is close
        y = self._ssd_forward(x_conv, dt, A, B_proj, C_proj)

        # Skip connection with D
        y = y + self.D.unsqueeze(0).unsqueeze(0) * x_conv

        # Gate with z
        y = y * F.silu(z)

        # Output projection
        y = self.out_proj(y)

        return y + residual

    def _ssd_forward(self, x, dt, A, B, C):
        """
        Simplified SSD forward pass using associative scan approximation.

        Full Mamba2 uses the SSD (Structured State-space Duality) formulation
        which expresses SSMs as semi-separable matrices. This is a simpler
        approximation that captures the key behavior.

        Args:
            x: [B, T, d_inner] - input
            dt: [B, T, d_inner] - timestep deltas
            A: [d_inner, d_state] - state transition (negative, learned)
            B: [B, T, d_state] - input-to-state projection
            C: [B, T, d_state] - state-to-output projection

        Returns:
            [B, T, d_inner]
        """
        B_batch, T, d_inner = x.shape
        d_state = A.shape[1]

        # Discretize with dt: A_bar = exp(dt * A)
        dt_mean = dt.mean(dim=-1, keepdim=True)  # [B, T, 1]
        A_expanded = A.unsqueeze(0).unsqueeze(0)  # [1, 1, d_inner, d_state]
        decay = torch.exp(dt_mean.unsqueeze(-1) * A_expanded)  # [B, T, d_inner, d_state]

        # Recurrent computation with state space
        outputs = []
        h = torch.zeros(B_batch, d_inner, d_state, device=x.device, dtype=x.dtype)

        for t in range(T):
            # x contribution: outer product of x with B, scaled by dt
            # x[:, t]: [B, d_inner], B[:, t]: [B, d_state]
            x_b = torch.einsum('bi,bs->bis', x[:, t] * dt_mean[:, t], B[:, t])  # [B, d_inner, d_state]

            # State update: h = decay * h + x_b
            h = decay[:, t] * h + x_b  # [B, d_inner, d_state]

            # Output: contract state with C
            # h: [B, d_inner, d_state], C[:, t]: [B, d_state]
            y_t = torch.einsum('bis,bs->bi', h, C[:, t])  # [B, d_inner]
            outputs.append(y_t.unsqueeze(1))  # [B, 1, d_inner]

        return torch.cat(outputs, dim=1)  # [B, T, d_inner]


class OptionsMamba2(nn.Module):
    """
    Mamba2 State Space Model encoder for temporal sequence processing.

    Pure PyTorch implementation of Mamba2 - no custom CUDA kernels needed.

    Configurable via TEMPORAL_ENCODER=mamba2.

    Additional env vars:
    - MAMBA2_D_STATE: SSM state dimension (default: 64)
    - MAMBA2_EXPAND: Expansion factor (default: 2)
    - MAMBA2_N_LAYERS: Number of Mamba2 layers (default: 4)
    """

    def __init__(
        self,
        input_dim: int,
        d_model: int = 128,
        n_layers: int = 4,
        d_state: int = 64,
        expand: int = 2,
        dropout: float = 0.15
    ):
        super().__init__()

        # Read config from env vars
        d_state = int(os.environ.get('MAMBA2_D_STATE', str(d_state)))
        expand = int(os.environ.get('MAMBA2_EXPAND', str(expand)))
        n_layers = int(os.environ.get('MAMBA2_N_LAYERS', str(n_layers)))

        # Input projection
        self.input_proj = nn.Linear(input_dim, d_model)
        self.input_norm = nn.LayerNorm(d_model)
        self.input_dropout = nn.Dropout(dropout)

        # Mamba2 layers
        self.layers = nn.ModuleList([
            Mamba2Block(d_model, d_state=d_state, expand=expand)
            for _ in range(n_layers)
        ])

        # Output processing
        self.output_norm = nn.LayerNorm(d_model)

        # Attention pooling
        self.attn_pool = nn.Linear(d_model, 1)

        # Context projection to standard 64-dim output
        self.ctx = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        logger.info(f"⚡ Mamba2 encoder: {n_layers} layers, d_model={d_model}, d_state={d_state}, expand={expand}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Process sequence through Mamba2 layers.

        Args:
            x: Input tensor [B, T, D] where T=60 timesteps, D=feature_dim

        Returns:
            Context tensor [B, 64]
        """
        # Project input to d_model
        x = self.input_proj(x)  # [B, T, d_model]
        x = self.input_norm(x)
        x = self.input_dropout(x)

        # Process through Mamba2 layers
        for layer in self.layers:
            x = layer(x)

        x = self.output_norm(x)

        # Attention pooling
        attn_weights = torch.softmax(self.attn_pool(x), dim=1)  # [B, T, 1]
        pooled = torch.sum(x * attn_weights, dim=1)  # [B, d_model]

        return self.ctx(pooled)  # [B, 64]


def get_temporal_encoder(input_dim: int, config: dict = None) -> nn.Module:
    """
    Get temporal encoder based on TEMPORAL_ENCODER env var.

    Options:
    - tcn (default): OptionsTCN - Temporal Convolutional Network
    - transformer: OptionsTransformer - Self-attention based
    - lstm: OptionsLSTM - Bidirectional LSTM
    - mamba2: OptionsMamba2 - State Space Model (Phase 51)
    """
    config = config or {}
    encoder_type = os.environ.get('TEMPORAL_ENCODER', 'tcn').lower()

    if encoder_type == 'transformer':
        return OptionsTransformer(
            input_dim,
            hidden_dim=config.get('hidden_dim', 128),
            num_layers=config.get('num_layers', 2),
            num_heads=config.get('num_heads', 4),
            dropout=config.get('dropout', 0.15)
        )
    elif encoder_type == 'mamba2':
        return OptionsMamba2(
            input_dim,
            d_model=config.get('hidden_dim', 128),
            n_layers=config.get('num_layers', 4),
            d_state=config.get('d_state', 64),
            expand=config.get('expand', 2),
            dropout=config.get('dropout', 0.15)
        )
    elif encoder_type == 'lstm':
        return OptionsLSTM(input_dim)
    else:  # tcn (default)
        return OptionsTCN(
            input_dim,
            hidden_dim=config.get('hidden_dim', 128),
            num_layers=config.get('num_layers', 5),
            dropout=config.get('dropout', 0.2)
        )


# ------------------------------------------------------------------------------
# LSTM Components (Legacy)
# ------------------------------------------------------------------------------
class OptionsLSTM(nn.Module):
    """
    Legacy LSTM encoder - kept for backward compatibility.
    
    Enhanced LSTM with:
    - Larger hidden dimension (128 vs 64) for more capacity
    - 3 layers (vs 2) for deeper pattern learning
    - Higher dropout (0.35 vs 0.2) for better generalization
    - Layer normalization for stable training
    """
    
    def __init__(self, input_dim: int, hidden_dim: int = 128, num_layers: int = 3):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(
            input_dim, hidden_dim, num_layers, batch_first=True, dropout=0.35, bidirectional=True
        )
        # Layer normalization for stable gradients
        self.layer_norm = nn.LayerNorm(hidden_dim * 2)
        
        # Multi-head attention for better temporal focus
        self.attn = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim), 
            nn.LayerNorm(hidden_dim),
            nn.Tanh(), 
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, 1)
        )
        self.ctx = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),  # GELU often works better than ReLU for transformers/LSTMs
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, 64),  # Increased from 32 to 64
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)  # [B, T, 2H]
        out = self.layer_norm(out)  # Normalize LSTM output
        w = torch.softmax(self.attn(out), dim=1)  # [B, T, 1]
        pooled = torch.sum(out * w, dim=1)  # [B, 2H]
        return self.ctx(pooled)  # [B, 64]


# ------------------------------------------------------------------------------
# Residual Block
# ------------------------------------------------------------------------------
class ResidualBlock(nn.Module):
    """Residual block with Bayesian layers for stable deep training."""
    
    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.3):
        super().__init__()
        self.needs_projection = (in_dim != out_dim)
        
        # Main path
        self.block = nn.Sequential(
            BayesianLinear(in_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        
        # Skip connection projection (if dimensions differ)
        if self.needs_projection:
            self.skip_proj = nn.Linear(in_dim, out_dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.skip_proj(x) if self.needs_projection else x
        return self.block(x) + residual * 0.1  # Scaled residual for stability


# ------------------------------------------------------------------------------
# Main Predictor Model
# ------------------------------------------------------------------------------
class UnifiedOptionsPredictor(nn.Module):
    """
    Main prediction model combining temporal encoding with Bayesian prediction heads.
    
    Enhanced predictor with:
    - Longer sequence length (60 vs 30) for more context
    - TCN or LSTM temporal encoding (configurable)
    - TRUE residual connections for deep networks
    - Better regularization
    - Improved gradient flow
    - Execution quality prediction heads
    
    Args:
        feature_dim: Number of input features
        sequence_length: Temporal sequence length
        use_gaussian_kernels: Enable RBF kernel features
        use_mamba: Use TCN (True) or legacy LSTM (False)
    """
    
    def __init__(self, feature_dim: int, sequence_length: int = 60, use_gaussian_kernels: bool = True,
                 use_mamba: bool = False):  # Default to LSTM for stability
        super().__init__()
        self.sequence_length = sequence_length
        self.use_mamba = use_mamba
        
        # Choose temporal encoder based on environment variable
        # Options: tcn (default), transformer, lstm
        # Use the swappable get_temporal_encoder() helper
        self.temporal_encoder = get_temporal_encoder(feature_dim, {
            'hidden_dim': 128,
            'num_layers': 5 if os.environ.get('TEMPORAL_ENCODER', 'tcn').lower() == 'tcn' else 2,
            'num_heads': 4,
            'dropout': 0.2
        })
        self.encoder_type = os.environ.get('TEMPORAL_ENCODER', 'tcn').lower()
        
        self.rbf_layer = RBFKernelLayer(feature_dim) if use_gaussian_kernels else None
        rbf_dim = 25 * 5 if use_gaussian_kernels else 0
        temporal_ctx_dim = 64  # Output from TCN/LSTM
        combined = feature_dim + temporal_ctx_dim + rbf_dim
        
        # Input projection with layer norm
        self.input_proj = nn.Sequential(
            nn.Linear(combined, 256),
            nn.LayerNorm(256),
            nn.GELU(),
        )
        
        # Residual blocks for stable deep training
        self.res_block1 = ResidualBlock(256, 256, dropout=0.35)
        self.res_block2 = ResidualBlock(256, 128, dropout=0.3)
        self.res_block3 = ResidualBlock(128, 64, dropout=0.2)
        
        # Prediction heads with shared intermediate representation
        self.head_common = nn.Sequential(
            nn.Linear(64, 64),
            nn.LayerNorm(64),
            nn.GELU(),
        )
        
        # Market prediction heads
        self.return_head = BayesianLinear(64, 1)
        self.vol_head = BayesianLinear(64, 1)
        self.dir_head = BayesianLinear(64, 3)       # [DOWN, NEUTRAL, UP]
        self.conf_head = BayesianLinear(64, 1)      # 0-1 via sigmoid in forward

        # Confidence head trainability flag (Fix 1: prevent gradient leakage when untrained)
        self._confidence_trainable = True

        # Execution quality prediction heads
        self.fillability_head = BayesianLinear(64, 1)  # p(fill within T seconds at mid-peg)
        self.slippage_head = BayesianLinear(64, 1)     # expected slippage in $/contract
        self.ttf_head = BayesianLinear(64, 1)          # expected time-to-fill (seconds)

    def forward(self, cur: torch.Tensor, seq: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            cur: Current feature vector [B, D]
            seq: Sequence of historical features [B, T, D]
            
        Returns:
            Dict with predictions:
            - return: Predicted return
            - volatility: Predicted volatility
            - direction: Direction probabilities [DOWN, NEUTRAL, UP]
            - confidence: Confidence score (sigmoid)
            - fillability: Probability of fill at mid-peg
            - exp_slippage: Expected slippage in dollars
            - exp_ttf: Expected time-to-fill in seconds
        """
        # Temporal encoding with TCN or LSTM (with error handling)
        try:
            temporal_ctx = self.temporal_encoder(seq)     # [B, 64]
        except RuntimeError as e:
            # Fallback: use mean pooling if encoder fails
            logger.warning(f"Temporal encoder error: {e}, using fallback")
            # Simple mean pooling as fallback
            seq_mean = seq.mean(dim=1)  # [B, D]
            # Project to 64 dims
            temporal_ctx = torch.zeros(seq.size(0), 64, device=seq.device, dtype=seq.dtype)
            temporal_ctx[:, :min(64, seq_mean.size(-1))] = seq_mean[:, :min(64, seq_mean.size(-1))]
        
        if self.rbf_layer is not None:
            rbf = self.rbf_layer(cur)                     # [B, 25*5]
            x = torch.cat([cur, temporal_ctx, rbf], dim=-1)  # [B, D+64+RBFD]
        else:
            x = torch.cat([cur, temporal_ctx], dim=-1)

        # Input projection
        h = self.input_proj(x)
        
        # Process through residual blocks (improved gradient flow)
        h = self.res_block1(h)  # 256 -> 256 with residual
        h = self.res_block2(h)  # 256 -> 128 with projection
        h = self.res_block3(h)  # 128 -> 64 with projection
        
        # Shared head processing
        h = self.head_common(h)
        
        # Expose the final shared embedding `h` for downstream decision models.
        # This is the compact latent representation produced by the predictor backbone.
        return {
            "embedding": h,
            "return": self.return_head(h),
            "volatility": self.vol_head(h),
            "direction": self.dir_head(h),
            "confidence": torch.sigmoid(self.conf_head(h)),
            # Execution quality predictions
            "fillability": torch.sigmoid(self.fillability_head(h)),  # 0-1 probability
            "exp_slippage": self.slippage_head(h),                   # dollars (can be negative)
            "exp_ttf": torch.relu(self.ttf_head(h)),                 # non-negative seconds
        }

    def set_confidence_trainable(self, enabled: bool) -> None:
        """
        Enable or disable gradient flow through the confidence head.

        Fix 1 for broken confidence head: When TRAIN_CONFIDENCE_BCE=0 (default),
        the confidence head receives no direct loss but still gets gradient leakage
        from the shared backbone, causing it to learn inverted correlations.

        Call set_confidence_trainable(False) when not training confidence to freeze
        the head and prevent gradient leakage entirely.
        """
        self._confidence_trainable = bool(enabled)
        for p in self.conf_head.parameters():
            p.requires_grad_(self._confidence_trainable)
        logger.info(f"Confidence head trainable: {self._confidence_trainable}")


# ------------------------------------------------------------------------------
# V2 Slim Bayesian Predictor (Reduced Regularization)
# ------------------------------------------------------------------------------
class SlimResidualBlock(nn.Module):
    """
    Residual block with DETERMINISTIC Linear layers (no Bayesian in backbone).
    
    Key differences from ResidualBlock:
    - Uses standard nn.Linear instead of BayesianLinear
    - Lower dropout (0.15-0.20 vs 0.30)
    - More stable gradients, less underfitting
    """
    
    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.15):
        super().__init__()
        self.needs_projection = (in_dim != out_dim)
        
        # Main path with deterministic Linear
        self.block = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        
        # Skip connection projection (if dimensions differ)
        if self.needs_projection:
            self.skip_proj = nn.Linear(in_dim, out_dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.skip_proj(x) if self.needs_projection else x
        return self.block(x) + residual * 0.1  # Scaled residual for stability


class UnifiedOptionsPredictorV2(nn.Module):
    """
    V2 "Slim Bayesian" Predictor - Reduced regularization variant.
    
    Key differences from V1:
    - Deterministic backbone (no BayesianLinear in residual blocks)
    - Lower dropout (0.15 → 0.10 per block vs 0.35 → 0.20)
    - BayesianLinear ONLY in final prediction heads
    - More capacity for pattern learning, less underfitting
    
    Use this when:
    - V1 underfits (high bias, low variance)
    - Predictions are too uncertain / too conservative
    - Model isn't capturing market patterns well
    """
    
    def __init__(self, feature_dim: int, sequence_length: int = 60, use_gaussian_kernels: bool = True,
                 use_mamba: bool = False):
        super().__init__()
        self.sequence_length = sequence_length
        self.use_mamba = use_mamba
        
        # Temporal encoder (same as V1)
        encoder_type = os.environ.get('TEMPORAL_ENCODER', 'tcn').lower()
        
        if encoder_type == 'tcn' or (use_mamba and encoder_type != 'lstm'):
            self.temporal_encoder = OptionsTCN(feature_dim, hidden_dim=128, num_layers=5, dropout=0.15)
            self.encoder_type = 'tcn'
            logger.info("⚡ V2 Predictor: Using TCN encoder (dropout=0.15)")
        else:
            self.temporal_encoder = OptionsLSTM(feature_dim, hidden_dim=128, num_layers=3)
            self.encoder_type = 'lstm'
            logger.info("📊 V2 Predictor: Using LSTM encoder")
        
        self.rbf_layer = RBFKernelLayer(feature_dim) if use_gaussian_kernels else None
        rbf_dim = 25 * 5 if use_gaussian_kernels else 0
        temporal_ctx_dim = 64
        combined = feature_dim + temporal_ctx_dim + rbf_dim
        
        # Input projection (same as V1)
        self.input_proj = nn.Sequential(
            nn.Linear(combined, 256),
            nn.LayerNorm(256),
            nn.GELU(),
        )
        
        # SLIM residual blocks - LOWER DROPOUT, NO BAYESIAN
        self.res_block1 = SlimResidualBlock(256, 256, dropout=0.20)  # Was 0.35
        self.res_block2 = SlimResidualBlock(256, 128, dropout=0.15)  # Was 0.30
        self.res_block3 = SlimResidualBlock(128, 64, dropout=0.10)   # Was 0.20
        
        # Shared head (deterministic)
        self.head_common = nn.Sequential(
            nn.Linear(64, 64),
            nn.LayerNorm(64),
            nn.GELU(),
        )
        
        # Prediction heads - BAYESIAN ONLY HERE
        self.return_head = BayesianLinear(64, 1)
        self.vol_head = BayesianLinear(64, 1)
        self.dir_head = BayesianLinear(64, 3)       # [DOWN, NEUTRAL, UP]
        self.conf_head = BayesianLinear(64, 1)      # 0-1 via sigmoid in forward

        # Confidence head trainability flag (Fix 1: prevent gradient leakage when untrained)
        self._confidence_trainable = True

        # Execution quality heads (Bayesian for uncertainty)
        self.fillability_head = BayesianLinear(64, 1)
        self.slippage_head = BayesianLinear(64, 1)
        self.ttf_head = BayesianLinear(64, 1)
        
        logger.info("🎯 V2 Slim Bayesian Predictor initialized")
        logger.info("   - Deterministic backbone (no Bayesian in residual blocks)")
        logger.info("   - Lower dropout: 0.20 → 0.15 → 0.10")
        logger.info("   - BayesianLinear only in prediction heads")

    def forward(self, cur: torch.Tensor, seq: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass (same interface as V1)."""
        try:
            temporal_ctx = self.temporal_encoder(seq)
        except RuntimeError as e:
            logger.warning(f"Temporal encoder error: {e}, using fallback")
            seq_mean = seq.mean(dim=1)
            temporal_ctx = torch.zeros(seq.size(0), 64, device=seq.device, dtype=seq.dtype)
            temporal_ctx[:, :min(64, seq_mean.size(-1))] = seq_mean[:, :min(64, seq_mean.size(-1))]
        
        if self.rbf_layer is not None:
            rbf = self.rbf_layer(cur)
            x = torch.cat([cur, temporal_ctx, rbf], dim=-1)
        else:
            x = torch.cat([cur, temporal_ctx], dim=-1)

        h = self.input_proj(x)
        h = self.res_block1(h)
        h = self.res_block2(h)
        h = self.res_block3(h)
        h = self.head_common(h)
        
        return {
            "embedding": h,
            "return": self.return_head(h),
            "volatility": self.vol_head(h),
            "direction": self.dir_head(h),
            "confidence": torch.sigmoid(self.conf_head(h)),
            "fillability": torch.sigmoid(self.fillability_head(h)),
            "exp_slippage": self.slippage_head(h),
            "exp_ttf": torch.relu(self.ttf_head(h)),
        }

    def set_confidence_trainable(self, enabled: bool) -> None:
        """
        Enable or disable gradient flow through the confidence head.

        Fix 1 for broken confidence head: When TRAIN_CONFIDENCE_BCE=0 (default),
        the confidence head receives no direct loss but still gets gradient leakage
        from the shared backbone, causing it to learn inverted correlations.

        Call set_confidence_trainable(False) when not training confidence to freeze
        the head and prevent gradient leakage entirely.
        """
        self._confidence_trainable = bool(enabled)
        for p in self.conf_head.parameters():
            p.requires_grad_(self._confidence_trainable)
        logger.info(f"Confidence head trainable: {self._confidence_trainable}")


def create_predictor(
    arch: str = "v2_slim_bayesian",
    feature_dim: int = 50,
    sequence_length: int = 60,
    use_gaussian_kernels: bool = True,
    use_mamba: bool = False,
) -> nn.Module:
    """
    Factory function to create predictor based on architecture config.

    Args:
        arch: "v1_original", "v2_slim_bayesian", or "v3_multi_horizon"
        feature_dim: Input feature dimension
        sequence_length: Temporal sequence length
        use_gaussian_kernels: Enable RBF kernel features
        use_mamba: Use TCN (True) or LSTM (False)

    Returns:
        Predictor model instance
    """
    # Allow env var override for easy testing
    arch = os.environ.get('PREDICTOR_ARCH', arch)

    if arch == "v3_multi_horizon":
        logger.info(f"Creating V3 Multi-Horizon Predictor (horizons: 5m, 15m, 30m, 45m)")
        return UnifiedOptionsPredictorV3(
            feature_dim=feature_dim,
            sequence_length=sequence_length,
            use_gaussian_kernels=use_gaussian_kernels,
            use_mamba=use_mamba,
        )
    elif arch == "v2_slim_bayesian":
        return UnifiedOptionsPredictorV2(
            feature_dim=feature_dim,
            sequence_length=sequence_length,
            use_gaussian_kernels=use_gaussian_kernels,
            use_mamba=use_mamba,
        )
    else:  # v1_original or default
        return UnifiedOptionsPredictor(
            feature_dim=feature_dim,
            sequence_length=sequence_length,
            use_gaussian_kernels=use_gaussian_kernels,
            use_mamba=use_mamba,
        )


# ------------------------------------------------------------------------------
# Dedicated Direction Predictor - Optimized for UP/DOWN classification
# ------------------------------------------------------------------------------
class DirectionPredictor(nn.Module):
    """
    Dedicated model for direction prediction only.

    Key differences from UnifiedOptionsPredictor:
    - 100% capacity focused on direction (not shared with 6 other outputs)
    - Multi-scale temporal analysis (1m, 5m, 15m patterns)
    - Deeper architecture with attention mechanism
    - Binary output (UP vs DOWN) with confidence

    Target: 70%+ direction accuracy to achieve 60% trading win rate.
    """

    def __init__(self, feature_dim: int, sequence_length: int = 60):
        super().__init__()
        self.sequence_length = sequence_length

        # Multi-scale temporal encoders (capture different timeframe patterns)
        # Now uses get_temporal_encoder() to support Mamba2/Transformer (Phase 52 fix)
        encoder_config = {'hidden_dim': 64, 'num_layers': 3, 'dropout': 0.1}
        self.tcn_1m = get_temporal_encoder(feature_dim, encoder_config)
        self.tcn_5m = get_temporal_encoder(feature_dim, encoder_config)
        self.tcn_15m = get_temporal_encoder(feature_dim, encoder_config)

        # RBF kernel for current features
        self.rbf_layer = RBFKernelLayer(feature_dim)
        rbf_dim = 25 * 5  # 125 features

        # Combine: current features + RBF + 3 temporal scales
        temporal_dim = 64 * 3  # 3 TCN outputs
        combined_dim = feature_dim + rbf_dim + temporal_dim

        # Deep direction-focused network
        self.direction_net = nn.Sequential(
            nn.Linear(combined_dim, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(0.15),

            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.10),

            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(0.05),

            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.GELU(),
        )

        # Attention over temporal features
        self.temporal_attention = nn.MultiheadAttention(64, num_heads=4, batch_first=True)

        # Final direction heads
        self.direction_logits = nn.Linear(64, 2)  # UP vs DOWN (binary)
        self.confidence_head = nn.Linear(64, 1)   # Confidence 0-1
        self.magnitude_head = nn.Linear(64, 1)    # Expected move magnitude

        encoder_type = os.environ.get('TEMPORAL_ENCODER', 'tcn').lower()
        logger.info("🎯 DirectionPredictor initialized")
        logger.info(f"   - Multi-scale {encoder_type.upper()} (1m, 5m, 15m)")
        logger.info("   - 512→256→128→64 direction network")
        logger.info("   - Temporal attention mechanism")

    def forward(self, cur: torch.Tensor, seq: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass for direction prediction.

        Args:
            cur: Current features [B, D]
            seq: Sequence features [B, T, D] where T=60 (1 hour of 1-min bars)

        Returns:
            Dict with:
                - direction_probs: [B, 2] softmax over [DOWN, UP]
                - direction: [B] argmax (0=DOWN, 1=UP)
                - confidence: [B, 1] prediction confidence
                - magnitude: [B, 1] expected move size
        """
        batch_size = cur.size(0)

        # Multi-scale temporal encoding
        # 1-minute scale: use last 15 bars
        seq_1m = seq[:, -15:, :] if seq.size(1) >= 15 else seq
        try:
            ctx_1m = self.tcn_1m(seq_1m)
        except:
            ctx_1m = torch.zeros(batch_size, 64, device=cur.device)

        # 5-minute scale: subsample every 5 bars, use last 12 points
        if seq.size(1) >= 60:
            seq_5m = seq[:, ::5, :][:, -12:, :]
        else:
            seq_5m = seq[:, ::max(1, seq.size(1)//12), :]
        try:
            ctx_5m = self.tcn_5m(seq_5m)
        except:
            ctx_5m = torch.zeros(batch_size, 64, device=cur.device)

        # 15-minute scale: subsample every 15 bars
        if seq.size(1) >= 60:
            seq_15m = seq[:, ::15, :][:, -4:, :]
        else:
            seq_15m = seq[:, :1, :]  # Just first bar as fallback
        try:
            ctx_15m = self.tcn_15m(seq_15m)
        except:
            ctx_15m = torch.zeros(batch_size, 64, device=cur.device)

        # RBF features from current
        rbf = self.rbf_layer(cur)

        # Combine all features
        combined = torch.cat([cur, rbf, ctx_1m, ctx_5m, ctx_15m], dim=-1)

        # Direction network
        h = self.direction_net(combined)

        # Apply temporal attention (self-attention on the hidden state)
        h_seq = h.unsqueeze(1)  # [B, 1, 64]
        h_attended, _ = self.temporal_attention(h_seq, h_seq, h_seq)
        h = h_attended.squeeze(1)  # [B, 64]

        # Output heads
        direction_logits = self.direction_logits(h)
        direction_probs = torch.softmax(direction_logits, dim=-1)
        direction = torch.argmax(direction_probs, dim=-1)

        confidence = torch.sigmoid(self.confidence_head(h))
        magnitude = torch.abs(self.magnitude_head(h))  # Always positive

        return {
            'direction_probs': direction_probs,  # [B, 2] - [DOWN, UP]
            'direction': direction,               # [B] - 0=DOWN, 1=UP
            'confidence': confidence,             # [B, 1]
            'magnitude': magnitude,               # [B, 1]
            'embedding': h,                       # [B, 64] for downstream use
        }

    def predict_direction(self, cur: torch.Tensor, seq: torch.Tensor) -> tuple:
        """
        Convenience method for inference.

        Returns:
            (direction, confidence, magnitude)
            direction: 1 for UP, -1 for DOWN
            confidence: 0-1 float
            magnitude: expected move size
        """
        self.eval()
        with torch.no_grad():
            out = self.forward(cur, seq)
            direction = 1 if out['direction'].item() == 1 else -1
            confidence = out['confidence'].item()
            magnitude = out['magnitude'].item()
        return direction, confidence, magnitude


# ------------------------------------------------------------------------------
# V3 Direction-Only Predictor - Simplified for better generalization
# ------------------------------------------------------------------------------
class DirectionPredictorV3(nn.Module):
    """
    V3 Direction-Only Predictor - Maximum simplicity for better generalization.

    Key design principles:
    - NO RBF kernels (major overfitting source removed)
    - Single lightweight TCN (3 blocks vs 5)
    - Reduced sequence length (30 vs 60)
    - Binary output ONLY (UP vs DOWN, no NEUTRAL)
    - Deterministic backbone, Bayesian only on output
    - Label smoothing ready

    Target: 55-60% direction accuracy to achieve 50%+ trading win rate.
    """

    def __init__(self, feature_dim: int, sequence_length: int = 30):
        super().__init__()
        self.feature_dim = feature_dim
        self.sequence_length = sequence_length

        # Lightweight TCN: 3 blocks, dilation 1,2,4 = receptive field ~21 timesteps
        self.temporal_encoder = nn.ModuleList([
            TCNBlock(feature_dim, 64, kernel_size=3, dilation=1, dropout=0.1),
            TCNBlock(64, 64, kernel_size=3, dilation=2, dropout=0.1),
            TCNBlock(64, 64, kernel_size=3, dilation=4, dropout=0.1),
        ])

        # Attention pooling for sequence aggregation
        self.attn_pool = nn.Sequential(
            nn.Linear(64, 32),
            nn.Tanh(),
            nn.Linear(32, 1)
        )

        # NO RBF - direct concat of current features + temporal context
        # combined_dim = feature_dim + 64 (temporal context)
        combined_dim = feature_dim + 64

        # Simple feedforward backbone (deterministic)
        self.backbone = nn.Sequential(
            nn.Linear(combined_dim, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(0.15),

            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Dropout(0.10),
        )

        # Binary direction output (UP vs DOWN only)
        self.direction_head = nn.Linear(64, 2)

        # Confidence head (will be calibrated post-training via Platt scaling)
        self.confidence_head = nn.Linear(64, 1)

        logger.info("🎯 DirectionPredictorV3 initialized")
        logger.info(f"   - Feature dim: {feature_dim}, Sequence length: {sequence_length}")
        logger.info("   - NO RBF kernels (simplicity over complexity)")
        logger.info("   - Lightweight TCN: 3 blocks, dilation 1,2,4")
        logger.info("   - Binary output: UP vs DOWN only")

    def forward(self, cur: torch.Tensor, seq: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass for direction prediction.

        Args:
            cur: Current features [B, D]
            seq: Sequence features [B, T, D] where T=30 (or whatever sequence_length)

        Returns:
            Dict with:
                - direction_logits: [B, 2] raw logits for [DOWN, UP]
                - direction_probs: [B, 2] softmax probabilities
                - direction: [B] argmax (0=DOWN, 1=UP)
                - confidence: [B, 1] raw confidence (calibrate separately)
                - embedding: [B, 64] for downstream use
        """
        batch_size = cur.size(0)

        # Handle sequence shape
        if seq.dim() == 2:
            seq = seq.unsqueeze(0)

        # Truncate or pad sequence to expected length
        if seq.size(1) > self.sequence_length:
            seq = seq[:, -self.sequence_length:, :]
        elif seq.size(1) < self.sequence_length:
            # Pad with zeros on the left (older timesteps)
            pad_len = self.sequence_length - seq.size(1)
            pad = torch.zeros(seq.size(0), pad_len, seq.size(2), device=seq.device, dtype=seq.dtype)
            seq = torch.cat([pad, seq], dim=1)

        # Project sequence features to TCN input dimension
        # Need to handle if feature_dim doesn't match input
        if seq.size(2) != self.feature_dim:
            # Use linear projection if dims don't match
            seq_proj = seq[:, :, :self.feature_dim]  # Truncate
            if seq.size(2) < self.feature_dim:
                pad = torch.zeros(seq.size(0), seq.size(1), self.feature_dim - seq.size(2),
                                 device=seq.device, dtype=seq.dtype)
                seq_proj = torch.cat([seq, pad], dim=-1)
            seq = seq_proj

        # TCN encoding: [B, T, D] -> transpose -> [B, D, T] for conv
        x = seq.transpose(1, 2)  # [B, D, T]

        # Pass through TCN blocks
        for tcn_block in self.temporal_encoder:
            x = tcn_block(x)  # [B, 64, T]

        # Transpose back: [B, 64, T] -> [B, T, 64]
        x = x.transpose(1, 2)

        # Attention pooling: [B, T, 64] -> [B, 64]
        attn_weights = torch.softmax(self.attn_pool(x), dim=1)  # [B, T, 1]
        temporal_ctx = torch.sum(x * attn_weights, dim=1)  # [B, 64]

        # Concatenate current features + temporal context
        combined = torch.cat([cur, temporal_ctx], dim=-1)  # [B, D+64]

        # Backbone
        h = self.backbone(combined)  # [B, 64]

        # Direction output
        direction_logits = self.direction_head(h)  # [B, 2]
        direction_probs = torch.softmax(direction_logits, dim=-1)
        direction = torch.argmax(direction_probs, dim=-1)  # [B]

        # Confidence output (raw - calibrate separately)
        confidence = torch.sigmoid(self.confidence_head(h))  # [B, 1]

        return {
            'direction_logits': direction_logits,  # For training with label smoothing
            'direction_probs': direction_probs,    # [B, 2] - [DOWN, UP]
            'direction': direction,                 # [B] - 0=DOWN, 1=UP
            'confidence': confidence,               # [B, 1]
            'embedding': h,                         # [B, 64] for downstream use
        }

    def predict_direction(self, cur: torch.Tensor, seq: torch.Tensor) -> tuple:
        """
        Convenience method for inference.

        Returns:
            (direction, confidence, up_prob)
            direction: 1 for UP, -1 for DOWN
            confidence: 0-1 float
            up_prob: probability of UP move
        """
        self.eval()
        with torch.no_grad():
            out = self.forward(cur, seq)
            direction = 1 if out['direction'].item() == 1 else -1
            confidence = out['confidence'].item()
            up_prob = out['direction_probs'][0, 1].item()  # P(UP)
        return direction, confidence, up_prob


# ------------------------------------------------------------------------------
# Multi-Horizon Predictor V3
# ------------------------------------------------------------------------------
class UnifiedOptionsPredictorV3(nn.Module):
    """
    V3 Predictor with Multi-Horizon Prediction Heads.

    Key improvements over V2:
    - Predicts returns and directions for {5m, 15m, 30m, 45m} horizons
    - RL policy can decide based on which horizon has edge
    - Solves horizon misalignment problem (prediction expires before exit)
    """

    HORIZONS = [5, 15, 30, 45]  # Minutes

    def __init__(self, feature_dim: int, sequence_length: int = 60, use_gaussian_kernels: bool = False,
                 use_mamba: bool = False):
        super().__init__()
        self.sequence_length = sequence_length
        self.use_mamba = use_mamba
        self.feature_dim = feature_dim

        # Temporal encoder
        encoder_type = os.environ.get('TEMPORAL_ENCODER', 'tcn').lower()

        if encoder_type == 'tcn' or (use_mamba and encoder_type != 'lstm'):
            self.temporal_encoder = OptionsTCN(feature_dim, hidden_dim=128, num_layers=5, dropout=0.15)
            self.encoder_type = 'tcn'
        else:
            self.temporal_encoder = OptionsLSTM(feature_dim, hidden_dim=128, num_layers=3)
            self.encoder_type = 'lstm'

        self.rbf_layer = RBFKernelLayer(feature_dim) if use_gaussian_kernels else None
        rbf_dim = 25 * 5 if use_gaussian_kernels else 0
        temporal_ctx_dim = 64
        combined = feature_dim + temporal_ctx_dim + rbf_dim

        # Shared backbone
        self.input_proj = nn.Sequential(
            nn.Linear(combined, 256),
            nn.LayerNorm(256),
            nn.GELU(),
        )

        self.res_block1 = SlimResidualBlock(256, 256, dropout=0.20)
        self.res_block2 = SlimResidualBlock(256, 128, dropout=0.15)
        self.res_block3 = SlimResidualBlock(128, 64, dropout=0.10)

        self.head_common = nn.Sequential(
            nn.Linear(64, 64),
            nn.LayerNorm(64),
            nn.GELU(),
        )

        # Per-horizon prediction heads
        self.horizon_heads = nn.ModuleDict()
        for h in self.HORIZONS:
            self.horizon_heads[f'return_{h}m'] = BayesianLinear(64, 1)
            self.horizon_heads[f'direction_{h}m'] = BayesianLinear(64, 3)
            self.horizon_heads[f'confidence_{h}m'] = BayesianLinear(64, 1)

        # Shared execution quality heads
        self.fillability_head = BayesianLinear(64, 1)
        self.slippage_head = BayesianLinear(64, 1)
        self.ttf_head = BayesianLinear(64, 1)
        self.vol_head = BayesianLinear(64, 1)

        # Confidence head trainability flag (Fix 1: prevent gradient leakage when untrained)
        self._confidence_trainable = True

        logger.info('V3 Multi-Horizon Predictor: horizons=' + str(self.HORIZONS))

    def forward(self, cur: torch.Tensor, seq: torch.Tensor) -> Dict[str, torch.Tensor]:
        try:
            temporal_ctx = self.temporal_encoder(seq)
        except RuntimeError as e:
            seq_mean = seq.mean(dim=1)
            temporal_ctx = torch.zeros(seq.size(0), 64, device=seq.device, dtype=seq.dtype)
            temporal_ctx[:, :min(64, seq_mean.size(-1))] = seq_mean[:, :min(64, seq_mean.size(-1))]

        if self.rbf_layer is not None:
            rbf = self.rbf_layer(cur)
            x = torch.cat([cur, temporal_ctx, rbf], dim=-1)
        else:
            x = torch.cat([cur, temporal_ctx], dim=-1)

        h = self.input_proj(x)
        h = self.res_block1(h)
        h = self.res_block2(h)
        h = self.res_block3(h)
        h = self.head_common(h)

        result = {'embedding': h}

        for horizon in self.HORIZONS:
            result[f'return_{horizon}m'] = self.horizon_heads[f'return_{horizon}m'](h)
            result[f'direction_{horizon}m'] = self.horizon_heads[f'direction_{horizon}m'](h)
            result[f'confidence_{horizon}m'] = torch.sigmoid(self.horizon_heads[f'confidence_{horizon}m'](h))

        # Default horizon - configurable via V3_DEFAULT_HORIZON env var
        # Options: 5, 15, 30, 45 (minutes)
        # Shorter horizons (5m) may have higher accuracy for short hold times
        default_horizon = int(os.environ.get('V3_DEFAULT_HORIZON', '15'))
        if default_horizon not in self.HORIZONS:
            default_horizon = 15

        # MTF Consensus Weighting (Jerry's Quantor-MTFuzz A.9)
        # Weight longer horizons more heavily for trend confirmation
        # Jerry: Daily=50%, 60m=35%, 5m=15%
        # Our horizons: 45m=40%, 30m=30%, 15m=20%, 5m=10%
        use_mtf_consensus = os.environ.get('MTF_CONSENSUS_ENABLED', '0') == '1'
        if use_mtf_consensus:
            # Weights for each horizon (sum to 1.0)
            mtf_weights = {
                5: float(os.environ.get('MTF_WEIGHT_5M', '0.10')),
                15: float(os.environ.get('MTF_WEIGHT_15M', '0.20')),
                30: float(os.environ.get('MTF_WEIGHT_30M', '0.30')),
                45: float(os.environ.get('MTF_WEIGHT_45M', '0.40')),
            }
            # Normalize weights
            total_weight = sum(mtf_weights.values())
            mtf_weights = {k: v / total_weight for k, v in mtf_weights.items()}

            # Weighted average of returns (scalar outputs)
            weighted_return = sum(
                mtf_weights[h] * result[f'return_{h}m']
                for h in self.HORIZONS
            )
            # Weighted average of directions (logits, then take max)
            weighted_direction = sum(
                mtf_weights[h] * result[f'direction_{h}m']
                for h in self.HORIZONS
            )
            # Weighted average of confidences
            weighted_confidence = sum(
                mtf_weights[h] * result[f'confidence_{h}m']
                for h in self.HORIZONS
            )

            result['return'] = weighted_return
            result['direction'] = weighted_direction
            result['confidence'] = weighted_confidence
            result['mtf_consensus'] = True
        else:
            result['return'] = result[f'return_{default_horizon}m']
            result['direction'] = result[f'direction_{default_horizon}m']
            result['confidence'] = result[f'confidence_{default_horizon}m']
            result['mtf_consensus'] = False

        result['fillability'] = torch.sigmoid(self.fillability_head(h))
        result['exp_slippage'] = self.slippage_head(h)
        result['exp_ttf'] = torch.relu(self.ttf_head(h))
        result['volatility'] = self.vol_head(h)

        return result

    def set_confidence_trainable(self, enabled: bool) -> None:
        """
        Enable or disable gradient flow through the confidence heads.

        Fix 1 for broken confidence head: When TRAIN_CONFIDENCE_BCE=0 (default),
        confidence heads receive no direct loss but still get gradient leakage
        from the shared backbone, causing inverted correlations.

        V3 Note: This affects ALL per-horizon confidence heads (5m, 15m, 30m, 45m).
        """
        self._confidence_trainable = bool(enabled)
        for h in self.HORIZONS:
            head = self.horizon_heads[f'confidence_{h}m']
            for p in head.parameters():
                p.requires_grad_(self._confidence_trainable)
        logger.info(f"Confidence heads trainable: {self._confidence_trainable}")


# ------------------------------------------------------------------------------
# Exports
# ------------------------------------------------------------------------------
__all__ = [
    'BayesianLinear',
    'RBFKernelLayer',
    'TCNBlock',
    'OptionsTCN',
    'OptionsLSTM',
    'ResidualBlock',
    'SlimResidualBlock',
    'UnifiedOptionsPredictor',
    'UnifiedOptionsPredictorV2',
    'UnifiedOptionsPredictorV3',
    'DirectionPredictor',
    'DirectionPredictorV3',
    'create_predictor',
]
