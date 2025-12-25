# Transformer Architecture Upgrade Plan

## Overview

Modular, swappable architecture upgrade integrating T5/Gemma2 innovations. **All components are configurable** via environment variables or config.json.

## Configuration System

### Environment Variables (Quick Testing)

```bash
# Temporal Encoder
TEMPORAL_ENCODER=tcn          # Original TCN (default)
TEMPORAL_ENCODER=transformer  # New Transformer
TEMPORAL_ENCODER=lstm         # Legacy LSTM

# Normalization
NORM_TYPE=layernorm           # Original LayerNorm (default)
NORM_TYPE=rmsnorm             # New RMSNorm

# Activation
ACTIVATION_TYPE=gelu          # Original GELU (default)
ACTIVATION_TYPE=geglu         # New GeGLU (gated)
ACTIVATION_TYPE=swiglu        # Alternative SwiGLU

# Gaussian Kernel
RBF_GATED=0                   # Original RBF (default)
RBF_GATED=1                   # Gated RBF

# Position Encoding (for Transformer)
POS_ENCODING=none             # No position encoding (default for TCN)
POS_ENCODING=rope             # Rotary Position Embedding
POS_ENCODING=sinusoidal       # Classic sinusoidal
```

### config.json (Persistent)

```json
{
  "neural_architecture": {
    "temporal_encoder": "tcn",
    "norm_type": "layernorm",
    "activation_type": "gelu",
    "rbf_gated": false,
    "pos_encoding": "none",

    "transformer": {
      "num_layers": 2,
      "num_heads": 4,
      "dropout": 0.15
    },

    "tcn": {
      "num_layers": 5,
      "kernel_size": 3,
      "dropout": 0.2
    }
  }
}
```

## Swappable Components

### 1. Normalization Layer

```python
def get_norm_layer(dim: int) -> nn.Module:
    norm_type = os.environ.get('NORM_TYPE', 'layernorm').lower()
    if norm_type == 'rmsnorm':
        return RMSNorm(dim)
    else:
        return nn.LayerNorm(dim)
```

**RMSNorm** (Gemma2/LLaMA):
```python
class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization"""
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        return self.weight * (x / rms)
```

### 2. Activation Layer

```python
def get_activation(in_dim: int = None, out_dim: int = None) -> nn.Module:
    act_type = os.environ.get('ACTIVATION_TYPE', 'gelu').lower()
    if act_type == 'geglu':
        return GeGLU(in_dim, out_dim)
    elif act_type == 'swiglu':
        return SwiGLU(in_dim, out_dim)
    else:
        return nn.GELU()
```

**GeGLU** (T5/Gemma2):
```python
class GeGLU(nn.Module):
    """Gated GELU - splits input, applies GELU to gate"""
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.proj = nn.Linear(in_dim, out_dim * 2)
        self.out_dim = out_dim

    def forward(self, x):
        x, gate = self.proj(x).chunk(2, dim=-1)
        return x * F.gelu(gate)
```

**SwiGLU** (LLaMA alternative):
```python
class SwiGLU(nn.Module):
    """Swish-Gated Linear Unit"""
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.proj = nn.Linear(in_dim, out_dim * 2)

    def forward(self, x):
        x, gate = self.proj(x).chunk(2, dim=-1)
        return x * F.silu(gate)  # SiLU = Swish
```

### 3. Temporal Encoder

```python
def get_temporal_encoder(input_dim: int, config: dict) -> nn.Module:
    encoder_type = os.environ.get('TEMPORAL_ENCODER',
                                   config.get('temporal_encoder', 'tcn')).lower()

    if encoder_type == 'transformer':
        return OptionsTransformer(
            input_dim,
            num_layers=config.get('transformer', {}).get('num_layers', 2),
            num_heads=config.get('transformer', {}).get('num_heads', 4),
            dropout=config.get('transformer', {}).get('dropout', 0.15)
        )
    elif encoder_type == 'lstm':
        return OptionsLSTM(input_dim)
    else:  # tcn (default)
        return OptionsTCN(
            input_dim,
            num_layers=config.get('tcn', {}).get('num_layers', 5),
            dropout=config.get('tcn', {}).get('dropout', 0.2)
        )
```

**OptionsTransformer** (New):
```python
class OptionsTransformer(nn.Module):
    """Small causal Transformer encoder"""

    def __init__(self, input_dim: int, hidden_dim: int = 128,
                 num_layers: int = 2, num_heads: int = 4, dropout: float = 0.15):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)

        # Position encoding (configurable)
        pos_type = os.environ.get('POS_ENCODING', 'rope').lower()
        if pos_type == 'rope':
            self.pos_emb = RotaryPositionEmbedding(hidden_dim // num_heads)
        elif pos_type == 'sinusoidal':
            self.pos_emb = SinusoidalPositionEmbedding(hidden_dim)
        else:
            self.pos_emb = None

        # Transformer layers
        self.layers = nn.ModuleList([
            TransformerBlock(hidden_dim, num_heads, dropout)
            for _ in range(num_layers)
        ])

        self.final_norm = get_norm_layer(hidden_dim)

        # Attention pooling (same as TCN)
        self.attn_pool = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1)
        )
        self.ctx = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            get_norm_layer(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 64),
        )

    def forward(self, x):
        # x: [B, T, D]
        B, T, D = x.shape
        x = self.input_proj(x)  # [B, T, H]

        # Apply position encoding
        if self.pos_emb is not None:
            x = self.pos_emb(x)

        # Causal mask
        mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()

        for layer in self.layers:
            x = layer(x, mask)

        x = self.final_norm(x)

        # Attention pooling
        attn_weights = torch.softmax(self.attn_pool(x), dim=1)
        pooled = torch.sum(x * attn_weights, dim=1)

        return self.ctx(pooled)  # [B, 64]


class TransformerBlock(nn.Module):
    """Single Transformer block with pre-norm"""

    def __init__(self, dim: int, num_heads: int, dropout: float):
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

    def forward(self, x, mask=None):
        # Pre-norm residual (Gemma2/LLaMA style)
        h = self.norm1(x)
        h, _ = self.attn(h, h, h, attn_mask=mask)
        x = x + self.dropout(h)

        h = self.norm2(x)
        x = x + self.dropout(self.ffn(h))
        return x
```

### 4. Gated RBF Kernel

```python
def get_rbf_layer(input_dim: int) -> nn.Module:
    if os.environ.get('RBF_GATED', '0') == '1':
        return GatedRBFKernelLayer(input_dim)
    else:
        return RBFKernelLayer(input_dim)


class GatedRBFKernelLayer(nn.Module):
    """RBF with learned gate to suppress irrelevant features"""

    def __init__(self, input_dim: int, n_centers: int = 25,
                 scales: list = None):
        super().__init__()
        self.rbf = RBFKernelLayer(input_dim, n_centers, scales)
        rbf_out_dim = n_centers * len(scales or [0.1, 0.5, 1.0, 2.0, 5.0])
        self.gate = nn.Sequential(
            nn.Linear(input_dim, rbf_out_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        rbf_features = self.rbf(x)
        gate_weights = self.gate(x)
        return rbf_features * gate_weights
```

### 5. Position Encodings

```python
class RotaryPositionEmbedding(nn.Module):
    """RoPE - Rotary Position Embedding (LLaMA/Gemma2 style)"""

    def __init__(self, dim: int, max_seq_len: int = 128):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)
        self.max_seq_len = max_seq_len

    def forward(self, x):
        # x: [B, T, D]
        T = x.size(1)
        pos = torch.arange(T, device=x.device).float()
        sincos = torch.einsum('i,j->ij', pos, self.inv_freq)
        sin, cos = sincos.sin(), sincos.cos()

        # Apply rotation to pairs of dimensions
        x1, x2 = x[..., ::2], x[..., 1::2]
        x_rotated = torch.stack([
            x1 * cos - x2 * sin,
            x1 * sin + x2 * cos
        ], dim=-1).flatten(-2)

        return x_rotated


class SinusoidalPositionEmbedding(nn.Module):
    """Classic sinusoidal position embedding"""

    def __init__(self, dim: int, max_seq_len: int = 128):
        super().__init__()
        pe = torch.zeros(max_seq_len, dim)
        position = torch.arange(0, max_seq_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, dim, 2).float() *
                            (-math.log(10000.0) / dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]
```

## Configuration Matrix

Test different combinations:

| Config Name | Encoder | Norm | Activation | RBF | Expected |
|-------------|---------|------|------------|-----|----------|
| baseline | tcn | layernorm | gelu | standard | Current behavior |
| phase1 | tcn | rmsnorm | geglu | standard | Low risk upgrade |
| phase2 | transformer | rmsnorm | geglu | standard | Full upgrade |
| phase2_gated | transformer | rmsnorm | geglu | gated | Full + gated RBF |
| minimal | transformer | layernorm | gelu | standard | Just encoder swap |

### Quick Test Commands

```bash
# Baseline (current)
python scripts/train_time_travel.py

# Phase 1: Just RMSNorm + GeGLU
NORM_TYPE=rmsnorm ACTIVATION_TYPE=geglu python scripts/train_time_travel.py

# Phase 2: Transformer + RMSNorm + GeGLU
TEMPORAL_ENCODER=transformer NORM_TYPE=rmsnorm ACTIVATION_TYPE=geglu python scripts/train_time_travel.py

# Minimal: Just Transformer
TEMPORAL_ENCODER=transformer python scripts/train_time_travel.py

# Full upgrade with gated RBF
TEMPORAL_ENCODER=transformer NORM_TYPE=rmsnorm ACTIVATION_TYPE=geglu RBF_GATED=1 python scripts/train_time_travel.py
```

## Implementation Order

1. **Add helper functions**: `get_norm_layer()`, `get_activation()`, `get_temporal_encoder()`, `get_rbf_layer()`
2. **Add new classes**: `RMSNorm`, `GeGLU`, `SwiGLU`, `GatedRBFKernelLayer`
3. **Add Transformer**: `OptionsTransformer`, `TransformerBlock`, `RotaryPositionEmbedding`
4. **Update UnifiedOptionsPredictor** to use helper functions
5. **Test each configuration independently**

## Rollback

Any configuration can instantly rollback to baseline:

```bash
# Force baseline
TEMPORAL_ENCODER=tcn NORM_TYPE=layernorm ACTIVATION_TYPE=gelu RBF_GATED=0 python scripts/train_time_travel.py
```

Or simply run without env vars (all defaults = baseline).

## What's Preserved

| Component | Status | Notes |
|-----------|--------|-------|
| RBFKernelLayer | ✅ Kept | Optionally gated |
| BayesianLinear heads | ✅ Kept | Unchanged |
| HMM regime values | ✅ Kept | Still concatenated to features |
| Attention pooling | ✅ Kept | Same for all encoders |
| Output shape [B, 64] | ✅ Kept | Drop-in compatible |
| Residual blocks | ✅ Kept | Use new norm/activation |
