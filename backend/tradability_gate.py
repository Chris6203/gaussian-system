#!/usr/bin/env python3
"""
Tradability Gate
================

A small model that estimates whether a proposed trade is "tradable" (i.e., likely
to beat friction such as spread/fees) over a short horizon.

This is intentionally simple and ML-first:
- Input: predictor outputs (predicted_return, confidence) + market context (VIX, momentum, volume, regime)
- Output: probability in [0,1] ≈ P(tradable)

It can be used as:
- a learned entry gate (approve/downgrade/veto)
- an auxiliary feature for RL (risk-aware action masking)
"""

from __future__ import annotations

import os
import logging
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


FEATURE_NAMES = [
    # Predictor outputs
    "predicted_return",      # fraction (e.g., 0.002 = +0.2%)
    "prediction_confidence", # fraction (0..1)
    # Market context
    "vix_level",             # raw
    "momentum_5m",           # fraction
    "volume_spike",          # ratio
    # Regime features (0..1)
    "hmm_trend",
    "hmm_volatility",
    "hmm_liquidity",
    "hmm_confidence",
]


def build_feature_vector(features: Dict) -> np.ndarray:
    """Build feature vector in the fixed FEATURE_NAMES order."""
    x = []
    for k in FEATURE_NAMES:
        v = features.get(k, 0.0)
        try:
            x.append(float(v))
        except Exception:
            x.append(0.0)
    return np.asarray(x, dtype=np.float32)


class TradabilityGateNet(nn.Module):
    """Tiny MLP for P(tradable)."""

    def __init__(self, input_dim: int, hidden_dim: int = 32, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


@dataclass
class TradabilityGateCheckpoint:
    """Serialized checkpoint format."""
    model_state_dict: Dict
    feature_names: Tuple[str, ...]
    mean: np.ndarray
    std: np.ndarray
    hidden_dim: int = 32
    dropout: float = 0.1


class TradabilityGateModel:
    """
    Wrapper around TradabilityGateNet that handles normalization and inference.
    """

    def __init__(
        self,
        device: str = "cpu",
        hidden_dim: int = 32,
        dropout: float = 0.1,
    ):
        self.device = torch.device(device)
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.feature_names = tuple(FEATURE_NAMES)

        self.model: Optional[TradabilityGateNet] = None
        self.mean: Optional[np.ndarray] = None
        self.std: Optional[np.ndarray] = None
        self.checkpoint_path: Optional[str] = None

    @property
    def is_loaded(self) -> bool:
        return self.model is not None and self.mean is not None and self.std is not None

    def load(self, path: str) -> bool:
        if not os.path.exists(path):
            logger.warning(f"[GATE] Checkpoint not found: {path}")
            return False

        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        feature_names = tuple(ckpt.get("feature_names", FEATURE_NAMES))
        mean = np.asarray(ckpt.get("mean", np.zeros(len(feature_names), dtype=np.float32)), dtype=np.float32)
        std = np.asarray(ckpt.get("std", np.ones(len(feature_names), dtype=np.float32)), dtype=np.float32)
        hidden_dim = int(ckpt.get("hidden_dim", self.hidden_dim))
        dropout = float(ckpt.get("dropout", self.dropout))

        self.feature_names = feature_names
        self.mean = mean
        self.std = np.clip(std, 1e-6, None)
        self.hidden_dim = hidden_dim
        self.dropout = dropout

        self.model = TradabilityGateNet(input_dim=len(self.feature_names), hidden_dim=hidden_dim, dropout=dropout).to(self.device)
        state = ckpt.get("model_state_dict", ckpt)
        self.model.load_state_dict(state, strict=False)
        self.model.eval()

        self.checkpoint_path = path
        logger.info(f"[GATE] Loaded tradability gate: {path}")
        return True

    def predict_proba(self, features: Dict) -> float:
        """
        Returns probability in [0,1] that the trade is tradable.
        """
        if not self.is_loaded:
            return 0.5

        x = []
        for k in self.feature_names:
            v = features.get(k, 0.0)
            try:
                x.append(float(v))
            except Exception:
                x.append(0.0)
        x = np.asarray(x, dtype=np.float32)

        x = (x - self.mean) / self.std
        xt = torch.from_numpy(x).float().unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = self.model(xt).squeeze()
            prob = torch.sigmoid(logits).item()
        return float(prob)





