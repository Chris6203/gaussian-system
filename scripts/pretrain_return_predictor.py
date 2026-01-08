#!/usr/bin/env python3
"""
Pretrain RL Policy on State → Predicted Return

Different approach from expert imitation:
- OLD: state → action (CALL/PUT) - teaches "what to do"
- NEW: state → return - teaches "what to expect"

This learns the underlying value function, which is more useful for RL.

Usage:
    python scripts/pretrain_return_predictor.py --epochs 30 --output models/pretrained/return_predictor.pt
"""

import os
import sys
import argparse
import sqlite3
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from glob import glob

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split

# Setup path
sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ReturnPredictionDataset(Dataset):
    """
    Dataset for state → return prediction.

    Each sample contains:
    - state: Market state features (variable dim, padded to max_dim)
    - return: Realized return that occurred
    - direction: Which way the market actually moved (0=down, 1=neutral, 2=up)
    - profitable: Whether a trade would have been profitable (binary)
    """

    def __init__(
        self,
        data_dir: str = 'models',
        max_samples: int = 200000,
        feature_dim: int = 50  # Standard feature dimension
    ):
        self.samples = []
        self.feature_dim = feature_dim
        self._load_from_tradability_datasets(data_dir, max_samples)

    def _load_from_tradability_datasets(self, data_dir: str, max_samples: int):
        """Load from tradability gate datasets (has realized_move)"""

        # Find all tradability datasets
        pattern = os.path.join(data_dir, '**/state/tradability_gate_dataset.jsonl')
        files = glob(pattern, recursive=True)

        logger.info(f"Found {len(files)} tradability datasets")

        samples_loaded = 0
        for filepath in files:
            if samples_loaded >= max_samples:
                break

            try:
                with open(filepath, 'r') as f:
                    for line in f:
                        if samples_loaded >= max_samples:
                            break

                        rec = json.loads(line.strip())

                        # Get realized move (the actual return that occurred)
                        realized_move = rec.get('realized_move')
                        if realized_move is None:
                            continue

                        # Get features
                        features = rec.get('features', {})
                        if not features:
                            continue

                        # Build feature vector
                        state = self._build_state_vector(features)

                        # Labels
                        return_val = float(realized_move)

                        # Direction: -1 to +1 range, discretize to 0/1/2
                        if return_val < -0.001:  # -0.1% threshold
                            direction = 0  # down
                        elif return_val > 0.001:  # +0.1% threshold
                            direction = 2  # up
                        else:
                            direction = 1  # neutral

                        # Would a CALL have been profitable? (return > 0)
                        # Would a PUT have been profitable? (return < 0)
                        # For now, use binary: would directional trade work?
                        profitable = 1 if abs(return_val) > 0.002 else 0  # 0.2% threshold

                        self.samples.append({
                            'state': state,
                            'return': return_val,
                            'direction': direction,
                            'profitable': profitable
                        })
                        samples_loaded += 1

            except Exception as e:
                logger.warning(f"Error loading {filepath}: {e}")
                continue

        logger.info(f"Loaded {len(self.samples)} samples from tradability datasets")

        # Log return distribution
        if self.samples:
            returns = [s['return'] for s in self.samples]
            logger.info(f"Return distribution: min={min(returns):.4f}, max={max(returns):.4f}, "
                       f"mean={np.mean(returns):.4f}, std={np.std(returns):.4f}")

    def _build_state_vector(self, features: dict) -> np.ndarray:
        """Build standardized feature vector from features dict"""
        state = np.zeros(self.feature_dim, dtype=np.float32)

        # Core features (matching typical order)
        feature_names = [
            'vix', 'vix_percentile', 'vix_term_structure',
            'spy_return_1m', 'spy_return_5m', 'spy_return_15m', 'spy_return_30m',
            'spy_volatility_5m', 'spy_volatility_15m',
            'rsi_14', 'macd', 'macd_signal', 'bb_position',
            'volume_ratio', 'price_momentum',
            'hmm_trend', 'hmm_volatility', 'hmm_liquidity', 'hmm_confidence',
            'predicted_return', 'confidence', 'direction_up', 'direction_down',
            'fear_greed', 'pcr', 'news_sentiment',
            'theta', 'delta', 'gamma', 'vega', 'iv',
            'time_to_expiry', 'moneyness',
            'sector_spy_corr', 'sector_strength',
            'tda_entropy_h0', 'tda_entropy_h1', 'tda_complexity'
        ]

        for i, name in enumerate(feature_names):
            if i >= self.feature_dim:
                break
            val = features.get(name, 0.0)
            if val is None:
                val = 0.0
            state[i] = float(val)

        return state

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        return {
            'state': torch.FloatTensor(sample['state']),
            'return': torch.FloatTensor([sample['return']]),
            'direction': torch.LongTensor([sample['direction']]),
            'profitable': torch.FloatTensor([sample['profitable']])
        }


class ReturnPredictorNetwork(nn.Module):
    """
    Simple MLP to predict returns from state.

    Multi-task outputs:
    - return_pred: Predicted return (regression)
    - direction_logits: Direction classification (down/neutral/up)
    - profitable_logit: Profitability prediction (binary)
    """

    def __init__(self, input_dim: int = 50, hidden_dim: int = 256):
        super().__init__()

        # Shared encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
        )

        # Return prediction head (regression)
        self.return_head = nn.Sequential(
            nn.Linear(hidden_dim // 2, 64),
            nn.GELU(),
            nn.Linear(64, 1)
        )

        # Direction classification head
        self.direction_head = nn.Sequential(
            nn.Linear(hidden_dim // 2, 64),
            nn.GELU(),
            nn.Linear(64, 3)  # down, neutral, up
        )

        # Profitability prediction head
        self.profitable_head = nn.Sequential(
            nn.Linear(hidden_dim // 2, 64),
            nn.GELU(),
            nn.Linear(64, 1)
        )

        # Track parameters
        total_params = sum(p.numel() for p in self.parameters())
        logger.info(f"ReturnPredictorNetwork: {total_params:,} parameters")

    def forward(self, x):
        h = self.encoder(x)
        return {
            'return_pred': self.return_head(h),
            'direction_logits': self.direction_head(h),
            'profitable_logit': self.profitable_head(h)
        }


def pretrain_return_predictor(
    epochs: int = 30,
    batch_size: int = 128,
    learning_rate: float = 0.001,
    output_path: str = 'models/pretrained/return_predictor.pt',
    data_dir: str = 'models',
    feature_dim: int = 50,
    device: str = 'cpu'
):
    """
    Pretrain return predictor on historical data.
    """
    logger.info("=" * 60)
    logger.info("Return Predictor Pretraining")
    logger.info("=" * 60)

    # Load dataset
    dataset = ReturnPredictionDataset(data_dir, feature_dim=feature_dim)

    if len(dataset) < 1000:
        logger.error(f"Not enough samples: {len(dataset)} (need 1000+)")
        return None

    # Split train/val
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=0)

    logger.info(f"Dataset: {len(dataset)} samples ({train_size} train, {val_size} val)")

    # Create model
    model = ReturnPredictorNetwork(input_dim=feature_dim).to(device)

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # Loss functions
    mse_loss = nn.MSELoss()
    ce_loss = nn.CrossEntropyLoss()
    bce_loss = nn.BCEWithLogitsLoss()

    best_val_loss = float('inf')
    best_state = None

    for epoch in range(epochs):
        # Training
        model.train()
        train_losses = {'return': 0, 'direction': 0, 'profitable': 0}
        train_correct = {'direction': 0, 'profitable': 0}
        train_total = 0

        for batch in train_loader:
            states = batch['state'].to(device)
            returns = batch['return'].to(device)
            directions = batch['direction'].squeeze(1).to(device)
            profitable = batch['profitable'].to(device)

            optimizer.zero_grad()

            # Forward
            out = model(states)

            # Multi-task loss
            loss_return = mse_loss(out['return_pred'], returns) * 100  # Scale up
            loss_direction = ce_loss(out['direction_logits'], directions)
            loss_profitable = bce_loss(out['profitable_logit'], profitable)

            loss = loss_return + loss_direction + loss_profitable

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_losses['return'] += loss_return.item()
            train_losses['direction'] += loss_direction.item()
            train_losses['profitable'] += loss_profitable.item()

            # Accuracy
            dir_preds = out['direction_logits'].argmax(dim=1)
            train_correct['direction'] += (dir_preds == directions).sum().item()

            prof_preds = (out['profitable_logit'] > 0).float()
            train_correct['profitable'] += (prof_preds == profitable).sum().item()

            train_total += len(states)

        # Validation
        model.eval()
        val_losses = {'return': 0, 'direction': 0, 'profitable': 0}
        val_correct = {'direction': 0, 'profitable': 0}
        val_total = 0

        with torch.no_grad():
            for batch in val_loader:
                states = batch['state'].to(device)
                returns = batch['return'].to(device)
                directions = batch['direction'].squeeze(1).to(device)
                profitable = batch['profitable'].to(device)

                out = model(states)

                val_losses['return'] += mse_loss(out['return_pred'], returns).item() * 100
                val_losses['direction'] += ce_loss(out['direction_logits'], directions).item()
                val_losses['profitable'] += bce_loss(out['profitable_logit'], profitable).item()

                dir_preds = out['direction_logits'].argmax(dim=1)
                val_correct['direction'] += (dir_preds == directions).sum().item()

                prof_preds = (out['profitable_logit'] > 0).float()
                val_correct['profitable'] += (prof_preds == profitable).sum().item()

                val_total += len(states)

        scheduler.step()

        # Compute metrics
        n_train_batches = len(train_loader)
        n_val_batches = len(val_loader)

        train_loss = sum(train_losses.values()) / n_train_batches
        val_loss = sum(val_losses.values()) / n_val_batches

        train_dir_acc = train_correct['direction'] / train_total * 100
        val_dir_acc = val_correct['direction'] / val_total * 100
        train_prof_acc = train_correct['profitable'] / train_total * 100
        val_prof_acc = val_correct['profitable'] / val_total * 100

        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if (epoch + 1) % 5 == 0 or epoch == 0:
            logger.info(
                f"Epoch {epoch+1:3d}/{epochs} | "
                f"Loss: {train_loss:.4f}/{val_loss:.4f} | "
                f"Dir Acc: {train_dir_acc:.1f}%/{val_dir_acc:.1f}% | "
                f"Prof Acc: {train_prof_acc:.1f}%/{val_prof_acc:.1f}% "
                f"{'*' if is_best else ''}"
            )

    # Save best model
    if best_state is not None:
        model.load_state_dict(best_state)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    torch.save({
        'model_state': model.state_dict(),
        'feature_dim': feature_dim,
        'best_val_loss': best_val_loss,
        'epochs': epochs,
        'samples': len(dataset),
        'timestamp': datetime.now().isoformat()
    }, output_path)

    logger.info(f"")
    logger.info(f"✅ Return predictor pretrained!")
    logger.info(f"   Best validation loss: {best_val_loss:.4f}")
    logger.info(f"   Saved to: {output_path}")

    return model


def main():
    parser = argparse.ArgumentParser(description='Pretrain return predictor')
    parser.add_argument('--epochs', type=int, default=30, help='Training epochs')
    parser.add_argument('--batch-size', type=int, default=128, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--output', type=str, default='models/pretrained/return_predictor.pt', help='Output path')
    parser.add_argument('--data-dir', type=str, default='models', help='Data directory')
    parser.add_argument('--feature-dim', type=int, default=50, help='Feature dimension')
    parser.add_argument('--device', type=str, default='cpu', help='Device')
    args = parser.parse_args()

    pretrain_return_predictor(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        output_path=args.output,
        data_dir=args.data_dir,
        feature_dim=args.feature_dim,
        device=args.device
    )


if __name__ == '__main__':
    main()
