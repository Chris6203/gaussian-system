#!/usr/bin/env python3
"""
Offline Pretraining Script for Confidence Head with BCE Loss

ROOT CAUSE DISCOVERED (2026-01-06):
The confidence head in neural_networks.py is NEVER TRAINED by default.
It outputs inverted values - high confidence correlates with LOW win rate:
  - 40%+ confidence → 0% actual win rate
  - 15-20% confidence → 7.2% actual win rate (highest!)

This script pretrains the model with BCE loss on historical data so that
the confidence head learns to predict P(win) correctly BEFORE deployment.

Usage:
    python scripts/pretrain_confidence.py --epochs 100 --output models/pretrained_bce.pt

Then use the pretrained model:
    LOAD_PRETRAINED=1 PRETRAINED_MODEL_PATH=models/pretrained_bce.pt python scripts/train_time_travel.py
"""

import os
import sys
import json
import argparse
import logging
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from bot_modules.neural_networks import UnifiedOptionsPredictor

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class TradabilityDataset(Dataset):
    """
    Dataset from tradability_gate_dataset.jsonl files.

    Each record contains:
    - features: dict of feature values at entry time
    - label: 1 if trade would have been profitable, 0 otherwise
    - proposed_action: BUY_CALLS or BUY_PUTS
    """

    def __init__(self, jsonl_paths: list, feature_dim: int = 50):
        self.samples = []
        self.feature_dim = feature_dim

        # Feature names we expect (subset that's consistently available)
        self.feature_names = [
            'predicted_return', 'prediction_confidence', 'vix_level',
            'momentum_5m', 'volume_spike', 'hmm_trend', 'hmm_volatility',
            'hmm_liquidity', 'hmm_confidence'
        ]

        for path in jsonl_paths:
            if not os.path.exists(path):
                logger.warning(f"File not found: {path}")
                continue

            logger.info(f"Loading {path}...")
            with open(path, 'r') as f:
                for line in f:
                    try:
                        rec = json.loads(line)
                        features = rec.get('features', {})
                        label = rec.get('label', 0)
                        action = rec.get('proposed_action', '')

                        # Only use CALL signals for now (they perform better)
                        if action != 'BUY_CALLS':
                            continue

                        # Extract feature vector
                        feat_vec = self._extract_features(features)
                        if feat_vec is not None:
                            self.samples.append({
                                'features': feat_vec,
                                'label': float(label),
                                'action': action,
                            })
                    except json.JSONDecodeError:
                        continue

        logger.info(f"Loaded {len(self.samples)} samples")

        # Calculate class balance
        n_pos = sum(1 for s in self.samples if s['label'] == 1)
        n_neg = len(self.samples) - n_pos
        logger.info(f"Class balance: {n_pos} positive ({100*n_pos/len(self.samples):.1f}%), {n_neg} negative")

    def _extract_features(self, features: dict) -> np.ndarray:
        """Extract feature vector from features dict."""
        vec = []
        for name in self.feature_names:
            val = features.get(name, 0.0)
            if val is None:
                val = 0.0
            vec.append(float(val))

        # Pad to feature_dim
        while len(vec) < self.feature_dim:
            vec.append(0.0)

        return np.array(vec[:self.feature_dim], dtype=np.float32)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        features = torch.tensor(sample['features'], dtype=torch.float32)
        label = torch.tensor([sample['label']], dtype=torch.float32)
        return features, label


def find_datasets(models_dir: str = 'models') -> list:
    """Find all tradability_gate_dataset.jsonl files."""
    datasets = []
    for root, dirs, files in os.walk(models_dir):
        for f in files:
            if f == 'tradability_gate_dataset.jsonl':
                datasets.append(os.path.join(root, f))
    return datasets


def pretrain_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int = 100,
    lr: float = 0.001,
    device: str = 'cpu'
) -> dict:
    """
    Pretrain model with BCE loss on confidence head.

    Returns:
        Training history dict
    """
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
    bce_loss = nn.BCELoss()

    history = {'train_loss': [], 'val_loss': [], 'val_accuracy': []}
    best_val_loss = float('inf')
    best_state = None

    for epoch in range(epochs):
        # Training
        model.train()
        train_losses = []

        for features, labels in train_loader:
            features = features.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            # Forward pass - need sequence input too
            # Create dummy sequence by repeating current features
            seq = features.unsqueeze(1).repeat(1, 60, 1)  # [B, 60, D]

            output = model(features, seq)
            confidence = output['confidence']

            # BCE loss - train confidence to predict P(win)
            loss = bce_loss(confidence, labels)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_losses.append(loss.item())

        avg_train_loss = np.mean(train_losses)
        history['train_loss'].append(avg_train_loss)

        # Validation
        model.eval()
        val_losses = []
        correct = 0
        total = 0

        with torch.no_grad():
            for features, labels in val_loader:
                features = features.to(device)
                labels = labels.to(device)

                seq = features.unsqueeze(1).repeat(1, 60, 1)
                output = model(features, seq)
                confidence = output['confidence']

                loss = bce_loss(confidence, labels)
                val_losses.append(loss.item())

                # Calculate accuracy (confidence > 0.5 predicts win)
                preds = (confidence > 0.5).float()
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        avg_val_loss = np.mean(val_losses)
        val_accuracy = correct / total if total > 0 else 0

        history['val_loss'].append(avg_val_loss)
        history['val_accuracy'].append(val_accuracy)

        # Learning rate scheduling
        scheduler.step(avg_val_loss)

        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_state = model.state_dict().copy()

        if (epoch + 1) % 10 == 0:
            logger.info(f"Epoch {epoch+1}/{epochs}: train_loss={avg_train_loss:.4f}, "
                       f"val_loss={avg_val_loss:.4f}, val_acc={val_accuracy:.1%}")

    # Restore best model
    if best_state is not None:
        model.load_state_dict(best_state)
        logger.info(f"Restored best model with val_loss={best_val_loss:.4f}")

    return history


def main():
    parser = argparse.ArgumentParser(description='Pretrain confidence head with BCE loss')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=64, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--output', type=str, default='models/pretrained_bce.pt', help='Output model path')
    parser.add_argument('--feature-dim', type=int, default=50, help='Feature dimension')
    parser.add_argument('--val-split', type=float, default=0.2, help='Validation split ratio')
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("CONFIDENCE HEAD PRETRAINING WITH BCE LOSS")
    logger.info("=" * 60)
    logger.info(f"Epochs: {args.epochs}, Batch size: {args.batch_size}, LR: {args.lr}")

    # Find all dataset files
    dataset_paths = find_datasets()
    if not dataset_paths:
        logger.error("No tradability_gate_dataset.jsonl files found!")
        return

    logger.info(f"Found {len(dataset_paths)} dataset files")

    # Load data
    full_dataset = TradabilityDataset(dataset_paths, feature_dim=args.feature_dim)

    if len(full_dataset) < 100:
        logger.error(f"Not enough samples: {len(full_dataset)}")
        return

    # Split into train/val
    val_size = int(len(full_dataset) * args.val_split)
    train_size = len(full_dataset) - val_size

    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size]
    )

    logger.info(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    # Create model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"Using device: {device}")

    model = UnifiedOptionsPredictor(
        feature_dim=args.feature_dim,
        sequence_length=60
    )

    # Pretrain
    history = pretrain_model(
        model, train_loader, val_loader,
        epochs=args.epochs, lr=args.lr, device=device
    )

    # Save model
    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    torch.save({
        'model_state_dict': model.state_dict(),
        'feature_dim': args.feature_dim,
        'training_history': history,
        'timestamp': datetime.now().isoformat(),
        'description': 'Pretrained with BCE loss for confidence calibration'
    }, args.output)

    logger.info(f"Model saved to {args.output}")

    # Print final stats
    logger.info("\n" + "=" * 60)
    logger.info("PRETRAINING COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Final val_loss: {history['val_loss'][-1]:.4f}")
    logger.info(f"Final val_accuracy: {history['val_accuracy'][-1]:.1%}")
    logger.info(f"\nTo use this model:")
    logger.info(f"  LOAD_PRETRAINED=1 PRETRAINED_MODEL_PATH={args.output} python scripts/train_time_travel.py")


if __name__ == '__main__':
    main()
