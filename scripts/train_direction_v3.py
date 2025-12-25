#!/usr/bin/env python3
"""
Train DirectionPredictorV3 - Standalone supervised training for direction prediction.

This script trains the simplified V3 direction predictor using:
- Historical price data from the database
- Walk-forward time-based train/val split
- Binary labels: UP (>0.3%) vs DOWN (<-0.3%), skip neutral
- Label smoothing for better calibration
- No RBF kernels (simpler model, better generalization)

Usage:
    python scripts/train_direction_v3.py --output models/direction_v3.pt
"""

import os
import sys
import json
import sqlite3
import logging
import argparse
from datetime import datetime, timedelta
from pathlib import Path
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from bot_modules.neural_networks import DirectionPredictorV3

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ==============================================================================
# Configuration
# ==============================================================================
@dataclass
class TrainingConfig:
    """Training hyperparameters."""
    # Data
    sequence_length: int = 30          # Number of historical bars
    horizon_minutes: int = 15          # Forward look for labels
    min_move_pct: float = 0.002        # 0.2% threshold for UP/DOWN (lower = more samples)

    # Training
    batch_size: int = 64
    learning_rate: float = 0.0003
    weight_decay: float = 0.01
    epochs: int = 50
    label_smoothing: float = 0.1       # Prevent overconfidence

    # Validation
    val_split: float = 0.2             # Last 20% for validation
    early_stop_patience: int = 10

    # Balancing
    balance_classes: bool = True       # Undersample majority class

    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


# ==============================================================================
# Dataset
# ==============================================================================
class DirectionDataset(Dataset):
    """Dataset for direction prediction training."""

    def __init__(self, features: np.ndarray, labels: np.ndarray, sequences: np.ndarray):
        """
        Args:
            features: Current features [N, D]
            labels: Binary direction labels [N] (0=DOWN, 1=UP)
            sequences: Historical feature sequences [N, T, D]
        """
        self.features = torch.FloatTensor(features)
        self.labels = torch.LongTensor(labels)
        self.sequences = torch.FloatTensor(sequences)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            'features': self.features[idx],
            'sequence': self.sequences[idx],
            'label': self.labels[idx],
        }


# ==============================================================================
# Data Loading
# ==============================================================================
def load_price_data(db_path: str, symbol: str = "SPY", limit: int = None) -> List[Dict]:
    """Load 1-minute price bars from historical database."""
    logger.info(f"Loading price data from {db_path}")

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    # Try polygon_intraday_1m first, fallback to fmp_intraday_1m
    query = """
        SELECT timestamp, open_price as open, high_price as high,
               low_price as low, close_price as close, volume
        FROM polygon_intraday_1m
        WHERE symbol = ?
        ORDER BY timestamp ASC
    """
    if limit:
        query += f" LIMIT {limit}"

    try:
        cursor = conn.execute(query, (symbol,))
        rows = [dict(r) for r in cursor.fetchall()]
    except sqlite3.OperationalError:
        # Fallback to fmp table
        logger.info("Falling back to fmp_intraday_1m table")
        query = query.replace("polygon_intraday_1m", "fmp_intraday_1m")
        cursor = conn.execute(query, (symbol,))
        rows = [dict(r) for r in cursor.fetchall()]

    conn.close()

    logger.info(f"Loaded {len(rows)} price bars for {symbol}")
    return rows


def create_features(bars: List[Dict], lookback: int = 30, include_regime: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create features from price bars.

    Features (6 base):
    - Price returns (1m, 5m, 15m)
    - Volatility (rolling std)
    - Momentum indicators
    - Volume features

    Regime Features (4 additional when include_regime=True):
    - trend_regime: Price vs MA20/MA50 (-1 to 1)
    - vol_regime: Current vol percentile (0 to 1)
    - range_regime: ATR percentile (0 to 1)
    - momentum_regime: RSI-style (-1 to 1)

    Returns:
        features: [N, D] current features
        sequences: [N, T, D] historical sequences
    """
    if len(bars) < lookback + 60:  # Need enough history
        raise ValueError(f"Need at least {lookback + 60} bars, got {len(bars)}")

    # Convert to arrays
    closes = np.array([b['close'] for b in bars], dtype=np.float32)
    highs = np.array([b['high'] for b in bars], dtype=np.float32)
    lows = np.array([b['low'] for b in bars], dtype=np.float32)
    volumes = np.array([b['volume'] for b in bars], dtype=np.float32)

    # Normalize volume
    vol_mean = np.mean(volumes) + 1e-8
    volumes_norm = volumes / vol_mean

    # Calculate returns at different scales
    ret_1m = np.zeros_like(closes)
    ret_1m[1:] = (closes[1:] - closes[:-1]) / (closes[:-1] + 1e-8)

    ret_5m = np.zeros_like(closes)
    ret_5m[5:] = (closes[5:] - closes[:-5]) / (closes[:-5] + 1e-8)

    ret_15m = np.zeros_like(closes)
    ret_15m[15:] = (closes[15:] - closes[:-15]) / (closes[:-15] + 1e-8)

    # Rolling volatility (20-bar)
    volatility = np.zeros_like(closes)
    for i in range(20, len(closes)):
        volatility[i] = np.std(ret_1m[i-20:i])

    # Momentum (price position in range)
    momentum = np.zeros_like(closes)
    for i in range(20, len(closes)):
        high_20 = np.max(highs[i-20:i])
        low_20 = np.min(lows[i-20:i])
        if high_20 > low_20:
            momentum[i] = (closes[i] - low_20) / (high_20 - low_20)
        else:
            momentum[i] = 0.5

    # Volume spike
    vol_spike = np.zeros_like(closes)
    for i in range(20, len(closes)):
        vol_mean_20 = np.mean(volumes[i-20:i]) + 1e-8
        vol_spike[i] = volumes[i] / vol_mean_20

    # === REGIME FEATURES ===
    if include_regime:
        # 1. Trend Regime: Price vs MA20 and MA50 combined
        # -1 = bearish (below both MAs), 0 = neutral, +1 = bullish (above both)
        trend_regime = np.zeros_like(closes)
        ma20 = np.zeros_like(closes)
        ma50 = np.zeros_like(closes)
        for i in range(50, len(closes)):
            ma20[i] = np.mean(closes[i-20:i])
            ma50[i] = np.mean(closes[i-50:i])
            # Score: +0.5 if above MA20, +0.5 if above MA50
            score = 0.0
            if closes[i] > ma20[i]:
                score += 0.5
            else:
                score -= 0.5
            if closes[i] > ma50[i]:
                score += 0.5
            else:
                score -= 0.5
            trend_regime[i] = score  # Range: -1 to +1

        # 2. Volatility Regime: Current vol vs 100-bar rolling percentile
        vol_regime = np.zeros_like(closes)
        for i in range(100, len(closes)):
            vol_history = volatility[i-100:i]
            if np.max(vol_history) > np.min(vol_history):
                vol_regime[i] = (volatility[i] - np.min(vol_history)) / (np.max(vol_history) - np.min(vol_history) + 1e-8)
            else:
                vol_regime[i] = 0.5

        # 3. Range Regime: ATR percentile (high ATR = ranging/volatile, low ATR = quiet)
        atr = np.zeros_like(closes)
        for i in range(1, len(closes)):
            atr[i] = highs[i] - lows[i]  # Simplified ATR (true range)

        range_regime = np.zeros_like(closes)
        for i in range(100, len(closes)):
            atr_history = atr[i-100:i]
            if np.max(atr_history) > np.min(atr_history):
                range_regime[i] = (atr[i] - np.min(atr_history)) / (np.max(atr_history) - np.min(atr_history) + 1e-8)
            else:
                range_regime[i] = 0.5

        # 4. Momentum Regime: RSI-style indicator, normalized to -1 to +1
        # Compares up-moves vs down-moves over 14 bars
        momentum_regime = np.zeros_like(closes)
        for i in range(14, len(closes)):
            gains = np.maximum(ret_1m[i-14:i], 0)
            losses = np.abs(np.minimum(ret_1m[i-14:i], 0))
            avg_gain = np.mean(gains)
            avg_loss = np.mean(losses) + 1e-8
            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))
            momentum_regime[i] = (rsi - 50) / 50  # Normalize: -1 to +1

        # Stack all features: 6 base + 4 regime = 10 features
        feature_dim = 10
        all_features = np.stack([
            ret_1m, ret_5m, ret_15m, volatility, momentum, vol_spike,
            trend_regime, vol_regime, range_regime, momentum_regime
        ], axis=1)  # [N, D]

        logger.info("Including 4 regime features: trend, vol, range, momentum")
    else:
        # Stack features: [return_1m, return_5m, return_15m, volatility, momentum, vol_spike]
        feature_dim = 6
        all_features = np.stack([
            ret_1m, ret_5m, ret_15m, volatility, momentum, vol_spike
        ], axis=1)  # [N, D]

    # Create sequences and current features
    start_idx = max(lookback + 15, 100)  # Need at least 100 bars for regime features
    end_idx = len(bars) - 15   # Stop 15 bars from end for labels

    n_samples = end_idx - start_idx
    features = np.zeros((n_samples, feature_dim), dtype=np.float32)
    sequences = np.zeros((n_samples, lookback, feature_dim), dtype=np.float32)

    for i, idx in enumerate(range(start_idx, end_idx)):
        features[i] = all_features[idx]
        sequences[i] = all_features[idx - lookback:idx]

    logger.info(f"Created {n_samples} samples with {feature_dim} features")
    return features, sequences


def create_labels(bars: List[Dict], horizon_minutes: int = 15, min_move_pct: float = 0.003,
                  lookback: int = 30, include_regime: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create binary direction labels from forward returns.

    Args:
        bars: Price bars
        horizon_minutes: Look-ahead for return calculation
        min_move_pct: Threshold for UP/DOWN (e.g., 0.003 = 0.3%)
        lookback: Sequence length (to align with features)
        include_regime: If True, use start_idx = 100 to align with regime features

    Returns:
        labels: [N] binary labels (0=DOWN, 1=UP)
        mask: [N] boolean mask (True = valid sample, False = neutral/skip)
    """
    closes = np.array([b['close'] for b in bars], dtype=np.float32)

    # Use same start_idx as create_features
    start_idx = max(lookback + 15, 100) if include_regime else (lookback + 15)
    end_idx = len(bars) - 15
    n_samples = end_idx - start_idx

    labels = np.zeros(n_samples, dtype=np.int64)
    mask = np.zeros(n_samples, dtype=bool)

    for i, idx in enumerate(range(start_idx, end_idx)):
        current_price = closes[idx]
        future_price = closes[idx + horizon_minutes] if idx + horizon_minutes < len(closes) else current_price

        forward_return = (future_price - current_price) / (current_price + 1e-8)

        if forward_return > min_move_pct:
            labels[i] = 1  # UP
            mask[i] = True
        elif forward_return < -min_move_pct:
            labels[i] = 0  # DOWN
            mask[i] = True
        # Else: neutral, mask stays False

    n_up = np.sum(labels[mask] == 1)
    n_down = np.sum(labels[mask] == 0)
    n_neutral = n_samples - np.sum(mask)

    logger.info(f"Labels: UP={n_up}, DOWN={n_down}, NEUTRAL(skipped)={n_neutral}")
    logger.info(f"Class balance: UP={n_up/(n_up+n_down)*100:.1f}%, DOWN={n_down/(n_up+n_down)*100:.1f}%")

    return labels, mask


# ==============================================================================
# Training
# ==============================================================================
def train_epoch(model: nn.Module, loader: DataLoader, optimizer: torch.optim.Optimizer,
                config: TrainingConfig) -> Tuple[float, float]:
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for batch in loader:
        features = batch['features'].to(config.device)
        sequences = batch['sequence'].to(config.device)
        labels = batch['label'].to(config.device)

        optimizer.zero_grad()

        out = model(features, sequences)
        logits = out['direction_logits']

        # Cross-entropy with label smoothing
        loss = F.cross_entropy(logits, labels, label_smoothing=config.label_smoothing)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item() * features.size(0)
        predictions = torch.argmax(logits, dim=1)
        correct += (predictions == labels).sum().item()
        total += features.size(0)

    avg_loss = total_loss / total
    accuracy = correct / total
    return avg_loss, accuracy


def validate(model: nn.Module, loader: DataLoader, config: TrainingConfig) -> Tuple[float, float]:
    """Validate model."""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in loader:
            features = batch['features'].to(config.device)
            sequences = batch['sequence'].to(config.device)
            labels = batch['label'].to(config.device)

            out = model(features, sequences)
            logits = out['direction_logits']

            loss = F.cross_entropy(logits, labels)

            total_loss += loss.item() * features.size(0)
            predictions = torch.argmax(logits, dim=1)
            correct += (predictions == labels).sum().item()
            total += features.size(0)

    avg_loss = total_loss / total
    accuracy = correct / total
    return avg_loss, accuracy


def train_model(model: nn.Module, train_loader: DataLoader, val_loader: DataLoader,
                config: TrainingConfig) -> Dict:
    """Full training loop with early stopping."""

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5
    )

    best_val_acc = 0.0
    best_state = None
    patience_counter = 0
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

    logger.info("="*60)
    logger.info("Starting training...")
    logger.info(f"Device: {config.device}")
    logger.info(f"Train samples: {len(train_loader.dataset)}")
    logger.info(f"Val samples: {len(val_loader.dataset)}")
    logger.info("="*60)

    for epoch in range(config.epochs):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, config)
        val_loss, val_acc = validate(model, val_loader, config)

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        scheduler.step(val_acc)

        improved = val_acc > best_val_acc
        if improved:
            best_val_acc = val_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
            marker = " *BEST*"
        else:
            patience_counter += 1
            marker = ""

        logger.info(f"Epoch {epoch+1:3d}/{config.epochs} | "
                   f"Train: loss={train_loss:.4f} acc={train_acc:.4f} | "
                   f"Val: loss={val_loss:.4f} acc={val_acc:.4f}{marker}")

        if patience_counter >= config.early_stop_patience:
            logger.info(f"Early stopping at epoch {epoch+1}")
            break

    # Restore best model
    if best_state:
        model.load_state_dict(best_state)

    return {
        'best_val_acc': best_val_acc,
        'history': history,
        'epochs_trained': epoch + 1,
    }


# ==============================================================================
# Main
# ==============================================================================
def main():
    parser = argparse.ArgumentParser(description="Train DirectionPredictorV3")
    parser.add_argument('--db', type=str, default='data/db/historical.db',
                       help='Path to historical database')
    parser.add_argument('--symbol', type=str, default='SPY',
                       help='Symbol to train on')
    parser.add_argument('--output', type=str, default='models/direction_v3.pt',
                       help='Output model path')
    parser.add_argument('--limit', type=int, default=None,
                       help='Limit number of bars (for testing)')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=64,
                       help='Batch size')
    parser.add_argument('--lr', type=float, default=0.0003,
                       help='Learning rate')
    parser.add_argument('--no-regime', action='store_true',
                       help='Disable regime features (use only 6 base features)')
    args = parser.parse_args()

    include_regime = not args.no_regime

    # Configuration
    config = TrainingConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
    )

    # Load data
    db_path = Path(args.db)
    if not db_path.is_absolute():
        db_path = Path(__file__).parent.parent / args.db

    bars = load_price_data(str(db_path), args.symbol, args.limit)

    if len(bars) < 1000:
        logger.error(f"Need at least 1000 bars for training, got {len(bars)}")
        sys.exit(1)

    # Create features and labels
    features, sequences = create_features(bars, lookback=config.sequence_length, include_regime=include_regime)
    labels, mask = create_labels(bars, horizon_minutes=config.horizon_minutes,
                                  min_move_pct=config.min_move_pct,
                                  lookback=config.sequence_length,
                                  include_regime=include_regime)

    # Filter to non-neutral samples
    features = features[mask]
    sequences = sequences[mask]
    labels = labels[mask]

    logger.info(f"Training on {len(labels)} non-neutral samples")

    # Walk-forward split (last 20% for validation)
    split_idx = int(len(labels) * (1 - config.val_split))

    train_features = features[:split_idx]
    train_sequences = sequences[:split_idx]
    train_labels = labels[:split_idx]

    val_features = features[split_idx:]
    val_sequences = sequences[split_idx:]
    val_labels = labels[split_idx:]

    # Class balancing - undersample majority class in training set
    if config.balance_classes:
        up_idx = np.where(train_labels == 1)[0]
        down_idx = np.where(train_labels == 0)[0]

        n_up = len(up_idx)
        n_down = len(down_idx)

        logger.info(f"Before balancing: UP={n_up}, DOWN={n_down}")

        # Undersample majority class
        if n_up > n_down:
            # Randomly sample from UP to match DOWN count
            np.random.seed(42)
            up_idx = np.random.choice(up_idx, size=n_down, replace=False)
        elif n_down > n_up:
            # Randomly sample from DOWN to match UP count
            np.random.seed(42)
            down_idx = np.random.choice(down_idx, size=n_up, replace=False)

        # Combine indices and sort to preserve some temporal order
        balanced_idx = np.sort(np.concatenate([up_idx, down_idx]))

        train_features = train_features[balanced_idx]
        train_sequences = train_sequences[balanced_idx]
        train_labels = train_labels[balanced_idx]

        logger.info(f"After balancing: {len(train_labels)} samples (50/50 split)")

    logger.info(f"Train: {len(train_labels)} samples, Val: {len(val_labels)} samples")

    # Create datasets and loaders
    train_dataset = DirectionDataset(train_features, train_labels, train_sequences)
    val_dataset = DirectionDataset(val_features, val_labels, val_sequences)

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)

    # Create model
    feature_dim = features.shape[1]
    model = DirectionPredictorV3(
        feature_dim=feature_dim,
        sequence_length=config.sequence_length
    ).to(config.device)

    logger.info(f"Model: DirectionPredictorV3 with {feature_dim} features")

    # Train
    result = train_model(model, train_loader, val_loader, config)

    # Save model
    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = Path(__file__).parent.parent / args.output

    output_path.parent.mkdir(parents=True, exist_ok=True)

    save_dict = {
        'state_dict': model.state_dict(),
        'config': {
            'feature_dim': feature_dim,
            'sequence_length': config.sequence_length,
            'horizon_minutes': config.horizon_minutes,
            'min_move_pct': config.min_move_pct,
            'include_regime': include_regime,
        },
        'training': {
            'best_val_acc': result['best_val_acc'],
            'epochs_trained': result['epochs_trained'],
            'train_samples': len(train_labels),
            'val_samples': len(val_labels),
        },
        'timestamp': datetime.now().isoformat(),
    }

    torch.save(save_dict, output_path)
    logger.info(f"Model saved to {output_path}")

    # Save metadata
    metadata_path = output_path.with_suffix('.json')
    with open(metadata_path, 'w') as f:
        json.dump({
            'config': save_dict['config'],
            'training': save_dict['training'],
            'timestamp': save_dict['timestamp'],
        }, f, indent=2)
    logger.info(f"Metadata saved to {metadata_path}")

    # Summary
    logger.info("="*60)
    logger.info("TRAINING COMPLETE")
    logger.info("="*60)
    logger.info(f"Best Validation Accuracy: {result['best_val_acc']*100:.2f}%")
    logger.info(f"Epochs Trained: {result['epochs_trained']}")
    logger.info(f"Model saved to: {output_path}")
    logger.info("="*60)

    # Target: 55-60% for 50% trading win rate
    if result['best_val_acc'] >= 0.55:
        logger.info("TARGET MET: Validation accuracy >= 55%")
    else:
        logger.info(f"TARGET NOT MET: Need 55%+, got {result['best_val_acc']*100:.1f}%")
        logger.info("Consider: more data, feature engineering, or architecture changes")


if __name__ == "__main__":
    main()
