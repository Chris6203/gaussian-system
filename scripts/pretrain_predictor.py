#!/usr/bin/env python3
"""
Comprehensive Pretraining Script for UnifiedOptionsPredictor.

This script pretrains the full predictor (temporal encoder + Bayesian heads) on:
1. Return regression (predict future returns)
2. Direction classification (UP/NEUTRAL/DOWN)
3. Confidence BCE (predict trade profitability)

Supports all temporal encoders: TCN, LSTM, Transformer, Mamba2

Usage:
    # Pretrain with default TCN encoder
    python scripts/pretrain_predictor.py --epochs 50 --output models/pretrained/predictor_tcn.pt

    # Pretrain with Mamba2 encoder
    TEMPORAL_ENCODER=mamba2 python scripts/pretrain_predictor.py --output models/pretrained/predictor_mamba2.pt

    # Use pretrained model for training
    LOAD_PRETRAINED=1 PRETRAINED_MODEL_PATH=models/pretrained/predictor_tcn.pt python scripts/train_time_travel.py
"""

import os
import sys
import json
import pickle
import argparse
import logging
from datetime import datetime
from pathlib import Path
from collections import deque
from typing import List, Tuple, Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, ConcatDataset

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from bot_modules.neural_networks import UnifiedOptionsPredictor, get_temporal_encoder

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ==============================================================================
# Dataset Classes
# ==============================================================================

class TradabilityDataset(Dataset):
    """Dataset from tradability_gate_dataset.jsonl files for confidence pretraining."""

    FEATURE_NAMES = [
        'predicted_return', 'prediction_confidence', 'vix_level',
        'momentum_5m', 'momentum_15m', 'volume_spike',
        'hmm_trend', 'hmm_volatility', 'hmm_liquidity', 'hmm_confidence',
        'spy_price', 'theta_decay', 'delta', 'iv_rank'
    ]

    def __init__(self, jsonl_paths: List[str], feature_dim: int = 50,
                 sequence_length: int = 60, max_records_per_file: int = 50000):
        self.samples = []
        self.feature_dim = feature_dim
        self.sequence_length = sequence_length

        for path in jsonl_paths:
            if not os.path.exists(path):
                continue

            logger.info(f"Loading {path}...")
            records_loaded = 0
            with open(path, 'r') as f:
                feature_buffer = deque(maxlen=sequence_length)

                for line in f:
                    if records_loaded >= max_records_per_file:
                        break
                    try:
                        rec = json.loads(line)
                        features = rec.get('features', {})
                        label = rec.get('label', 0)  # 1=profitable, 0=not
                        action = rec.get('proposed_action', '')

                        # Extract feature vector
                        feat_vec = self._extract_features(features)
                        feature_buffer.append(feat_vec)

                        if len(feature_buffer) >= 30:  # Need some history
                            # Create sequence
                            seq = np.array(list(feature_buffer))
                            # Pad if needed
                            if len(seq) < sequence_length:
                                pad = np.zeros((sequence_length - len(seq), self.feature_dim))
                                seq = np.vstack([pad, seq])

                            # Get predicted return for regression
                            pred_return = features.get('predicted_return', 0.0) or 0.0

                            # Direction based on momentum or prediction
                            momentum = features.get('momentum_15m', 0.0) or 0.0
                            if momentum > 0.002:
                                direction = 2  # UP
                            elif momentum < -0.002:
                                direction = 0  # DOWN
                            else:
                                direction = 1  # NEUTRAL

                            self.samples.append({
                                'features': feat_vec,
                                'sequence': seq,
                                'label': float(label),
                                'predicted_return': pred_return,
                                'direction': direction,
                                'action': action,
                            })
                            records_loaded += 1

                    except (json.JSONDecodeError, Exception):
                        continue

        logger.info(f"Loaded {len(self.samples)} samples from tradability datasets")

        # Class balance
        n_pos = sum(1 for s in self.samples if s['label'] == 1)
        logger.info(f"Class balance: {n_pos} positive ({100*n_pos/max(len(self.samples),1):.1f}%)")

    def _extract_features(self, features: dict) -> np.ndarray:
        vec = []
        for name in self.FEATURE_NAMES:
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
        return {
            'features': torch.tensor(sample['features'], dtype=torch.float32),
            'sequence': torch.tensor(sample['sequence'], dtype=torch.float32),
            'label': torch.tensor([sample['label']], dtype=torch.float32),
            'target_return': torch.tensor([sample['predicted_return']], dtype=torch.float32),
            'direction': torch.tensor(sample['direction'], dtype=torch.long),
        }


class FeatureBufferDataset(Dataset):
    """Dataset from saved feature buffer pickle files."""

    def __init__(self, buffer_paths: List[str], feature_dim: int = 50,
                 sequence_length: int = 60):
        self.samples = []
        self.feature_dim = feature_dim
        self.sequence_length = sequence_length

        for path in buffer_paths:
            try:
                with open(path, 'rb') as f:
                    buffer = pickle.load(f)

                if not isinstance(buffer, (list, deque)):
                    continue

                buffer = list(buffer)
                if len(buffer) < sequence_length:
                    continue

                logger.info(f"Processing {path}: {len(buffer)} samples")

                # Create sequences from buffer
                for i in range(sequence_length, len(buffer)):
                    seq = np.array(buffer[i-sequence_length:i])
                    current = np.array(buffer[i])

                    # Ensure correct shape
                    if seq.shape[-1] < feature_dim:
                        # Pad features
                        pad = np.zeros((seq.shape[0], feature_dim - seq.shape[-1]))
                        seq = np.hstack([seq, pad])
                    seq = seq[:, :feature_dim]

                    if len(current) < feature_dim:
                        current = np.pad(current, (0, feature_dim - len(current)))
                    current = current[:feature_dim]

                    # Use next sample's return as label
                    if i + 1 < len(buffer):
                        next_sample = np.array(buffer[i + 1])
                        # Approximate return from momentum features
                        target_return = float(next_sample[4]) if len(next_sample) > 4 else 0.0  # momentum_5m
                    else:
                        target_return = 0.0

                    # Direction
                    if target_return > 0.002:
                        direction = 2
                    elif target_return < -0.002:
                        direction = 0
                    else:
                        direction = 1

                    self.samples.append({
                        'features': current.astype(np.float32),
                        'sequence': seq.astype(np.float32),
                        'target_return': target_return,
                        'direction': direction,
                        'label': 1.0 if target_return > 0 else 0.0,
                    })

            except Exception as e:
                logger.warning(f"Error loading {path}: {e}")
                continue

        logger.info(f"Loaded {len(self.samples)} samples from feature buffers")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        return {
            'features': torch.tensor(sample['features'], dtype=torch.float32),
            'sequence': torch.tensor(sample['sequence'], dtype=torch.float32),
            'label': torch.tensor([sample['label']], dtype=torch.float32),
            'target_return': torch.tensor([sample['target_return']], dtype=torch.float32),
            'direction': torch.tensor(sample['direction'], dtype=torch.long),
        }


# ==============================================================================
# Training Logic
# ==============================================================================

def find_data_files(models_dir: str = 'models') -> Tuple[List[str], List[str]]:
    """Find all training data files."""
    tradability_files = []
    buffer_files = []

    for root, dirs, files in os.walk(models_dir):
        for f in files:
            full_path = os.path.join(root, f)
            if f == 'tradability_gate_dataset.jsonl':
                tradability_files.append(full_path)
            elif f == 'feature_buffer.pkl':
                buffer_files.append(full_path)

    return tradability_files, buffer_files


def pretrain_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int = 50,
    lr: float = 0.0003,
    device: str = 'cpu',
    patience: int = 10,
) -> Dict:
    """
    Pretrain model with multi-task loss:
    - Return regression (MSE)
    - Direction classification (CrossEntropy)
    - Confidence/Profitability (BCE)
    """
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

    # Loss functions
    mse_loss = nn.MSELoss()
    ce_loss = nn.CrossEntropyLoss(label_smoothing=0.1)
    bce_loss = nn.BCELoss()

    # Loss weights
    w_return = 1.0
    w_direction = 1.0
    w_confidence = 1.0

    history = {
        'train_loss': [], 'val_loss': [],
        'val_direction_acc': [], 'val_confidence_acc': []
    }

    best_val_loss = float('inf')
    best_state = None
    patience_counter = 0

    for epoch in range(epochs):
        # Training
        model.train()
        train_losses = []

        for batch in train_loader:
            features = batch['features'].to(device)
            sequence = batch['sequence'].to(device)
            target_return = batch['target_return'].to(device)
            direction = batch['direction'].to(device)
            label = batch['label'].to(device)

            optimizer.zero_grad()

            output = model(features, sequence)

            # Multi-task loss
            # Note: Output keys are 'return', 'direction', 'confidence' (not *_mean, *_probs)
            loss_return = mse_loss(output['return'], target_return)
            loss_direction = ce_loss(output['direction'], direction)
            loss_confidence = bce_loss(output['confidence'], label)

            loss = w_return * loss_return + w_direction * loss_direction + w_confidence * loss_confidence

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_losses.append(loss.item())

        avg_train_loss = np.mean(train_losses)
        history['train_loss'].append(avg_train_loss)

        # Validation
        model.eval()
        val_losses = []
        direction_correct = 0
        confidence_correct = 0
        total = 0

        with torch.no_grad():
            for batch in val_loader:
                features = batch['features'].to(device)
                sequence = batch['sequence'].to(device)
                target_return = batch['target_return'].to(device)
                direction = batch['direction'].to(device)
                label = batch['label'].to(device)

                output = model(features, sequence)

                loss_return = mse_loss(output['return'], target_return)
                loss_direction = ce_loss(output['direction'], direction)
                loss_confidence = bce_loss(output['confidence'], label)

                loss = w_return * loss_return + w_direction * loss_direction + w_confidence * loss_confidence
                val_losses.append(loss.item())

                # Accuracy
                pred_dir = output['direction'].argmax(dim=1)
                direction_correct += (pred_dir == direction).sum().item()

                pred_conf = (output['confidence'] > 0.5).float()
                confidence_correct += (pred_conf == label).sum().item()

                total += direction.size(0)

        avg_val_loss = np.mean(val_losses)
        dir_acc = direction_correct / max(total, 1)
        conf_acc = confidence_correct / max(total, 1)

        history['val_loss'].append(avg_val_loss)
        history['val_direction_acc'].append(dir_acc)
        history['val_confidence_acc'].append(conf_acc)

        scheduler.step(avg_val_loss)

        # Early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        if (epoch + 1) % 5 == 0:
            logger.info(f"Epoch {epoch+1}/{epochs}: train={avg_train_loss:.4f}, val={avg_val_loss:.4f}, "
                       f"dir_acc={dir_acc:.1%}, conf_acc={conf_acc:.1%}")

        if patience_counter >= patience:
            logger.info(f"Early stopping at epoch {epoch+1}")
            break

    # Restore best
    if best_state is not None:
        model.load_state_dict(best_state)
        logger.info(f"Restored best model with val_loss={best_val_loss:.4f}")

    return history


def main():
    parser = argparse.ArgumentParser(description='Pretrain UnifiedOptionsPredictor')
    parser.add_argument('--epochs', type=int, default=50, help='Training epochs')
    parser.add_argument('--batch-size', type=int, default=64, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.0003, help='Learning rate')
    parser.add_argument('--output', type=str, default='models/pretrained/predictor.pt',
                        help='Output model path')
    parser.add_argument('--feature-dim', type=int, default=50, help='Feature dimension')
    parser.add_argument('--val-split', type=float, default=0.2, help='Validation split')
    parser.add_argument('--max-samples', type=int, default=500000, help='Max samples to use')
    parser.add_argument('--models-dir', type=str, default='models', help='Models directory')
    args = parser.parse_args()

    # Get temporal encoder from env
    encoder_type = os.environ.get('TEMPORAL_ENCODER', 'tcn')

    logger.info("=" * 70)
    logger.info("UNIFIED OPTIONS PREDICTOR PRETRAINING")
    logger.info("=" * 70)
    logger.info(f"Temporal Encoder: {encoder_type}")
    logger.info(f"Epochs: {args.epochs}, Batch size: {args.batch_size}, LR: {args.lr}")
    logger.info(f"Output: {args.output}")

    # Find data files
    tradability_files, buffer_files = find_data_files(args.models_dir)
    logger.info(f"Found {len(tradability_files)} tradability files, {len(buffer_files)} feature buffers")

    if not tradability_files and not buffer_files:
        logger.error("No training data found!")
        return

    # Create datasets
    datasets = []

    if tradability_files:
        trad_dataset = TradabilityDataset(
            tradability_files,
            feature_dim=args.feature_dim,
            max_records_per_file=args.max_samples // max(len(tradability_files), 1)
        )
        if len(trad_dataset) > 0:
            datasets.append(trad_dataset)

    if buffer_files:
        # Limit buffer files to avoid memory issues
        buffer_files = buffer_files[:100]
        buf_dataset = FeatureBufferDataset(
            buffer_files,
            feature_dim=args.feature_dim
        )
        if len(buf_dataset) > 0:
            datasets.append(buf_dataset)

    if not datasets:
        logger.error("No valid training samples found!")
        return

    # Combine datasets
    full_dataset = ConcatDataset(datasets) if len(datasets) > 1 else datasets[0]

    # Limit total samples
    if len(full_dataset) > args.max_samples:
        indices = np.random.choice(len(full_dataset), args.max_samples, replace=False)
        full_dataset = torch.utils.data.Subset(full_dataset, indices)

    logger.info(f"Total training samples: {len(full_dataset)}")

    # Split train/val
    val_size = int(len(full_dataset) * args.val_split)
    train_size = len(full_dataset) - val_size

    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size]
    )

    logger.info(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    # Create model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"Using device: {device}")

    # Set encoder type via environment
    os.environ['TEMPORAL_ENCODER'] = encoder_type

    model = UnifiedOptionsPredictor(
        feature_dim=args.feature_dim,
        sequence_length=60
    )

    # Log encoder type
    encoder_class = type(model.temporal_encoder).__name__
    logger.info(f"Created predictor with {encoder_class} temporal encoder")

    # Pretrain
    history = pretrain_model(
        model, train_loader, val_loader,
        epochs=args.epochs, lr=args.lr, device=device
    )

    # Save
    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)

    # Save as trained_model.pth format expected by LOAD_PRETRAINED
    torch.save(model.state_dict(), args.output)

    # Also save metadata
    metadata_path = args.output.replace('.pt', '_metadata.json').replace('.pth', '_metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump({
            'feature_dim': args.feature_dim,
            'temporal_encoder': encoder_type,
            'encoder_class': encoder_class,
            'epochs': args.epochs,
            'final_val_loss': history['val_loss'][-1],
            'final_dir_acc': history['val_direction_acc'][-1],
            'final_conf_acc': history['val_confidence_acc'][-1],
            'train_samples': len(train_dataset),
            'timestamp': datetime.now().isoformat(),
        }, f, indent=2)

    logger.info(f"\nModel saved to {args.output}")
    logger.info(f"Metadata saved to {metadata_path}")

    # Summary
    logger.info("\n" + "=" * 70)
    logger.info("PRETRAINING COMPLETE")
    logger.info("=" * 70)
    logger.info(f"Temporal Encoder: {encoder_type} ({encoder_class})")
    logger.info(f"Final val_loss: {history['val_loss'][-1]:.4f}")
    logger.info(f"Direction accuracy: {history['val_direction_acc'][-1]:.1%}")
    logger.info(f"Confidence accuracy: {history['val_confidence_acc'][-1]:.1%}")
    logger.info(f"\nTo use this model:")
    logger.info(f"  LOAD_PRETRAINED=1 PRETRAINED_MODEL_PATH={args.output} python scripts/train_time_travel.py")


if __name__ == '__main__':
    main()
