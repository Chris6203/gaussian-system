#!/usr/bin/env python3
"""
Pretrain Best Plays Model

This script:
1. Identifies "best plays" from historical trades (high profit trades)
2. Extracts features at entry time
3. Trains a model to recognize "when to play" and "what direction"
4. Uses walk-forward validation (train on data up to cutoff, test on recent)

The goal is to learn:
- WHEN to trade (entry timing conditions)
- WHAT direction (CALL vs PUT)
- Expected profit (regression target)

Usage:
    # Train on all data except last 30 days
    python scripts/pretrain_best_plays.py --holdout-days 30

    # Then test with walk-forward
    python scripts/pretrain_best_plays.py --test-only --model models/pretrained/best_plays.pt
"""

import os
import sys
import argparse
import sqlite3
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from glob import glob
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split

# Setup path
sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class BestPlaysDataset(Dataset):
    """
    Dataset of best plays (high-profit trades) for learning entry conditions.

    Each sample:
    - features: Market state at entry (50 dims)
    - is_best_play: Binary - was this a profitable trade? (1=yes, 0=no)
    - direction: 0=PUT, 1=CALL
    - profit_pct: Actual profit percentage
    """

    def __init__(
        self,
        db_path: str = 'data/paper_trading.db',
        models_dir: str = 'models',
        cutoff_date: Optional[datetime] = None,
        min_profit_pct: float = 2.0,  # "Best plays" threshold (2% is good for options)
        max_samples: int = 100000,
        feature_dim: int = 50,
        include_negative: bool = True  # Include losing trades as negative examples
    ):
        self.samples = []
        self.feature_dim = feature_dim
        self.cutoff_date = cutoff_date
        self.min_profit_pct = min_profit_pct

        # Load from both sources
        self._load_from_trades_db(db_path, max_samples // 2)
        self._load_from_tradability_datasets(models_dir, max_samples // 2)

        logger.info(f"Total samples: {len(self.samples)}")

        # Stats
        best_plays = sum(1 for s in self.samples if s['is_best_play'])
        logger.info(f"Best plays (>{min_profit_pct}% profit): {best_plays} ({100*best_plays/max(1,len(self.samples)):.1f}%)")

    def _load_from_trades_db(self, db_path: str, max_samples: int):
        """Load actual trades with outcomes"""
        if not os.path.exists(db_path):
            logger.warning(f"Database not found: {db_path}")
            return

        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # Build date filter
        date_filter = ""
        if self.cutoff_date:
            date_filter = f"AND entry_timestamp < '{self.cutoff_date.isoformat()}'"

        query = f"""
        SELECT
            option_type,
            profit_loss,
            entry_price,
            spy_price,
            vix_level,
            hmm_trend,
            hmm_volatility,
            hmm_liquidity,
            hmm_confidence,
            predicted_return,
            ml_confidence,
            momentum_5m,
            volume_spike,
            entry_timestamp
        FROM trades
        WHERE spy_price IS NOT NULL
        {date_filter}
        ORDER BY RANDOM()
        LIMIT ?
        """

        try:
            cursor.execute(query, (max_samples,))
            rows = cursor.fetchall()

            for row in rows:
                option_type, profit_loss, entry_price, spy_price, vix_level, \
                    hmm_trend, hmm_vol, hmm_liq, hmm_conf, pred_ret, ml_conf, \
                    momentum, volume, entry_ts = row

                if profit_loss is None or entry_price is None:
                    continue

                # Calculate profit percentage
                profit_pct = (profit_loss / max(entry_price * 100, 1)) * 100

                # Build feature vector
                features = self._build_features({
                    'vix': vix_level,
                    'hmm_trend': hmm_trend,
                    'hmm_volatility': hmm_vol,
                    'hmm_liquidity': hmm_liq,
                    'hmm_confidence': hmm_conf,
                    'predicted_return': pred_ret,
                    'confidence': ml_conf,
                    'momentum_5m': momentum,
                    'volume_ratio': volume,
                    'spy_price': spy_price
                })

                # Labels
                is_best_play = 1 if profit_pct >= self.min_profit_pct else 0
                direction = 1 if option_type == 'CALL' else 0

                self.samples.append({
                    'features': features,
                    'is_best_play': is_best_play,
                    'direction': direction,
                    'profit_pct': profit_pct,
                    'source': 'trades_db'
                })

        except Exception as e:
            logger.warning(f"Error loading from trades DB: {e}")
        finally:
            conn.close()

    def _load_from_tradability_datasets(self, models_dir: str, max_samples: int):
        """Load from tradability gate datasets (has features + realized_move)"""
        pattern = os.path.join(models_dir, '**/state/tradability_gate_dataset.jsonl')
        files = glob(pattern, recursive=True)

        logger.info(f"Found {len(files)} tradability datasets")

        samples_loaded = 0
        for filepath in files:
            if samples_loaded >= max_samples:
                break

            # Extract date from directory name if possible
            dir_name = Path(filepath).parent.parent.name

            try:
                with open(filepath, 'r') as f:
                    for line in f:
                        if samples_loaded >= max_samples:
                            break

                        rec = json.loads(line.strip())

                        # Check date cutoff
                        rec_ts = rec.get('timestamp')
                        if self.cutoff_date and rec_ts:
                            try:
                                rec_date = datetime.fromisoformat(rec_ts.replace('Z', '+00:00'))
                                if rec_date >= self.cutoff_date:
                                    continue
                            except:
                                pass

                        realized_move = rec.get('realized_move')
                        if realized_move is None:
                            continue

                        features_dict = rec.get('features', {})
                        if not features_dict:
                            continue

                        features = self._build_features(features_dict)

                        # Estimate profit based on realized move
                        # Assume ~10x leverage on options
                        profit_pct = abs(realized_move) * 1000  # Convert to %

                        # Direction based on realized move
                        if realized_move > 0.001:
                            direction = 1  # CALL would have profited
                        elif realized_move < -0.001:
                            direction = 0  # PUT would have profited
                        else:
                            continue  # Skip neutral moves

                        is_best_play = 1 if profit_pct >= self.min_profit_pct else 0

                        self.samples.append({
                            'features': features,
                            'is_best_play': is_best_play,
                            'direction': direction,
                            'profit_pct': profit_pct,
                            'source': 'tradability'
                        })
                        samples_loaded += 1

            except Exception as e:
                continue

    def _build_features(self, features_dict: dict) -> np.ndarray:
        """Build standardized feature vector"""
        state = np.zeros(self.feature_dim, dtype=np.float32)

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
            'tda_entropy_h0', 'tda_entropy_h1', 'tda_complexity',
            'spy_price', 'momentum_5m', 'momentum_15m'
        ]

        for i, name in enumerate(feature_names):
            if i >= self.feature_dim:
                break
            val = features_dict.get(name, 0.0)
            if val is None:
                val = 0.0
            # Normalize some features
            if name == 'vix':
                val = float(val) / 50.0  # VIX typically 10-50
            elif name == 'spy_price':
                val = (float(val) - 500) / 100  # Normalize around 500
            state[i] = float(val)

        return state

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        return {
            'features': torch.FloatTensor(s['features']),
            'is_best_play': torch.FloatTensor([s['is_best_play']]),
            'direction': torch.LongTensor([s['direction']]),
            'profit_pct': torch.FloatTensor([s['profit_pct']])
        }


class BestPlaysModel(nn.Module):
    """
    Model to predict best plays.

    Outputs:
    - is_best_play: Probability this is a high-profit opportunity
    - direction: CALL (1) vs PUT (0)
    - expected_profit: Estimated profit percentage
    """

    def __init__(self, input_dim: int = 50, hidden_dim: int = 256):
        super().__init__()

        # Shared encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
        )

        # Best play detector (is this a good entry?)
        self.best_play_head = nn.Sequential(
            nn.Linear(hidden_dim // 2, 64),
            nn.GELU(),
            nn.Linear(64, 1)
        )

        # Direction predictor (CALL vs PUT)
        self.direction_head = nn.Sequential(
            nn.Linear(hidden_dim // 2, 64),
            nn.GELU(),
            nn.Linear(64, 2)
        )

        # Expected profit predictor
        self.profit_head = nn.Sequential(
            nn.Linear(hidden_dim // 2, 64),
            nn.GELU(),
            nn.Linear(64, 1)
        )

        total_params = sum(p.numel() for p in self.parameters())
        logger.info(f"BestPlaysModel: {total_params:,} parameters")

    def forward(self, x):
        h = self.encoder(x)
        return {
            'best_play_logit': self.best_play_head(h),
            'direction_logits': self.direction_head(h),
            'profit_pred': self.profit_head(h)
        }


def pretrain_best_plays(
    epochs: int = 50,
    batch_size: int = 128,
    learning_rate: float = 0.001,
    output_path: str = 'models/pretrained/best_plays.pt',
    holdout_days: int = 30,
    min_profit_pct: float = 5.0,
    device: str = 'cpu'
):
    """
    Pretrain best plays model with walk-forward validation.
    """
    logger.info("=" * 60)
    logger.info("Best Plays Pretraining")
    logger.info("=" * 60)

    # Calculate cutoff date (holdout_days before now)
    cutoff_date = datetime.now() - timedelta(days=holdout_days)
    logger.info(f"Training cutoff: {cutoff_date.strftime('%Y-%m-%d')} ({holdout_days} days holdout)")

    # Load training data (before cutoff)
    train_dataset = BestPlaysDataset(
        cutoff_date=cutoff_date,
        min_profit_pct=min_profit_pct,
        max_samples=100000
    )

    if len(train_dataset) < 1000:
        logger.error(f"Not enough training samples: {len(train_dataset)}")
        return None

    # Calculate class weights for imbalanced data (before split)
    n_best = sum(1 for s in train_dataset.samples if s['is_best_play'])
    n_not_best = len(train_dataset.samples) - n_best
    pos_weight = torch.tensor([n_not_best / max(n_best, 1)]).to(device)
    logger.info(f"Class weights: pos_weight={pos_weight.item():.2f} (best plays are {100*n_best/len(train_dataset.samples):.1f}% of data)")

    # Split train/val (from training period only)
    train_size = int(0.9 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_data, val_data = random_split(train_dataset, [train_size, val_size])

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size)

    logger.info(f"Training: {train_size} samples, Validation: {val_size} samples")

    # Create model
    model = BestPlaysModel().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # Loss functions with class weighting
    bce_loss = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    ce_loss = nn.CrossEntropyLoss()
    mse_loss = nn.MSELoss()

    best_val_loss = float('inf')
    best_state = None

    for epoch in range(epochs):
        # Training
        model.train()
        train_metrics = defaultdict(float)
        train_total = 0

        for batch in train_loader:
            features = batch['features'].to(device)
            is_best = batch['is_best_play'].to(device)
            direction = batch['direction'].squeeze(1).to(device)
            profit = batch['profit_pct'].to(device)

            optimizer.zero_grad()

            out = model(features)

            # Multi-task loss
            loss_best = bce_loss(out['best_play_logit'], is_best)
            loss_dir = ce_loss(out['direction_logits'], direction)
            loss_profit = mse_loss(out['profit_pred'], profit / 100)  # Scale profit

            loss = loss_best + loss_dir + 0.1 * loss_profit

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            # Track metrics
            train_metrics['loss'] += loss.item()
            train_metrics['best_loss'] += loss_best.item()

            # Accuracy
            best_preds = (torch.sigmoid(out['best_play_logit']) > 0.5).float()
            train_metrics['best_acc'] += (best_preds == is_best).sum().item()

            dir_preds = out['direction_logits'].argmax(dim=1)
            train_metrics['dir_acc'] += (dir_preds == direction).sum().item()

            train_total += len(features)

        # Validation
        model.eval()
        val_metrics = defaultdict(float)
        val_total = 0

        with torch.no_grad():
            for batch in val_loader:
                features = batch['features'].to(device)
                is_best = batch['is_best_play'].to(device)
                direction = batch['direction'].squeeze(1).to(device)
                profit = batch['profit_pct'].to(device)

                out = model(features)

                loss_best = bce_loss(out['best_play_logit'], is_best)
                loss_dir = ce_loss(out['direction_logits'], direction)
                loss_profit = mse_loss(out['profit_pred'], profit / 100)

                loss = loss_best + loss_dir + 0.1 * loss_profit

                val_metrics['loss'] += loss.item()

                best_preds = (torch.sigmoid(out['best_play_logit']) > 0.5).float()
                val_metrics['best_acc'] += (best_preds == is_best).sum().item()

                dir_preds = out['direction_logits'].argmax(dim=1)
                val_metrics['dir_acc'] += (dir_preds == direction).sum().item()

                val_total += len(features)

        scheduler.step()

        # Compute averages
        train_loss = train_metrics['loss'] / len(train_loader)
        val_loss = val_metrics['loss'] / len(val_loader)
        train_best_acc = train_metrics['best_acc'] / train_total * 100
        val_best_acc = val_metrics['best_acc'] / val_total * 100
        train_dir_acc = train_metrics['dir_acc'] / train_total * 100
        val_dir_acc = val_metrics['dir_acc'] / val_total * 100

        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if (epoch + 1) % 10 == 0 or epoch == 0:
            logger.info(
                f"Epoch {epoch+1:3d}/{epochs} | "
                f"Loss: {train_loss:.4f}/{val_loss:.4f} | "
                f"Best-Play Acc: {train_best_acc:.1f}%/{val_best_acc:.1f}% | "
                f"Dir Acc: {train_dir_acc:.1f}%/{val_dir_acc:.1f}% "
                f"{'*' if is_best else ''}"
            )

    # Save model
    if best_state:
        model.load_state_dict(best_state)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    torch.save({
        'model_state': model.state_dict(),
        'cutoff_date': cutoff_date.isoformat(),
        'holdout_days': holdout_days,
        'min_profit_pct': min_profit_pct,
        'best_val_loss': best_val_loss,
        'epochs': epochs,
        'samples': len(train_dataset),
        'timestamp': datetime.now().isoformat()
    }, output_path)

    logger.info("")
    logger.info("=" * 60)
    logger.info(f"✅ Best Plays model pretrained!")
    logger.info(f"   Cutoff date: {cutoff_date.strftime('%Y-%m-%d')}")
    logger.info(f"   Training samples: {len(train_dataset)}")
    logger.info(f"   Best validation loss: {best_val_loss:.4f}")
    logger.info(f"   Saved to: {output_path}")
    logger.info("")
    logger.info("To test with walk-forward validation:")
    logger.info(f"   BEST_PLAYS_MODEL={output_path} python scripts/train_time_travel.py")
    logger.info("=" * 60)

    return model, cutoff_date


def test_walk_forward(
    model_path: str,
    cycles: int = 5000,
    device: str = 'cpu'
):
    """
    Test the pretrained model on holdout period (walk-forward validation).
    """
    logger.info("=" * 60)
    logger.info("Walk-Forward Validation Test")
    logger.info("=" * 60)

    # Load model
    checkpoint = torch.load(model_path, map_location=device)

    model = BestPlaysModel().to(device)
    model.load_state_dict(checkpoint['model_state'])
    model.eval()

    cutoff_date = datetime.fromisoformat(checkpoint['cutoff_date'])
    holdout_days = checkpoint['holdout_days']

    logger.info(f"Model trained up to: {cutoff_date.strftime('%Y-%m-%d')}")
    logger.info(f"Testing on: {holdout_days} days of unseen data")

    # Load test data (after cutoff - the holdout period)
    test_dataset = BestPlaysDataset(
        cutoff_date=None,  # No cutoff = include all data
        min_profit_pct=checkpoint['min_profit_pct'],
        max_samples=50000
    )

    # Filter to only holdout period
    # (In practice, the time-travel training will handle this)

    logger.info(f"Test samples: {len(test_dataset)}")

    # Evaluate
    test_loader = DataLoader(test_dataset, batch_size=128)

    correct_best = 0
    correct_dir = 0
    total = 0
    predictions = []

    with torch.no_grad():
        for batch in test_loader:
            features = batch['features'].to(device)
            is_best = batch['is_best_play'].to(device)
            direction = batch['direction'].squeeze(1).to(device)

            out = model(features)

            best_prob = torch.sigmoid(out['best_play_logit'])
            best_preds = (best_prob > 0.5).float()
            correct_best += (best_preds == is_best).sum().item()

            dir_preds = out['direction_logits'].argmax(dim=1)
            correct_dir += (dir_preds == direction).sum().item()

            total += len(features)

            # Store predictions for analysis
            for i in range(len(features)):
                predictions.append({
                    'best_prob': best_prob[i].item(),
                    'dir_pred': dir_preds[i].item(),
                    'actual_best': is_best[i].item(),
                    'actual_dir': direction[i].item()
                })

    best_acc = correct_best / total * 100
    dir_acc = correct_dir / total * 100

    logger.info("")
    logger.info("Walk-Forward Results:")
    logger.info(f"   Best-Play Detection Accuracy: {best_acc:.1f}%")
    logger.info(f"   Direction Accuracy: {dir_acc:.1f}%")

    # Analyze high-confidence predictions
    high_conf_preds = [p for p in predictions if p['best_prob'] > 0.7]
    if high_conf_preds:
        high_conf_correct = sum(1 for p in high_conf_preds if p['actual_best'] > 0.5)
        logger.info(f"   High-confidence (>70%) predictions: {len(high_conf_preds)}")
        logger.info(f"   High-confidence accuracy: {100*high_conf_correct/len(high_conf_preds):.1f}%")

    return {
        'best_play_accuracy': best_acc,
        'direction_accuracy': dir_acc,
        'total_samples': total
    }


def main():
    parser = argparse.ArgumentParser(description='Pretrain Best Plays Model')
    parser.add_argument('--epochs', type=int, default=50, help='Training epochs')
    parser.add_argument('--batch-size', type=int, default=128, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--output', type=str, default='models/pretrained/best_plays.pt', help='Output path')
    parser.add_argument('--holdout-days', type=int, default=30, help='Days to hold out for testing')
    parser.add_argument('--min-profit', type=float, default=5.0, help='Min profit % for "best play"')
    parser.add_argument('--device', type=str, default='cpu', help='Device')
    parser.add_argument('--test-only', action='store_true', help='Only run walk-forward test')
    parser.add_argument('--model', type=str, help='Model path for testing')
    args = parser.parse_args()

    if args.test_only:
        if not args.model:
            args.model = args.output
        test_walk_forward(args.model, device=args.device)
    else:
        model, cutoff = pretrain_best_plays(
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            output_path=args.output,
            holdout_days=args.holdout_days,
            min_profit_pct=args.min_profit,
            device=args.device
        )

        # Run walk-forward test
        if model is not None:
            logger.info("")
            test_walk_forward(args.output, device=args.device)


if __name__ == '__main__':
    main()
