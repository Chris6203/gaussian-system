#!/usr/bin/env python3
"""
Pretrain AlphaStar RL Policy on Expert Trades

Like DeepMind's approach:
1. AlphaGo: Pretrained on 30M expert Go moves
2. AlphaStar: Pretrained on human StarCraft replays
3. Us: Pretrain on profitable historical trades

This script:
1. Loads historical profitable trades from paper_trading.db
2. Extracts state features at entry time
3. Trains AlphaStar to imitate expert (profitable) actions
4. Saves pretrained weights for RL fine-tuning

Usage:
    python scripts/pretrain_alphastar.py --epochs 50 --output models/alphastar_pretrained.pt
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

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split

# Setup path
sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.alphastar_policy import AlphaStarRLPolicy, AlphaStarPolicyNetwork

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ExpertTradeDataset(Dataset):
    """
    Dataset of profitable trades for expert imitation.

    Each sample contains:
    - state: Market state at entry time (22 features)
    - action: Expert action (CALL=1, PUT=2)
    - profit: Normalized profit (for weighting)
    """

    def __init__(
        self,
        db_path: str = 'data/paper_trading.db',
        min_profit_pct: float = 0.0,  # Only include profitable trades
        max_samples: int = 50000
    ):
        self.samples = []
        self._load_trades(db_path, min_profit_pct, max_samples)

    def _load_trades(self, db_path: str, min_profit_pct: float, max_samples: int):
        """Load profitable trades from database"""
        if not os.path.exists(db_path):
            logger.warning(f"Database not found: {db_path}")
            return

        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # Get profitable trades with available features
        # Note: hmm_* may be NULL for older trades, we'll use defaults
        query = """
        SELECT
            option_type,
            profit_loss,
            spy_price,
            vix_level,
            hmm_trend,
            hmm_volatility,
            hmm_liquidity,
            hmm_confidence,
            predicted_return,
            ml_confidence,
            momentum_5m,
            volume_spike
        FROM trades
        WHERE profit_loss > ?
        AND spy_price IS NOT NULL
        ORDER BY profit_loss DESC
        LIMIT ?
        """

        cursor.execute(query, (min_profit_pct, max_samples))
        rows = cursor.fetchall()

        for row in rows:
            option_type, profit_loss, spy_price, vix_level, hmm_trend, hmm_vol, \
                hmm_liq, hmm_conf, pred_ret, ml_conf, momentum, volume = row

            # Build 22-dim state (matching TradeState)
            state = np.zeros(22, dtype=np.float32)

            # Position (4) - not in trade at entry
            state[0] = 0.0  # is_in_trade
            state[1] = 1.0 if option_type == 'CALL' else 0.0  # is_call
            state[2] = 0.0  # pnl_pct
            state[3] = 0.0  # drawdown

            # Time (2) - at entry
            state[4] = 0.0  # minutes_held
            state[5] = 1.0  # minutes_to_expiry (normalized)

            # Prediction (3)
            state[6] = float(pred_ret or 0) * 100  # predicted_direction
            state[7] = float(ml_conf or 0.5)  # confidence
            state[8] = float(momentum or 0)  # momentum_5m

            # Market (2)
            state[9] = float(vix_level or 18) / 50  # vix normalized
            state[10] = float(volume or 1.0)  # volume_spike

            # HMM Regime (4)
            state[11] = float(hmm_trend or 0.5)
            state[12] = float(hmm_vol or 0.5)
            state[13] = float(hmm_liq or 0.5)
            state[14] = float(hmm_conf or 0.5)

            # Greeks (2)
            state[15] = 0.0  # theta
            state[16] = 0.5  # delta

            # Sentiment (4)
            state[17] = 0.5  # fear_greed
            state[18] = 0.0  # pcr
            state[19] = 0.0  # contrarian
            state[20] = 0.0  # news
            state[21] = 0.0  # padding

            # Action: CALL=1, PUT=2
            action = 1 if option_type == 'CALL' else 2

            # Weight by profit (bigger profits = more important)
            weight = min(float(profit_loss) / 100, 5.0)  # Cap at 5x

            self.samples.append({
                'state': state,
                'action': action,
                'weight': max(weight, 0.1)  # Min weight 0.1
            })

        conn.close()
        logger.info(f"Loaded {len(self.samples)} expert trades from {db_path}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        return {
            'state': torch.FloatTensor(sample['state']),
            'action': torch.LongTensor([sample['action']]),
            'weight': torch.FloatTensor([sample['weight']])
        }


def pretrain_alphastar(
    epochs: int = 50,
    batch_size: int = 64,
    learning_rate: float = 0.0003,
    output_path: str = 'models/alphastar_pretrained.pt',
    db_path: str = 'data/paper_trading.db',
    min_profit_pct: float = 0.0,
    device: str = 'cpu'
):
    """
    Pretrain AlphaStar on expert trades.

    Returns:
        Trained AlphaStarRLPolicy
    """
    logger.info("=" * 60)
    logger.info("AlphaStar Expert Pretraining")
    logger.info("=" * 60)

    # Load dataset
    dataset = ExpertTradeDataset(db_path, min_profit_pct)

    if len(dataset) < 100:
        logger.error(f"Not enough expert trades: {len(dataset)} (need 100+)")
        logger.info("Generate trades first: python scripts/train_time_travel.py")
        return None

    # Split train/val
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    logger.info(f"Dataset: {len(dataset)} samples ({train_size} train, {val_size} val)")

    # Create model
    policy = AlphaStarRLPolicy(state_dim=22, learning_rate=learning_rate, device=device)
    model = policy.network
    model.train()

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # Loss function (weighted cross-entropy)
    criterion = nn.CrossEntropyLoss(reduction='none')

    best_val_acc = 0.0
    best_state = None

    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for batch in train_loader:
            states = batch['state'].to(device)
            actions = batch['action'].squeeze(1).to(device)
            weights = batch['weight'].squeeze(1).to(device)

            optimizer.zero_grad()

            # Forward pass
            out = model.forward(states, use_history=False)
            logits = out['action_logits']

            # Weighted loss
            loss = (criterion(logits, actions) * weights).mean()

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()

            train_loss += loss.item()
            preds = logits.argmax(dim=1)
            train_correct += (preds == actions).sum().item()
            train_total += len(actions)

        # Validation
        model.eval()
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for batch in val_loader:
                states = batch['state'].to(device)
                actions = batch['action'].squeeze(1).to(device)

                out = model.forward(states, use_history=False)
                logits = out['action_logits']
                preds = logits.argmax(dim=1)

                val_correct += (preds == actions).sum().item()
                val_total += len(actions)

        scheduler.step()

        train_acc = train_correct / train_total * 100
        val_acc = val_correct / val_total * 100

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = model.state_dict().copy()

        if (epoch + 1) % 5 == 0 or epoch == 0:
            logger.info(
                f"Epoch {epoch+1:3d}/{epochs} | "
                f"Loss: {train_loss/len(train_loader):.4f} | "
                f"Train Acc: {train_acc:.1f}% | "
                f"Val Acc: {val_acc:.1f}% {'*' if val_acc == best_val_acc else ''}"
            )

    # Save best model
    if best_state is not None:
        model.load_state_dict(best_state)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    torch.save({
        'network_state': model.state_dict(),
        'best_val_acc': best_val_acc,
        'epochs': epochs,
        'samples': len(dataset),
        'timestamp': datetime.now().isoformat()
    }, output_path)

    logger.info(f"")
    logger.info(f"✅ AlphaStar pretrained!")
    logger.info(f"   Best validation accuracy: {best_val_acc:.1f}%")
    logger.info(f"   Saved to: {output_path}")
    logger.info(f"")
    logger.info(f"To use in training:")
    logger.info(f"   RL_ARCHITECTURE=alphastar ALPHASTAR_PRETRAINED={output_path} python scripts/train_time_travel.py")

    return policy


def main():
    parser = argparse.ArgumentParser(description='Pretrain AlphaStar on expert trades')
    parser.add_argument('--epochs', type=int, default=50, help='Training epochs')
    parser.add_argument('--batch-size', type=int, default=64, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.0003, help='Learning rate')
    parser.add_argument('--output', type=str, default='models/alphastar_pretrained.pt', help='Output path')
    parser.add_argument('--db', type=str, default='data/paper_trading.db', help='Database path')
    parser.add_argument('--min-profit', type=float, default=0.0, help='Min profit % to include')
    parser.add_argument('--device', type=str, default='cpu', help='Device (cpu/cuda)')
    args = parser.parse_args()

    pretrain_alphastar(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        output_path=args.output,
        db_path=args.db,
        min_profit_pct=args.min_profit,
        device=args.device
    )


if __name__ == '__main__':
    main()
