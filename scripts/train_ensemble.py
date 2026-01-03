#!/usr/bin/env python3
"""
Pre-train the Phase 44 Ensemble Predictor on historical data.

This script:
1. Loads historical feature buffers from past runs
2. Calculates price returns as labels
3. Trains TCN, LSTM, and XGBoost components
4. Saves the trained ensemble for use in live trading
"""

import os
import sys
import pickle
import json
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from datetime import datetime
from collections import deque

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from bot_modules.ensemble_predictor import EnsemblePredictor


def load_decision_records(run_dir: str, max_records: int = 50000) -> list:
    """Load decision records from a run directory."""
    records_path = Path(run_dir) / "state" / "decision_records.jsonl"
    if not records_path.exists():
        return []

    records = []
    with open(records_path) as f:
        for i, line in enumerate(f):
            if i >= max_records:
                break
            try:
                record = json.loads(line.strip())
                records.append(record)
            except:
                continue
    return records


def extract_training_data(records: list) -> tuple:
    """
    Extract feature sequences and labels from decision records.

    Returns:
        sequences: List of feature sequences [seq_len, feature_dim]
        labels: List of return labels
    """
    sequences = []
    labels = []

    # Feature buffer for sequence
    feature_buffer = deque(maxlen=60)

    for i, record in enumerate(records):
        # Get feature_vector from decision record
        feature_vec = record.get('feature_vector', [])
        if not feature_vec or len(feature_vec) < 10:
            continue

        # Pad or truncate to 50 features
        if len(feature_vec) < 50:
            feature_vec = list(feature_vec) + [0.0] * (50 - len(feature_vec))
        else:
            feature_vec = list(feature_vec)[:50]

        feature_buffer.append(feature_vec)

        # Get label - use predicted return as proxy for direction
        # We'll predict the 15-min return
        trade_state = record.get('trade_state', {})
        predicted_return = trade_state.get('predicted_return', 0)
        momentum_15m = 0
        for j, name in enumerate(record.get('feature_names', [])):
            if name == 'momentum_15m' and j < len(record.get('feature_vector', [])):
                momentum_15m = record['feature_vector'][j]
                break

        # Use momentum as label (what actually happened)
        label = momentum_15m if momentum_15m != 0 else predicted_return

        # Only use if we have enough history
        if len(feature_buffer) >= 30:
            seq = np.array(list(feature_buffer)[-30:])
            sequences.append(seq)
            labels.append(label)

    return sequences, labels


def train_ensemble(ensemble: EnsemblePredictor, sequences: list, labels: list,
                   epochs: int = 10, batch_size: int = 32):
    """Train the ensemble predictor."""

    if len(sequences) < 100:
        print(f"Not enough data: {len(sequences)} samples (need 100+)")
        return False

    print(f"\nTraining ensemble on {len(sequences)} samples...")

    # Convert to tensors
    X = torch.tensor(np.array(sequences), dtype=torch.float32)
    y = torch.tensor(np.array(labels), dtype=torch.float32)

    # Normalize labels
    y_mean, y_std = y.mean(), y.std()
    if y_std > 0:
        y_norm = (y - y_mean) / y_std
    else:
        y_norm = y

    # Train TCN
    print("\n1. Training TCN...")
    tcn_optimizer = torch.optim.Adam(ensemble.tcn.parameters(), lr=0.001)
    tcn_head = nn.Linear(ensemble.hidden_dim, 1)
    head_optimizer = torch.optim.Adam(tcn_head.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    for epoch in range(epochs):
        total_loss = 0
        n_batches = 0

        indices = torch.randperm(len(X))
        for i in range(0, len(X), batch_size):
            batch_idx = indices[i:i+batch_size]
            batch_x = X[batch_idx]
            batch_y = y_norm[batch_idx]

            tcn_optimizer.zero_grad()
            head_optimizer.zero_grad()

            hidden = ensemble._tcn_forward(batch_x)
            pred = tcn_head(hidden).squeeze()
            loss = criterion(pred, batch_y)

            loss.backward()
            tcn_optimizer.step()
            head_optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        if (epoch + 1) % 2 == 0:
            print(f"   Epoch {epoch+1}/{epochs}: Loss = {total_loss/n_batches:.4f}")

    # Train LSTM
    print("\n2. Training LSTM...")
    lstm_optimizer = torch.optim.Adam(ensemble.lstm.parameters(), lr=0.001)
    attn_optimizer = torch.optim.Adam(ensemble.attention.parameters(), lr=0.001)
    lstm_head = nn.Linear(ensemble.hidden_dim, 1)
    lstm_head_optimizer = torch.optim.Adam(lstm_head.parameters(), lr=0.001)

    for epoch in range(epochs):
        total_loss = 0
        n_batches = 0

        indices = torch.randperm(len(X))
        for i in range(0, len(X), batch_size):
            batch_idx = indices[i:i+batch_size]
            batch_x = X[batch_idx]
            batch_y = y_norm[batch_idx]

            lstm_optimizer.zero_grad()
            attn_optimizer.zero_grad()
            lstm_head_optimizer.zero_grad()

            hidden = ensemble._lstm_forward(batch_x)
            pred = lstm_head(hidden).squeeze()
            loss = criterion(pred, batch_y)

            loss.backward()
            lstm_optimizer.step()
            attn_optimizer.step()
            lstm_head_optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        if (epoch + 1) % 2 == 0:
            print(f"   Epoch {epoch+1}/{epochs}: Loss = {total_loss/n_batches:.4f}")

    # Train XGBoost
    print("\n3. Training XGBoost...")
    X_np = X.numpy()
    y_np = y.numpy()
    ensemble.train_xgboost(X_np, y_np)
    print("   XGBoost trained successfully")

    # Train meta-learner
    print("\n4. Training Meta-Learner...")
    meta_X = []
    meta_y = []

    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            batch_x = X[i:i+batch_size]
            batch_y = y[i:i+batch_size]

            tcn_pred = ensemble._tcn_forward(batch_x).mean(dim=1)
            lstm_pred = ensemble._lstm_forward(batch_x).mean(dim=1)

            for j in range(len(batch_x)):
                seq_np = batch_x[j].numpy()
                xgb_features = ensemble._prepare_xgb_features(seq_np)
                xgb_features_scaled = ensemble.xgb_scaler.transform([xgb_features])
                xgb_pred = ensemble.xgb_model.predict(xgb_features_scaled)[0]

                meta_X.append([tcn_pred[j].item(), lstm_pred[j].item(), xgb_pred])
                meta_y.append(batch_y[j].item())

    meta_X = np.array(meta_X)
    meta_y = np.array(meta_y)

    ensemble.meta_scaler.fit(meta_X)
    meta_X_scaled = ensemble.meta_scaler.transform(meta_X)
    ensemble.meta_learner.fit(meta_X_scaled, meta_y)
    ensemble.is_fitted = True
    print("   Meta-learner trained successfully")

    return True


def save_ensemble(ensemble: EnsemblePredictor, save_path: str):
    """Save trained ensemble to disk."""
    save_dict = {
        'tcn_state': ensemble.tcn.state_dict(),
        'lstm_state': ensemble.lstm.state_dict(),
        'attention_state': ensemble.attention.state_dict(),
        'xgb_model': ensemble.xgb_model,
        'xgb_scaler': ensemble.xgb_scaler,
        'meta_learner': ensemble.meta_learner,
        'meta_scaler': ensemble.meta_scaler,
        'is_fitted': ensemble.is_fitted,
        'feature_dim': ensemble.feature_dim,
        'hidden_dim': ensemble.hidden_dim,
    }

    with open(save_path, 'wb') as f:
        pickle.dump(save_dict, f)
    print(f"\nEnsemble saved to: {save_path}")


def main():
    print("=" * 60)
    print("PHASE 44: ENSEMBLE PRE-TRAINING")
    print("=" * 60)

    # Find runs with decision records
    models_dir = Path("models")
    all_sequences = []
    all_labels = []

    print("\nLoading training data from past runs...")

    # Look for runs with decision records
    for run_dir in sorted(models_dir.glob("*")):
        if not run_dir.is_dir():
            continue

        records_path = run_dir / "state" / "decision_records.jsonl"
        if not records_path.exists():
            continue

        # Get file size to prioritize larger datasets
        size_mb = records_path.stat().st_size / (1024 * 1024)
        if size_mb < 1:  # Skip tiny files
            continue

        print(f"  Loading {run_dir.name} ({size_mb:.1f} MB)...")
        records = load_decision_records(str(run_dir), max_records=10000)

        if len(records) > 100:
            seqs, labels = extract_training_data(records)
            if len(seqs) > 50:
                all_sequences.extend(seqs)
                all_labels.extend(labels)
                print(f"    -> Added {len(seqs)} samples")

    print(f"\nTotal training samples: {len(all_sequences)}")

    if len(all_sequences) < 500:
        print("ERROR: Not enough training data. Run more experiments first.")
        return

    # Limit to prevent memory issues
    max_samples = 50000
    if len(all_sequences) > max_samples:
        indices = np.random.choice(len(all_sequences), max_samples, replace=False)
        all_sequences = [all_sequences[i] for i in indices]
        all_labels = [all_labels[i] for i in indices]
        print(f"Sampled {max_samples} for training")

    # Create and train ensemble
    ensemble = EnsemblePredictor(feature_dim=50, hidden_dim=64, device='cpu')

    success = train_ensemble(ensemble, all_sequences, all_labels, epochs=10, batch_size=64)

    if success:
        # Save ensemble
        save_path = "models/pretrained_ensemble.pkl"
        save_ensemble(ensemble, save_path)

        print("\n" + "=" * 60)
        print("PRE-TRAINING COMPLETE")
        print("=" * 60)
        print(f"\nTo use the pre-trained ensemble:")
        print(f"  ENSEMBLE_ENABLED=1 ENSEMBLE_PRETRAINED=1 python scripts/train_time_travel.py")
    else:
        print("\nPre-training failed.")


if __name__ == "__main__":
    main()
