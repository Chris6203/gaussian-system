#!/usr/bin/env python3
"""
Train Tradability Gate
======================

Trains a small MLP to predict whether a proposed trade is "tradable" (likely to
beat friction) over the configured horizon.

Input dataset:
  - JSONL rows produced by scripts/train_time_travel.py:
    state/tradability_gate_dataset.jsonl

Output:
  - models/tradability_gate.pt

Usage:
  python scripts/train_tradability_gate.py --dataset models/run_YYYYMMDD_HHMMSS/state/tradability_gate_dataset.jsonl
  python scripts/train_tradability_gate.py --dataset-glob "models/run_*/state/tradability_gate_dataset.jsonl"
"""

import argparse
import glob
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# Ensure output3/ is on sys.path so we import output3/backend/*
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from backend.tradability_gate import FEATURE_NAMES, TradabilityGateNet


class JsonlGateDataset(Dataset):
    def __init__(self, rows, feature_names):
        self.rows = rows
        self.feature_names = feature_names

        X = []
        y = []
        for r in rows:
            feats = r.get("features", {})
            X.append([float(feats.get(k, 0.0) or 0.0) for k in feature_names])
            y.append(int(r.get("label", 0)))
        self.X = np.asarray(X, dtype=np.float32)
        self.y = np.asarray(y, dtype=np.int64)

        self.mean = self.X.mean(axis=0)
        self.std = np.clip(self.X.std(axis=0), 1e-6, None)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        x = (self.X[idx] - self.mean) / self.std
        return torch.from_numpy(x).float(), torch.tensor(self.y[idx]).float()


def load_rows(paths):
    rows = []
    for p in paths:
        with open(p, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except Exception:
                    continue
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=str, default=None, help="Path to a single dataset jsonl")
    ap.add_argument("--dataset-glob", type=str, default=None, help="Glob for multiple dataset jsonl files")
    ap.add_argument("--out", type=str, default="models/tradability_gate.pt", help="Output checkpoint path")
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--hidden-dim", type=int, default=32)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--device", type=str, default="cpu")
    args = ap.parse_args()

    paths = []
    if args.dataset:
        paths = [args.dataset]
    elif args.dataset_glob:
        paths = sorted(glob.glob(args.dataset_glob))
    else:
        # default: latest run datasets
        paths = sorted(glob.glob("models/run_*/state/tradability_gate_dataset.jsonl"))

    if not paths:
        raise SystemExit("No dataset files found.")

    rows = load_rows(paths)
    if len(rows) < 50:
        raise SystemExit(f"Too few rows: {len(rows)} (need at least 50).")
    if len(rows) < 500:
        print(f"[WARN] Small dataset ({len(rows)} rows). Gate may be noisy; run more time-travel to improve.")

    # Shuffle and split
    rng = np.random.default_rng(42)
    idx = rng.permutation(len(rows))
    split = int(0.8 * len(rows))
    train_rows = [rows[i] for i in idx[:split]]
    val_rows = [rows[i] for i in idx[split:]]

    ds_train = JsonlGateDataset(train_rows, FEATURE_NAMES)
    ds_val = JsonlGateDataset(val_rows, FEATURE_NAMES)

    dl_train = DataLoader(ds_train, batch_size=args.batch_size, shuffle=True)
    dl_val = DataLoader(ds_val, batch_size=args.batch_size, shuffle=False)

    device = torch.device(args.device)
    model = TradabilityGateNet(input_dim=len(FEATURE_NAMES), hidden_dim=args.hidden_dim, dropout=args.dropout).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    loss_fn = nn.BCEWithLogitsLoss()

    def eval_loader(dl):
        model.eval()
        losses = []
        correct = 0
        total = 0
        with torch.no_grad():
            for x, y in dl:
                x = x.to(device)
                y = y.to(device)
                logits = model(x).squeeze(-1)
                loss = loss_fn(logits, y)
                losses.append(loss.item())
                probs = torch.sigmoid(logits)
                preds = (probs >= 0.5).float()
                correct += (preds == y).sum().item()
                total += y.numel()
        return float(np.mean(losses)), (correct / total if total else 0.0)

    for epoch in range(1, args.epochs + 1):
        model.train()
        for x, y in dl_train:
            x = x.to(device)
            y = y.to(device)
            logits = model(x).squeeze(-1)
            loss = loss_fn(logits, y)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

        tr_loss, tr_acc = eval_loader(dl_train)
        va_loss, va_acc = eval_loader(dl_val)
        print(f"Epoch {epoch:02d} | train loss {tr_loss:.4f} acc {tr_acc:.3f} | val loss {va_loss:.4f} acc {va_acc:.3f}")

    # Save checkpoint (includes normalization)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "feature_names": list(FEATURE_NAMES),
            "mean": ds_train.mean.astype(np.float32),
            "std": ds_train.std.astype(np.float32),
            "hidden_dim": int(args.hidden_dim),
            "dropout": float(args.dropout),
            "dataset_files": paths,
            "num_rows": len(rows),
        },
        str(out_path),
    )
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()





