# Gaussian Options Trading Bot - Architecture Documentation

## Overview

The Gaussian Options Trading Bot is an ML-powered automated options trading system that combines neural networks, reinforcement learning, Hidden Markov Models, and sophisticated calibration to make intraday trading decisions on SPY/QQQ options.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         GAUSSIAN TRADING SYSTEM                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐ │
│   │   Data      │───▶│   Feature   │───▶│   Signal    │───▶│   Trade     │ │
│   │   Layer     │    │   Engine    │    │   Pipeline  │    │   Execution │ │
│   └─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘ │
│         │                  │                  │                  │          │
│         ▼                  ▼                  ▼                  ▼          │
│   ┌─────────────────────────────────────────────────────────────────────┐  │
│   │                    LEARNING & ADAPTATION LAYER                       │  │
│   │  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐   │  │
│   │  │ Online  │  │   RL    │  │ Calibra-│  │  Regime │  │  Model  │   │  │
│   │  │ Tuning  │  │ Policy  │  │  tion   │  │ Models  │  │  Health │   │  │
│   │  └─────────┘  └─────────┘  └─────────┘  └─────────┘  └─────────┘   │  │
│   └─────────────────────────────────────────────────────────────────────┘  │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Table of Contents

1. [System Components](#1-system-components)
2. [Data Flow](#2-data-flow)
3. [Neural Network Architecture](#3-neural-network-architecture)
4. [Signal Generation Pipeline](#4-signal-generation-pipeline)
5. [Calibration System](#5-calibration-system)
6. [Reinforcement Learning](#6-reinforcement-learning)
7. [Regime Detection](#7-regime-detection)
8. [Trade Execution](#8-trade-execution)
9. [Risk Management](#9-risk-management)
10. [Monitoring & Health](#10-monitoring--health)
11. [File Structure](#11-file-structure)

---

## 1. System Components

### Core Components

| Component | File | Purpose |
|-----------|------|---------|
| **UnifiedOptionsBot** | `unified_options_trading_bot.py` | Main bot orchestrating all ML/trading logic |
| **LiveTradingEngine** | `backend/live_trading_engine.py` | Orchestrates live trading cycles |
| **PaperTradingSystem** | `backend/paper_trading_system.py` | Simulated order execution |
| **TradierTradingSystem** | `backend/tradier_trading_system.py` | Live broker integration |

### ML Components

| Component | File | Purpose |
|-----------|------|---------|
| **UnifiedOptionsPredictor** | `bot_modules/neural_networks.py` | Main neural network model |
| **CalibrationTracker** | `backend/calibration_tracker.py` | Platt/Isotonic calibration |
| **MultiDimensionalHMM** | `backend/multi_dimensional_hmm.py` | Regime detection |
| **RLThresholdLearner** | `backend/rl_threshold_learner.py` | Learns optimal thresholds |
| **RLExitPolicy** | `backend/rl_exit_policy.py` | Learns when to exit positions |

### Infrastructure

| Component | File | Purpose |
|-----------|------|---------|
| **RobustFetcher** | `backend/robust_fetcher.py` | Data fetching with retries/caching |
| **FeatureCache** | `backend/feature_cache.py` | LRU cache for features |
| **HealthCheckSystem** | `backend/health_checks.py` | Pre-cycle validation |
| **TradingMonitor** | `backend/monitoring_dashboard.py` | Position/rejection tracking |

---

## 2. Data Flow

### Per-Cycle Data Flow

```
┌──────────────────────────────────────────────────────────────────────────┐
│                           TRADING CYCLE                                   │
│                         (Every 60 seconds)                                │
└──────────────────────────────────────────────────────────────────────────┘
                                   │
                                   ▼
┌──────────────────────────────────────────────────────────────────────────┐
│ STEP 1: DATA FETCH                                                        │
│ ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐      │
│ │ SPY 1min    │  │ QQQ 1min    │  │ VIX 1min    │  │ Context     │      │
│ │ OHLCV       │  │ OHLCV       │  │ values      │  │ Symbols     │      │
│ └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘      │
│                                                                           │
│ Sources: Tradier API (primary) → Polygon (backup) → Yahoo (fallback)     │
└──────────────────────────────────────────────────────────────────────────┘
                                   │
                                   ▼
┌──────────────────────────────────────────────────────────────────────────┐
│ STEP 2: FEATURE COMPUTATION                                               │
│                                                                           │
│ Price Features:          Technical Indicators:    Cross-Asset:           │
│ • Returns (1m,5m,15m)    • RSI (14,28)           • SPY/QQQ correlation   │
│ • Volatility (ATR)       • MACD                  • Sector ETF flows      │
│ • Price momentum         • Bollinger Bands       • Bond yields (TLT)     │
│ • Volume analysis        • Stochastic            • USD strength (UUP)    │
│                                                                           │
│ Regime Features:         Options Features:        Time Features:         │
│ • HMM state probs        • Implied volatility    • Hour of day           │
│ • Regime transitions     • Put/Call skew         • Day of week           │
│ • Vol regime             • Term structure        • Time to close         │
│                                                                           │
│ Output: 50-dimensional feature vector                                     │
└──────────────────────────────────────────────────────────────────────────┘
                                   │
                                   ▼
┌──────────────────────────────────────────────────────────────────────────┐
│ STEP 3: SIGNAL GENERATION                                                 │
│                                                                           │
│ ┌─────────────────┐   ┌─────────────────┐   ┌─────────────────┐         │
│ │ Neural Network  │   │ Multi-Timeframe │   │ HMM Regime      │         │
│ │ Prediction      │──▶│ Scaling         │──▶│ Adjustment      │         │
│ │                 │   │ (1m,5m,15m)     │   │                 │         │
│ └─────────────────┘   └─────────────────┘   └─────────────────┘         │
│          │                                                                │
│          ▼                                                                │
│ ┌─────────────────────────────────────────────────────────────────────┐ │
│ │ Signal = { action: 'BUY_CALL'|'BUY_PUT'|'HOLD',                     │ │
│ │            direction: -1 to +1,                                      │ │
│ │            raw_confidence: 0-1,                                      │ │
│ │            volatility_forecast: float }                              │ │
│ └─────────────────────────────────────────────────────────────────────┘ │
└──────────────────────────────────────────────────────────────────────────┘
                                   │
                                   ▼
┌──────────────────────────────────────────────────────────────────────────┐
│ STEP 4: CALIBRATION                                                       │
│                                                                           │
│ ┌─────────────────┐   ┌─────────────────┐   ┌─────────────────┐         │
│ │ Raw Confidence  │──▶│ Platt Scaling   │──▶│ Calibrated      │         │
│ │ (from NN)       │   │ + Isotonic      │   │ Confidence      │         │
│ │ e.g., 0.72      │   │ (hybrid 40/60)  │   │ e.g., 0.58      │         │
│ └─────────────────┘   └─────────────────┘   └─────────────────┘         │
│                                                                           │
│ Metrics tracked: Brier Score, ECE, Direction Accuracy                    │
└──────────────────────────────────────────────────────────────────────────┘
                                   │
                                   ▼
┌──────────────────────────────────────────────────────────────────────────┐
│ STEP 5: GATING (Trade/No-Trade Decision)                                  │
│                                                                           │
│ Gate Checks:                                                              │
│ ┌─────────────────────────────────────────────────────────────────────┐ │
│ │ ✓ Calibrated confidence ≥ min_threshold (regime-adjusted)          │ │
│ │ ✓ Brier score < 0.35                                                │ │
│ │ ✓ ECE < 0.15                                                        │ │
│ │ ✓ Multi-horizon agreement (5min + 15min same direction)            │ │
│ │ ✓ Trades this hour < max_trades_per_hour                           │ │
│ │ ✓ Not counter-trend in strong trends                               │ │
│ │ ✓ Sufficient calibration samples (≥50)                             │ │
│ └─────────────────────────────────────────────────────────────────────┘ │
│                                                                           │
│ If ANY gate fails → REJECT (log reason) → Return to cycle                │
│ If ALL gates pass → Proceed to execution                                 │
└──────────────────────────────────────────────────────────────────────────┘
                                   │
                                   ▼
┌──────────────────────────────────────────────────────────────────────────┐
│ STEP 6: TRADE EXECUTION                                                   │
│                                                                           │
│ ┌─────────────────┐   ┌─────────────────┐   ┌─────────────────┐         │
│ │ Contract        │──▶│ Liquidity       │──▶│ Order           │         │
│ │ Selection       │   │ Validation      │   │ Execution       │         │
│ │ (strike, exp)   │   │ (OI, spread)    │   │ (limit order)   │         │
│ └─────────────────┘   └─────────────────┘   └─────────────────┘         │
│                                                                           │
│ Position sizing: Based on calibrated confidence + Kelly-inspired formula │
└──────────────────────────────────────────────────────────────────────────┘
                                   │
                                   ▼
┌──────────────────────────────────────────────────────────────────────────┐
│ STEP 7: POSITION MANAGEMENT                                               │
│                                                                           │
│ Exit Conditions (RL-learned):                                             │
│ • Take profit: +30-60% (regime-adjusted)                                 │
│ • Stop loss: -15-30% (regime-adjusted)                                   │
│ • Trailing stop: locks in profits                                        │
│ • Max hold time: 45-120 minutes (regime-adjusted)                        │
│ • RL exit signal: policy recommends exit                                 │
└──────────────────────────────────────────────────────────────────────────┘
                                   │
                                   ▼
┌──────────────────────────────────────────────────────────────────────────┐
│ STEP 8: LEARNING & ADAPTATION                                             │
│                                                                           │
│ After each trade/cycle:                                                   │
│ • Update calibration buffer (confidence, hit/miss)                       │
│ • RL reward signal to threshold learner                                  │
│ • RL reward signal to exit policy                                        │
│ • Update regime model training data                                      │
│ • Track model health (drift detection)                                   │
│                                                                           │
│ Periodic (every N cycles):                                               │
│ • Refit Platt/Isotonic calibrators                                       │
│ • Retrain neural network (every 20 cycles)                               │
│ • Retrain regime-specific models                                         │
└──────────────────────────────────────────────────────────────────────────┘
```

### Data Sources

```
┌─────────────────────────────────────────────────────────────────────────┐
│                          DATA SOURCES                                    │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  PRIMARY: Tradier API                                                    │
│  ├── Real-time quotes                                                    │
│  ├── 1-minute OHLCV bars                                                │
│  ├── Options chains                                                      │
│  └── Account/positions data                                              │
│                                                                          │
│  BACKUP: Polygon.io                                                      │
│  ├── Historical data                                                     │
│  └── Extended hours data                                                 │
│                                                                          │
│  FALLBACK: Yahoo Finance                                                 │
│  └── Free backup when APIs fail                                          │
│                                                                          │
│  LOCAL CACHE: SQLite (data/market_data.db)                              │
│  ├── Historical bars                                                     │
│  ├── Incremental updates only                                            │
│  └── Used for training                                                   │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Neural Network Architecture

### UnifiedOptionsPredictor

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    UNIFIED OPTIONS PREDICTOR                             │
│                      (Main Neural Network)                               │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  Input: 50-dimensional feature vector                                    │
│                                                                          │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │                    FEATURE PROCESSING                            │    │
│  │                                                                   │    │
│  │  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐          │    │
│  │  │ Batch Norm  │───▶│ RBF Kernel  │───▶│ Gaussian    │          │    │
│  │  │             │    │ Layer       │    │ Processor   │          │    │
│  │  └─────────────┘    └─────────────┘    └─────────────┘          │    │
│  │                                                                   │    │
│  │  RBF: Radial Basis Function for non-linear feature expansion     │    │
│  │  Gaussian: Uncertainty-aware processing                          │    │
│  └─────────────────────────────────────────────────────────────────┘    │
│                              │                                           │
│                              ▼                                           │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │                   TEMPORAL MODELING                              │    │
│  │                                                                   │    │
│  │  ┌─────────────────────────────────────────────────────────┐    │    │
│  │  │              TCN (Temporal Convolutional Network)        │    │    │
│  │  │                                                          │    │    │
│  │  │  ┌─────┐   ┌─────┐   ┌─────┐   ┌─────┐                  │    │    │
│  │  │  │TCN  │──▶│TCN  │──▶│TCN  │──▶│TCN  │  Dilations:      │    │    │
│  │  │  │d=1  │   │d=2  │   │d=4  │   │d=8  │  1,2,4,8         │    │    │
│  │  │  └─────┘   └─────┘   └─────┘   └─────┘                  │    │    │
│  │  │                                                          │    │    │
│  │  │  Captures patterns at multiple time scales               │    │    │
│  │  └─────────────────────────────────────────────────────────┘    │    │
│  │                              +                                   │    │
│  │  ┌─────────────────────────────────────────────────────────┐    │    │
│  │  │              LSTM (Long Short-Term Memory)               │    │    │
│  │  │                                                          │    │    │
│  │  │  Bidirectional, 2 layers, hidden_size=128               │    │    │
│  │  │  Captures long-range dependencies                        │    │    │
│  │  └─────────────────────────────────────────────────────────┘    │    │
│  └─────────────────────────────────────────────────────────────────┘    │
│                              │                                           │
│                              ▼                                           │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │                    OUTPUT HEADS                                  │    │
│  │                                                                   │    │
│  │  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐          │    │
│  │  │ Direction   │    │ Volatility  │    │ Confidence  │          │    │
│  │  │ Head        │    │ Head        │    │ Head        │          │    │
│  │  │ (tanh)      │    │ (softplus)  │    │ (sigmoid)   │          │    │
│  │  │ [-1, +1]    │    │ [0, ∞)      │    │ [0, 1]      │          │    │
│  │  └─────────────┘    └─────────────┘    └─────────────┘          │    │
│  │                                                                   │    │
│  │  Direction: Expected price movement                              │    │
│  │  Volatility: Expected volatility (for position sizing)          │    │
│  │  Confidence: Raw model confidence (before calibration)          │    │
│  └─────────────────────────────────────────────────────────────────┘    │
│                                                                          │
│  Output: { direction: float, volatility: float, confidence: float }     │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### Bayesian Components

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     BAYESIAN UNCERTAINTY                                 │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  BayesianLinear Layer:                                                   │
│  • Weights sampled from learned distributions                            │
│  • Provides epistemic uncertainty estimates                              │
│  • Uses reparameterization trick for gradients                          │
│                                                                          │
│  Monte Carlo Sampling (inference):                                       │
│  • N forward passes with dropout enabled                                 │
│  • Mean = prediction, Std = uncertainty                                  │
│  • Higher uncertainty → lower effective confidence                      │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 4. Signal Generation Pipeline

### Multi-Timeframe Predictions

```
┌─────────────────────────────────────────────────────────────────────────┐
│                   MULTI-TIMEFRAME SCALING                                │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  For each timeframe (1min, 5min, 15min):                                │
│                                                                          │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │ 1-MINUTE PREDICTION                                              │    │
│  │                                                                   │    │
│  │ • Most responsive to immediate price action                      │    │
│  │ • Higher noise, lower weight in final signal                     │    │
│  │ • Used for: Scalping opportunities, quick reversals              │    │
│  │ • Horizon: Next 1-5 minutes                                      │    │
│  └─────────────────────────────────────────────────────────────────┘    │
│                                                                          │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │ 5-MINUTE PREDICTION                                              │    │
│  │                                                                   │    │
│  │ • Balanced signal-to-noise                                       │    │
│  │ • Primary timeframe for trade entry                              │    │
│  │ • Used for: Core trading decisions                               │    │
│  │ • Horizon: Next 5-15 minutes                                     │    │
│  └─────────────────────────────────────────────────────────────────┘    │
│                                                                          │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │ 15-MINUTE PREDICTION (PRIMARY CALIBRATION HORIZON)               │    │
│  │                                                                   │    │
│  │ • Smoothest signal, trend confirmation                           │    │
│  │ • Used for: Calibration, trend filter                           │    │
│  │ • Horizon: Next 15-30 minutes                                    │    │
│  │ • All calibration metrics use this horizon                       │    │
│  └─────────────────────────────────────────────────────────────────┘    │
│                                                                          │
│  Combination:                                                            │
│  final_direction = 0.2 * pred_1m + 0.4 * pred_5m + 0.4 * pred_15m      │
│                                                                          │
│  Agreement Check:                                                        │
│  • All timeframes must agree on direction for high-confidence trades    │
│  • Disagreement → HOLD or reduce position size                          │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### Signal Combination

```
┌─────────────────────────────────────────────────────────────────────────┐
│                      SIGNAL COMBINATION                                  │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  Inputs:                                                                 │
│  ├── Neural network predictions (multi-timeframe)                        │
│  ├── HMM regime state                                                    │
│  ├── Current position status                                             │
│  └── Recent prediction consistency                                       │
│                                                                          │
│  Process:                                                                │
│  1. Aggregate direction across timeframes                                │
│  2. Apply regime-based scaling (reduce in high vol)                      │
│  3. Calculate raw confidence from NN + consistency                       │
│  4. Apply calibration (Platt/Isotonic)                                   │
│  5. Determine action based on thresholds                                 │
│                                                                          │
│  Output:                                                                 │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │ signal = {                                                       │    │
│  │   'action': 'BUY_CALL' | 'BUY_PUT' | 'HOLD',                    │    │
│  │   'direction': -1 to +1,                                         │    │
│  │   'raw_confidence': 0.72,                                        │    │
│  │   'confidence': 0.58,  # calibrated                             │    │
│  │   'volatility_forecast': 0.015,                                  │    │
│  │   'regime': 'normal_vol',                                        │    │
│  │   'multi_timeframe_predictions': {...}                           │    │
│  │ }                                                                │    │
│  └─────────────────────────────────────────────────────────────────┘    │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 5. Calibration System

### Purpose

Raw neural network confidence scores often don't reflect true probabilities. A model saying "72% confident" might only be correct 55% of the time. Calibration fixes this.

### Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     CALIBRATION TRACKER                                  │
│                  (backend/calibration_tracker.py)                        │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  Rolling Buffer (size=1000):                                             │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │ [(conf_1, correct_1), (conf_2, correct_2), ..., (conf_n, corr_n)] │    │
│  │                                                                   │    │
│  │ conf: Raw confidence from neural network                         │    │
│  │ correct: 1 if direction prediction was correct, 0 otherwise      │    │
│  │ Horizon: 15 minutes (PRIMARY_HORIZON_MINUTES)                    │    │
│  └─────────────────────────────────────────────────────────────────┘    │
│                                                                          │
│  Calibration Methods:                                                    │
│                                                                          │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │ PLATT SCALING (40% weight)                                       │    │
│  │                                                                   │    │
│  │ P(correct) = 1 / (1 + exp(A*conf + B))                           │    │
│  │                                                                   │    │
│  │ • Fits logistic regression on (conf, correct) pairs              │    │
│  │ • Fast, works well with limited data                             │    │
│  │ • Refits every 50 new samples                                    │    │
│  └─────────────────────────────────────────────────────────────────┘    │
│                                                                          │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │ ISOTONIC REGRESSION (60% weight)                                 │    │
│  │                                                                   │    │
│  │ Non-parametric monotonic mapping                                  │    │
│  │                                                                   │    │
│  │ • More flexible than Platt                                       │    │
│  │ • Better with sufficient data (>200 samples)                     │    │
│  │ • Preserves monotonicity of confidence                           │    │
│  └─────────────────────────────────────────────────────────────────┘    │
│                                                                          │
│  Hybrid Combination:                                                     │
│  calibrated_conf = 0.4 * platt_conf + 0.6 * isotonic_conf              │
│                                                                          │
│  Metrics:                                                                │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │ Brier Score = mean((calibrated_conf - actual)²)                  │    │
│  │ • Perfect = 0, Random = 0.25, Threshold: < 0.35                  │    │
│  │                                                                   │    │
│  │ ECE (Expected Calibration Error)                                 │    │
│  │ • Measures gap between confidence and actual accuracy            │    │
│  │ • Perfect = 0, Threshold: < 0.15                                 │    │
│  │                                                                   │    │
│  │ Direction Accuracy = correct_predictions / total_predictions     │    │
│  │ • Target: > 52% (need edge over random)                          │    │
│  └─────────────────────────────────────────────────────────────────┘    │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### Calibration Flow

```
Raw Confidence: 0.72 (NN says 72% sure)
        │
        ▼
┌─────────────────────────────────────────┐
│ Historical data shows:                   │
│ When NN says 70-75%, actual win = 55%   │
└─────────────────────────────────────────┘
        │
        ▼
Calibrated Confidence: 0.58 (actual 58% likely)
        │
        ▼
Trade decision uses 0.58, not 0.72
```

---

## 6. Reinforcement Learning

### RL Threshold Learner

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    RL THRESHOLD LEARNER                                  │
│                (backend/rl_threshold_learner.py)                         │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  Purpose: Learn optimal confidence threshold for trade entry             │
│                                                                          │
│  State Space:                                                            │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │ • Current VIX level (normalized)                                 │    │
│  │ • Recent win rate                                                │    │
│  │ • Current drawdown                                               │    │
│  │ • Time of day (encoded)                                          │    │
│  │ • Position count                                                 │    │
│  │ • Recent PnL                                                     │    │
│  │ • Calibration metrics (Brier, ECE)                              │    │
│  └─────────────────────────────────────────────────────────────────┘    │
│                                                                          │
│  Action Space:                                                           │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │ Continuous: threshold adjustment [-0.1, +0.1]                    │    │
│  │ Base threshold: 0.55                                             │    │
│  │ Effective range: [0.35, 0.75]                                    │    │
│  └─────────────────────────────────────────────────────────────────┘    │
│                                                                          │
│  Reward Function:                                                        │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │ reward = pnl_component + sharpe_component - risk_penalty         │    │
│  │                                                                   │    │
│  │ pnl_component: Realized PnL from trades                          │    │
│  │ sharpe_component: Rolling Sharpe ratio contribution              │    │
│  │ risk_penalty: Penalty for excessive drawdown                     │    │
│  └─────────────────────────────────────────────────────────────────┘    │
│                                                                          │
│  Training:                                                               │
│  • PPO (Proximal Policy Optimization) algorithm                         │
│  • Online learning from trade outcomes                                   │
│  • Prioritized experience replay for important transitions              │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### RL Exit Policy

```
┌─────────────────────────────────────────────────────────────────────────┐
│                      RL EXIT POLICY                                      │
│                  (backend/rl_exit_policy.py)                             │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  Purpose: Learn optimal exit timing for open positions                   │
│                                                                          │
│  State Space (per position):                                             │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │ • Current PnL %                                                  │    │
│  │ • Hold duration (minutes)                                        │    │
│  │ • Max PnL achieved                                               │    │
│  │ • Drawdown from max                                              │    │
│  │ • Current volatility                                             │    │
│  │ • Original prediction confidence                                 │    │
│  │ • Updated prediction (current NN output)                         │    │
│  │ • Time to market close                                           │    │
│  │ • Greeks (delta, theta, vega)                                    │    │
│  └─────────────────────────────────────────────────────────────────┘    │
│                                                                          │
│  Action Space:                                                           │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │ Discrete: { HOLD, EXIT }                                         │    │
│  │ Or Continuous: exit_probability [0, 1]                           │    │
│  └─────────────────────────────────────────────────────────────────┘    │
│                                                                          │
│  Reward:                                                                 │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │ • Realized PnL on exit                                           │    │
│  │ • Bonus for capturing large moves                                │    │
│  │ • Penalty for holding too long (theta decay)                     │    │
│  │ • Penalty for giving back profits                                │    │
│  └─────────────────────────────────────────────────────────────────┘    │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 7. Regime Detection

### Multi-Dimensional HMM

```
┌─────────────────────────────────────────────────────────────────────────┐
│                   MULTI-DIMENSIONAL HMM                                  │
│                (backend/multi_dimensional_hmm.py)                        │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  Architecture: 3 independent HMMs combined                               │
│                                                                          │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐         │
│  │ TREND HMM       │  │ VOLATILITY HMM  │  │ LIQUIDITY HMM   │         │
│  │ (3 states)      │  │ (3 states)      │  │ (3 states)      │         │
│  │                 │  │                 │  │                 │         │
│  │ 0: Bullish      │  │ 0: Low Vol      │  │ 0: High Liq     │         │
│  │ 1: Neutral      │  │ 1: Normal Vol   │  │ 1: Normal Liq   │         │
│  │ 2: Bearish      │  │ 2: High Vol     │  │ 2: Low Liq      │         │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘         │
│                                                                          │
│  Combined States: 3 × 3 × 3 = 27 possible regimes                       │
│                                                                          │
│  Observations (per HMM):                                                 │
│  • Trend: Returns, momentum, moving average crossovers                   │
│  • Volatility: ATR, VIX level, Bollinger width                          │
│  • Liquidity: Volume, bid-ask spread, market depth                      │
│                                                                          │
│  Training:                                                               │
│  • Baum-Welch EM algorithm (unsupervised)                               │
│  • Auto-discovery of optimal state count via BIC                        │
│  • Retraining triggered by drift detection                              │
│                                                                          │
│  Output:                                                                 │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │ {                                                                │    │
│  │   'trend_state': 0,  # Bullish                                   │    │
│  │   'vol_state': 1,    # Normal                                    │    │
│  │   'liq_state': 0,    # High                                      │    │
│  │   'state_probs': [0.7, 0.2, 0.1],  # Trend state probabilities  │    │
│  │   'regime_id': 'bullish_normal_vol'                              │    │
│  │ }                                                                │    │
│  └─────────────────────────────────────────────────────────────────┘    │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### Regime-Specific Strategies

```
┌─────────────────────────────────────────────────────────────────────────┐
│                   REGIME STRATEGY MANAGER                                │
│                 (backend/regime_strategies.py)                           │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  VIX-Based Regimes:                                                      │
│                                                                          │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │ ULTRA-LOW VOL (VIX < 12)                                         │    │
│  │ • Min confidence: 45% (lower to get more trades)                 │    │
│  │ • Position scale: 1.2x                                           │    │
│  │ • Stop loss: 15% (tighter)                                       │    │
│  │ • Max trades/hour: 8                                             │    │
│  │ • Multi-TF agreement: NOT required                               │    │
│  └─────────────────────────────────────────────────────────────────┘    │
│                                                                          │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │ NORMAL VOL (VIX 15-20)                                           │    │
│  │ • Min confidence: 55%                                            │    │
│  │ • Position scale: 1.0x                                           │    │
│  │ • Stop loss: 20%                                                 │    │
│  │ • Max trades/hour: 5                                             │    │
│  │ • Multi-TF agreement: Required                                   │    │
│  └─────────────────────────────────────────────────────────────────┘    │
│                                                                          │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │ HIGH VOL (VIX 25-35)                                             │    │
│  │ • Min confidence: 65% (higher bar)                               │    │
│  │ • Position scale: 0.5x                                           │    │
│  │ • Stop loss: 30% (wider)                                         │    │
│  │ • Max trades/hour: 3                                             │    │
│  │ • Counter-trend: NOT allowed                                     │    │
│  └─────────────────────────────────────────────────────────────────┘    │
│                                                                          │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │ EXTREME VOL (VIX > 35)                                           │    │
│  │ • Min confidence: 75%                                            │    │
│  │ • Position scale: 0.3x                                           │    │
│  │ • Max trades/hour: 2                                             │    │
│  │ • Mode: Capital preservation                                     │    │
│  └─────────────────────────────────────────────────────────────────┘    │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 8. Trade Execution

### Contract Selection

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    CONTRACT SELECTION                                    │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  Selection Criteria:                                                     │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │ 1. Expiration: Same-day or next-day (0-1 DTE)                   │    │
│  │    • Maximum theta capture                                       │    │
│  │    • Most liquid options                                         │    │
│  │                                                                   │    │
│  │ 2. Strike Selection:                                             │    │
│  │    • ATM or slightly OTM (within 1-2 strikes)                   │    │
│  │    • Target delta: 0.40-0.50 for calls, -0.40 to -0.50 for puts │    │
│  │    • Maximum gamma exposure                                      │    │
│  │                                                                   │    │
│  │ 3. Liquidity Requirements:                                       │    │
│  │    • Open Interest ≥ 100                                         │    │
│  │    • Volume ≥ 150                                                │    │
│  │    • Bid-Ask Spread ≤ 4%                                         │    │
│  │                                                                   │    │
│  │ 4. Strike Deviation:                                             │    │
│  │    • Max 15% from current price                                  │    │
│  └─────────────────────────────────────────────────────────────────┘    │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### Order Execution

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     ORDER EXECUTION                                      │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  Execution Strategy:                                                     │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │ 1. Midpoint Entry:                                               │    │
│  │    • Start with limit at midpoint                                │    │
│  │    • Wait 5-10 seconds for fill                                  │    │
│  │                                                                   │    │
│  │ 2. Price Ladder (if unfilled):                                   │    │
│  │    • Improve price by 1 tick every 5 seconds                     │    │
│  │    • Max 3-5 improvements before canceling                       │    │
│  │                                                                   │    │
│  │ 3. Fallback to Spread (if single leg fails):                    │    │
│  │    • Convert to vertical spread                                  │    │
│  │    • Reduced cost, limited profit potential                      │    │
│  └─────────────────────────────────────────────────────────────────┘    │
│                                                                          │
│  Position Sizing:                                                        │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │ Base Formula:                                                    │    │
│  │ position_size = account_balance * risk_per_trade * confidence_adj│    │
│  │                                                                   │    │
│  │ Constraints:                                                      │    │
│  │ • Max 2% of account per trade                                    │    │
│  │ • Max 2-5 concurrent positions (regime-dependent)               │    │
│  │ • Scale by calibrated confidence                                 │    │
│  │ • Scale by regime (reduce in high vol)                          │    │
│  └─────────────────────────────────────────────────────────────────┘    │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 9. Risk Management

### Position Limits

```
┌─────────────────────────────────────────────────────────────────────────┐
│                      RISK LIMITS                                         │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  Per-Trade Limits:                                                       │
│  • Max loss: 2% of account                                               │
│  • Max position size: 5% of account value                               │
│  • Stop loss: Always set (15-40% depending on regime)                   │
│                                                                          │
│  Portfolio Limits:                                                       │
│  • Max concurrent positions: 2-6 (regime-dependent)                      │
│  • Max daily trades: 20                                                  │
│  • Max hourly trades: 3-8 (regime-dependent)                            │
│  • Max same-direction positions: 2                                       │
│                                                                          │
│  Drawdown Controls:                                                      │
│  • Daily loss limit: 5% → reduce position sizes                         │
│  • Daily loss limit: 10% → pause trading                                │
│  • Weekly loss limit: 15% → full stop                                   │
│                                                                          │
│  PDT Protection:                                                         │
│  • Track day trades (open + close same day)                             │
│  • If account < $25k: max 3 day trades per 5 days                       │
│  • Warning at 2 day trades                                               │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### Exit Management

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    EXIT MANAGEMENT                                       │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  Exit Triggers (checked every cycle):                                    │
│                                                                          │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │ RULE-BASED EXITS                                                 │    │
│  │                                                                   │    │
│  │ • Stop Loss Hit: PnL < -stop_loss_pct                           │    │
│  │ • Take Profit Hit: PnL > take_profit_pct                        │    │
│  │ • Trailing Stop: PnL dropped trailing_stop_pct from peak        │    │
│  │ • Max Hold Time: position_age > max_hold_minutes                │    │
│  │ • End of Day: 15 minutes before market close                    │    │
│  └─────────────────────────────────────────────────────────────────┘    │
│                                                                          │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │ RL-BASED EXITS                                                   │    │
│  │                                                                   │    │
│  │ • RL policy recommends exit based on learned patterns           │    │
│  │ • Can override rule-based if learned behavior is better         │    │
│  │ • Confidence threshold for RL exit: 60%                         │    │
│  └─────────────────────────────────────────────────────────────────┘    │
│                                                                          │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │ PREDICTION-BASED EXITS                                           │    │
│  │                                                                   │    │
│  │ • Current NN prediction reverses vs entry prediction            │    │
│  │ • Confidence in reversal > 55%                                  │    │
│  │ • Multi-timeframe agreement on reversal                         │    │
│  └─────────────────────────────────────────────────────────────────┘    │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 10. Monitoring & Health

### Health Check System

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    HEALTH CHECK SYSTEM                                   │
│                  (backend/health_checks.py)                              │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  Pre-Cycle Checks (run every cycle):                                     │
│                                                                          │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │ SYSTEM RESOURCES                                                 │    │
│  │ • CPU usage < 90%                                                │    │
│  │ • Memory usage < 85%                                             │    │
│  │ • Disk space available                                           │    │
│  └─────────────────────────────────────────────────────────────────┘    │
│                                                                          │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │ DATA FRESHNESS                                                   │    │
│  │ • Last price update < 5 minutes old                             │    │
│  │ • All required symbols have data                                 │    │
│  │ • No stale data being used                                       │    │
│  └─────────────────────────────────────────────────────────────────┘    │
│                                                                          │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │ MODEL HEALTH                                                     │    │
│  │ • Model loaded successfully                                      │    │
│  │ • Recent prediction errors < threshold                           │    │
│  │ • Feature drift within bounds                                    │    │
│  │ • No NaN/Inf in outputs                                          │    │
│  └─────────────────────────────────────────────────────────────────┘    │
│                                                                          │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │ CALIBRATION HEALTH                                               │    │
│  │ • Sufficient samples (≥50)                                       │    │
│  │ • Brier score < 0.35                                             │    │
│  │ • ECE < 0.15                                                     │    │
│  │ • Recent calibration refit                                       │    │
│  └─────────────────────────────────────────────────────────────────┘    │
│                                                                          │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │ POSITION LIMITS                                                  │    │
│  │ • Under max positions                                            │    │
│  │ • Under daily trade limit                                        │    │
│  │ • Under drawdown limits                                          │    │
│  └─────────────────────────────────────────────────────────────────┘    │
│                                                                          │
│  Overall Health Score: 0-100%                                            │
│  • < 50%: Skip trading, alert                                           │
│  • 50-80%: Reduced position sizes                                       │
│  • > 80%: Normal operation                                              │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### Trading Monitor Dashboard

```
┌─────────────────────────────────────────────────────────────────────────┐
│                   MONITORING DASHBOARD                                   │
│               (backend/monitoring_dashboard.py)                          │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  Web Interface: http://localhost:5001                                    │
│                                                                          │
│  Tracked Metrics:                                                        │
│                                                                          │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │ POSITION MONITORING                                              │    │
│  │                                                                   │    │
│  │ Per Position:                                                    │    │
│  │ • Symbol, direction, entry price/time                           │    │
│  │ • Current PnL %                                                  │    │
│  │ • Hold duration                                                  │    │
│  │ • Status: OPEN, STUCK, AT_RISK, PROFITABLE                      │    │
│  │                                                                   │    │
│  │ Alerts:                                                          │    │
│  │ • Position stuck > 2 hours                                       │    │
│  │ • Position at risk (> -15% PnL)                                 │    │
│  │ • Long hold time warning                                         │    │
│  └─────────────────────────────────────────────────────────────────┘    │
│                                                                          │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │ REJECTION ANALYSIS                                               │    │
│  │                                                                   │    │
│  │ By Reason:                                                       │    │
│  │ • Low confidence: 45%                                            │    │
│  │ • Multi-TF disagreement: 25%                                     │    │
│  │ • Trade limit: 15%                                               │    │
│  │ • Poor calibration: 10%                                          │    │
│  │ • Other: 5%                                                      │    │
│  │                                                                   │    │
│  │ By Hour:                                                         │    │
│  │ • 9:30-10:00: High rejection (opening volatility)               │    │
│  │ • 12:00-14:00: Low rejection (stable lunch)                     │    │
│  │ • 15:30-16:00: High rejection (closing)                         │    │
│  └─────────────────────────────────────────────────────────────────┘    │
│                                                                          │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │ REGIME PERFORMANCE                                               │    │
│  │                                                                   │    │
│  │ Per Regime:                                                      │    │
│  │ • Trades, wins, win rate                                         │    │
│  │ • Total PnL, average PnL                                         │    │
│  │ • Sharpe ratio                                                   │    │
│  │                                                                   │    │
│  │ Example:                                                         │    │
│  │ low_vol:    50 trades, 58% win, +$450                           │    │
│  │ normal_vol: 120 trades, 52% win, +$200                          │    │
│  │ high_vol:   30 trades, 48% win, -$100                           │    │
│  └─────────────────────────────────────────────────────────────────┘    │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 11. File Structure

```
gaussian/output3/
├── unified_options_trading_bot.py    # Main bot class
├── train_then_go_live.py             # Training + live trading script
├── run_simulation.py                 # Historical simulation
├── config.json                       # Configuration
├── ARCHITECTURE.md                   # This document
│
├── backend/
│   ├── live_trading_engine.py        # Live trading orchestration
│   ├── paper_trading_system.py       # Paper trading simulation
│   ├── tradier_trading_system.py     # Live Tradier integration
│   │
│   ├── calibration_tracker.py        # Platt/Isotonic calibration
│   ├── multi_dimensional_hmm.py      # Regime detection
│   ├── rl_threshold_learner.py       # RL for thresholds
│   ├── rl_exit_policy.py             # RL for exits
│   │
│   ├── regime_strategies.py          # Regime-specific params
│   ├── regime_models.py              # Per-regime ML models
│   │
│   ├── robust_fetcher.py             # Data fetching with retries
│   ├── feature_cache.py              # LRU caching
│   ├── async_operations.py           # Parallel execution
│   │
│   ├── health_checks.py              # Pre-cycle validation
│   ├── model_health.py               # Drift detection
│   ├── monitoring_dashboard.py       # Web dashboard
│   │
│   ├── integration.py                # Component wiring
│   ├── tradier_data_source.py        # Tradier API client
│   ├── enhanced_data_sources.py      # Multi-source data
│   └── liquidity_validator.py        # Options liquidity checks
│
├── bot_modules/
│   ├── neural_networks.py            # NN architectures
│   ├── features.py                   # Feature computation
│   └── signals.py                    # Signal generation
│
├── models/
│   ├── unified_predictor.pt          # Main NN weights
│   ├── rl_threshold_learner.pth      # RL threshold model
│   ├── rl_exit_policy_live.pth       # RL exit model
│   ├── multi_dimensional_hmm.pkl     # HMM model
│   └── regime_models/                # Per-regime models
│
├── data/
│   ├── market_data.db                # Historical OHLCV
│   └── paper_trades.db               # Trade history
│
├── logs/
│   └── *.log                         # Trading logs
│
└── tests/
    └── test_calibration.py           # Unit tests
```

---

## Quick Reference

### Key Configuration (`config.json`)

```json
{
  "trading": {
    "symbol": "SPY",
    "min_confidence_threshold": 0.55,
    "max_positions": 5
  },
  "calibration": {
    "min_confidence_threshold": 0.55,
    "max_brier_score": 0.35,
    "max_ece": 0.15,
    "require_horizon_agreement": true
  },
  "regime": {
    "vix_thresholds": [12, 15, 20, 25, 35],
    "use_hmm": true
  }
}
```

### Key Metrics to Monitor

| Metric | Good | Warning | Bad |
|--------|------|---------|-----|
| Win Rate | > 55% | 50-55% | < 50% |
| Brier Score | < 0.25 | 0.25-0.35 | > 0.35 |
| ECE | < 0.10 | 0.10-0.15 | > 0.15 |
| Sharpe Ratio | > 1.5 | 0.5-1.5 | < 0.5 |
| Direction Accuracy | > 55% | 50-55% | < 50% |

### Common Commands

```bash
# Start training + live trading
python train_then_go_live.py

# Run historical simulation only
python run_simulation.py

# Start monitoring dashboard
python -m backend.monitoring_dashboard

# Run tests
python -m pytest tests/
```

---

*Last Updated: December 2024*

