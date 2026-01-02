# Gaussian Options Trading Bot - Current Architecture

*Last Updated: 2026-01-02*

## Overview

The Gaussian Options Trading Bot is an algorithmic SPY options trading system combining:
- Bayesian neural networks with Gaussian kernel processors
- Multi-dimensional HMM (3×3×3 = 27 states) for market regime detection
- Multiple swappable temporal encoders (TCN, Transformer, LSTM)
- Multiple entry controllers (Bandit, Consensus, Q-Scorer, RL)
- Paper trading simulation and live Tradier execution

---

## System Data Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                    MARKET DATA SOURCES                          │
│  Tradier API → Polygon.io → Data Manager → Yahoo Finance        │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    FEATURE PIPELINE                              │
│  50-200 features: Equity, Options, Breadth, Macro, Crypto, Meta │
└─────────────────────────────────────────────────────────────────┘
                              │
        ┌─────────────────────┼─────────────────────┐
        ▼                     ▼                     ▼
┌───────────────┐    ┌───────────────┐    ┌───────────────┐
│ FEATURE BUFFER│    │ HMM REGIME    │    │ NEURAL        │
│               │    │               │    │ PREDICTOR     │
│ Rolling 60    │    │ Trend (0-1)   │    │               │
│ timesteps     │───→│ Volatility    │───→│ Return, Vol   │
│ (1 hour)      │    │ Liquidity     │    │ Direction     │
│               │    │ Confidence    │    │ Confidence    │
└───────────────┘    └───────────────┘    └───────────────┘
                              │                     │
                              └──────────┬──────────┘
                                         ▼
                    ┌─────────────────────────────────────┐
                    │        ENTRY CONTROLLER             │
                    │  Bandit | Consensus | Q-Scorer | RL │
                    │                                     │
                    │  Action: HOLD / BUY_CALL / BUY_PUT  │
                    └─────────────────────────────────────┘
                                         │
                                         ▼
                    ┌─────────────────────────────────────┐
                    │          EXIT MANAGER               │
                    │                                     │
                    │  Safety Rules: -8% SL, +12% TP      │
                    │  Model-Based: XGBoost / NN Exit     │
                    └─────────────────────────────────────┘
                                         │
                                         ▼
                    ┌─────────────────────────────────────┐
                    │      PAPER / LIVE TRADING           │
                    │                                     │
                    │  Paper: Simulated fills at mid      │
                    │  Live: Tradier API execution        │
                    └─────────────────────────────────────┘
```

---

## Entry Points

### Training: `scripts/train_time_travel.py`
- Time-travel training - feeds historical data as live
- Primary training script (~7400 lines)

```bash
# Basic training
python scripts/train_time_travel.py

# With configuration
MODEL_RUN_DIR=models/my_test \
TT_MAX_CYCLES=10000 \
TEMPORAL_ENCODER=transformer \
PAPER_TRADING=True \
python scripts/train_time_travel.py
```

### Live Trading: `go_live_only.py`
- Execute trained models in live/paper mode

```bash
# Paper trading (default)
python go_live_only.py models/run_YYYYMMDD_HHMMSS

# Live trading (set PAPER_ONLY_MODE=False first!)
python go_live_only.py models/run_YYYYMMDD_HHMMSS
```

---

## Neural Network Architecture

### Location
`bot_modules/neural_networks.py` (1483 lines)

### Temporal Encoders (Swappable)

| Encoder | Receptive Field | Speed | Use Case |
|---------|-----------------|-------|----------|
| **TCN** (default) | ~90 timesteps | 3-5x faster | Production recommended |
| **Transformer** | Full attention | Moderate | Best OOS generalization |
| **LSTM** | ~30 timesteps | Slowest | Legacy fallback |

Configure via: `TEMPORAL_ENCODER=transformer`

### Main Predictor Architecture

```
Input: Current Features [B, D] + Sequence [B, 60, D]
                    │
                    ▼
         ┌──────────────────┐
         │  RBF Kernel Layer │  25 centers × 5 scales = 125 features
         └──────────────────┘
                    │
                    ▼
         ┌──────────────────┐
         │ Temporal Encoder │  TCN / Transformer / LSTM → 64-dim
         └──────────────────┘
                    │
                    ▼
         ┌──────────────────┐
         │ Bayesian ResBlocks│  256 → 256 → 128 → 64
         └──────────────────┘
                    │
                    ▼
         ┌──────────────────┐
         │  Output Heads     │
         │                   │
         │  • return (1)     │  Predicted % return
         │  • volatility (1) │  Predicted volatility
         │  • direction (3)  │  [DOWN, NEUTRAL, UP]
         │  • confidence (1) │  0-1 prediction confidence
         │  • fillability (1)│  P(fill at mid-peg)
         │  • embedding (64) │  For downstream models
         └──────────────────┘
```

### Predictor Versions

| Version | Key Feature | Best Use |
|---------|-------------|----------|
| **V1** | Baseline | Legacy |
| **V2 Slim Bayesian** | Faster training | Default |
| **V3 Multi-Horizon** | 5m/15m/30m/45m predictions | Best P&L (+1327%) |

Configure via: `PREDICTOR_ARCH=v3_multi_horizon`

---

## Feature Pipeline

### Location
`features/` directory (modular architecture)

### Feature Categories

| Category | Prefix | Features | Source |
|----------|--------|----------|--------|
| Equity/ETF | `eq_*` | Returns, volatility, MAs | SPY, QQQ, IWM, TLT, GLD |
| Options Surface | `opt_*` | Put-call ratios, IV, skew | Options chain |
| Market Breadth | `brd_*` | Breadth indicators | Major indices |
| Macro | `mac_*` | Rates, credit, commodities | VIX, yields, sectors |
| Crypto | `cry_*` | BTC/ETH returns, correlation | Crypto markets |
| Meta | `meta_*` | Time-of-day, day-of-week | System |
| Jerry (optional) | `jerry_*` | Fuzzy logic scoring | Quantor-MTFuzz |

**Total Features**: 50-200+ depending on configuration

### Feature Configuration
```python
enable_equity_etf: bool = True
enable_options_surface: bool = True
enable_breadth: bool = True
enable_macro: bool = True
enable_extended_macro: bool = True
enable_crypto: bool = True
enable_meta: bool = True
enable_jerry: bool = False  # Set JERRY_FEATURES=1 to enable
```

---

## Entry Controllers

### Location
`backend/` directory

### Available Controllers

| Controller | Config Value | Description | When to Use |
|------------|--------------|-------------|-------------|
| **Bandit** | `bandit` | HMM-only with strict thresholds | Default, proven profitable |
| **Consensus** | `consensus` | Multi-signal agreement (6 signals) | High-confidence trades |
| **Q-Scorer** | `q_scorer` | Offline Q-regression | Experimental |
| **RL Policy** | `rl` | PPO with credit assignment | After collecting data |
| **V3 Direction** | `v3_direction` | Standalone direction predictor | ~56% accuracy |

### Bandit Controller (Default)
```python
# HMM thresholds
HMM_BULLISH > 0.6 → Allow CALL
HMM_BEARISH < 0.4 → Allow PUT
Conflict → Block trade

# Confidence minimum
MIN_CONFIDENCE = 0.50-0.55

# VIX range
VIX_MIN = 12, VIX_MAX = 35
```

### Consensus Controller (6 Signals)
1. Timeframe Agreement (15m, 30m, 1h same direction)
2. HMM Alignment (hard veto if conflict)
3. Momentum Confirmation (momentum + jerk + RSI)
4. Volatility Filter (VIX range, volume spike)
5. Technical Confirmation (MACD, Bollinger, breadth)
6. Straddle Detection (high vol + no clear direction)

---

## HMM Regime Detection

### Location
`backend/multi_dimensional_hmm.py` (1195 lines)

### Architecture

```
Market Data (returns, volatility, volume)
           │
     ┌─────┼─────┐
     ▼     ▼     ▼
  ┌─────┐ ┌─────┐ ┌─────┐
  │Trend│ │ Vol │ │ Liq │
  │ HMM │ │ HMM │ │ HMM │
  └─────┘ └─────┘ └─────┘
     │     │     │
     ▼     ▼     ▼
  Bullish  Low   High
  Neutral  Normal Normal
  Bearish  High   Low
     │     │     │
     └─────┼─────┘
           ▼
  Normalized Output [0-1]:
  • hmm_trend: 0=bearish, 1=bullish
  • hmm_volatility: 0=low, 1=high
  • hmm_liquidity: 0=low, 1=high
  • hmm_confidence: 0=uncertain, 1=confident
```

### Trade Gating Logic

| HMM Trend | Neural Prediction | Action |
|-----------|-------------------|--------|
| Bullish (>0.6) | CALL | ✅ ALIGNED - 1.15x boost |
| Bullish (>0.6) | PUT | ❌ CONFLICT - Block |
| Bearish (<0.4) | PUT | ✅ ALIGNED - 1.15x boost |
| Bearish (<0.4) | CALL | ❌ CONFLICT - Block |
| Neutral + High Vol | Any | ❌ CHOPPY - Block |

---

## Exit Management

### Location
`backend/unified_exit_manager.py`

### Exit Priority Order

**1. HARD SAFETY RULES (Always Checked First)**
| Rule | Default | Description |
|------|---------|-------------|
| Stop Loss | -8% | Hard exit on loss |
| Take Profit | +12% | Hard exit on gain |
| Max Hold | 45 min | Prevent theta decay |
| Near Expiry | <30 min | Exit before expiration |
| Trailing Stop | +4% activation, 2% trail | Lock in profits |

**2. MODEL-BASED EXIT (If Safety Doesn't Trigger)**
- XGBoost Exit: Tree-based exit probability
- NN Exit: Neural network exit probability
- Threshold: 0.55-0.60

### Configuration
```json
{
  "architecture": {
    "exit_policy": {
      "type": "xgboost_exit",
      "hard_stop_loss_pct": 8.0,
      "hard_take_profit_pct": 15.0,
      "hard_max_hold_minutes": 45,
      "trailing_stop": {
        "activation_pct": 8.0,
        "distance_pct": 4.0
      }
    }
  }
}
```

---

## Paper Trading System

### Location
`backend/paper_trading_system.py` (3000+ lines)

### Trade Lifecycle
```
Entry Signal → Order Creation → Execution → Tracking → Exit Decision → Closure
                                                              │
                                              Calculate P&L, Greeks, Slippage
```

### Features
- Fills at mid (bid+ask)/2
- Simulates slippage based on volatility
- Tracks Greeks (delta, gamma, theta, vega)
- SQLite database for trade records
- Position limits (default: 3 concurrent)

---

## Live Execution

### Location
`execution/` directory

### Components

| Component | File | Purpose |
|-----------|------|---------|
| TradierAdapter | `tradier_adapter.py` | REST API wrapper |
| TradierTradingSystem | `tradier_trading_system.py` | Order management |
| LiquidityExecutor | `liquidity_exec.py` | Smart execution |

### Liquidity Screening
- Minimum open interest
- Minimum daily volume
- Maximum bid-ask spread
- Delta targeting

### Order Tactics
- Midpoint-peg orders
- Price ladder (incremental lifts)
- IOC (Immediate-or-Cancel) probes
- Cancel-replace for staleness

---

## Key Environment Variables

| Variable | Purpose | Default | Example |
|----------|---------|---------|---------|
| `PREDICTOR_ARCH` | Predictor version | auto | `v3_multi_horizon` |
| `TEMPORAL_ENCODER` | Sequence encoder | `tcn` | `transformer` |
| `TT_MAX_CYCLES` | Training iterations | 0 | `10000` |
| `TT_PRINT_EVERY` | Log interval | 20 | `500` |
| `ENTRY_CONTROLLER` | Entry type | bandit | `consensus` |
| `PAPER_TRADING` | Paper mode | False | `True` |
| `LOAD_PRETRAINED` | Load existing model | 0 | `1` |

---

## Key Dimensions

| Component | Input Shape | Output Shape |
|-----------|-------------|--------------|
| Current Features | [B, D] | - (D=50-200) |
| Sequence | [B, 60, D] | - (60 timesteps) |
| RBF Output | [B, D] | [B, 125] |
| Temporal Encoder | [B, 60, D] | [B, 64] |
| RL Policy | 18 features | 4 actions |
| HMM | [N, F] | 4 values [0-1] |

---

## Directory Structure

```
gaussian-system/
├── backend/                    # Core trading infrastructure
│   ├── unified_rl_policy.py   # RL entry/exit policy
│   ├── unified_exit_manager.py # Exit decision system
│   ├── multi_dimensional_hmm.py # HMM regime detection
│   ├── paper_trading_system.py # Paper trading simulation
│   ├── consensus_entry_controller.py
│   ├── q_entry_controller.py
│   └── xgboost_exit_policy.py
├── bot_modules/               # Neural networks & features
│   ├── neural_networks.py     # TCN, Transformer, Predictors
│   └── technical_indicators.py
├── execution/                 # Live trading
│   ├── tradier_adapter.py
│   └── liquidity_exec.py
├── features/                  # Feature pipeline
│   ├── pipeline.py
│   ├── macro.py
│   └── equity.py
├── scripts/                   # Training scripts
│   └── train_time_travel.py
├── training/                  # Q-Scorer training
├── models/                    # Saved model checkpoints
├── data/                      # SQLite databases
├── docs/                      # Documentation
└── config.json               # Configuration (GITIGNORED)
```

---

## Recent Improvements (Phase 35)

### Key Finding: Transformer Generalizes Best

| Model | Training P&L | December OOS P&L | Verdict |
|-------|--------------|------------------|---------|
| Transformer (10K) | +2777% | **+32.65%** | ✅ BEST |
| V3 Multi-Horizon (10K) | +3119% | -3.34% | ❌ |
| 20K Pretrain (TCN) | +3753% | -3.95% | ❌ |

**Recommendation**: Use `TEMPORAL_ENCODER=transformer` for production.

---

## Quick Start Commands

```bash
# Train with transformer encoder (best generalization)
TEMPORAL_ENCODER=transformer \
MODEL_RUN_DIR=models/my_model \
TT_MAX_CYCLES=10000 \
PAPER_TRADING=True \
python scripts/train_time_travel.py

# Test on different time period
LOAD_PRETRAINED=1 \
PRETRAINED_MODEL_PATH=models/my_model/state \
TRAINING_START_DATE=2025-12-01 \
TRAINING_END_DATE=2025-12-15 \
MODEL_RUN_DIR=models/my_model_dec_test \
TT_MAX_CYCLES=5000 \
python scripts/train_time_travel.py

# Run live (paper mode)
python go_live_only.py models/my_model
```

---

*Document generated from codebase analysis on 2026-01-02*
