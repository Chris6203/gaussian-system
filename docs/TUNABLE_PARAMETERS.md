# Tunable Parameters for Bot Optimization

## 1. Entry Strategy Parameters

### 1.1 HMM Thresholds
| Parameter | Current | Range | Description |
|-----------|---------|-------|-------------|
| `HMM_STRONG_BULLISH` | 0.70 | 0.60-0.80 | Trend threshold for bullish entry |
| `HMM_STRONG_BEARISH` | 0.30 | 0.20-0.40 | Trend threshold for bearish entry |
| `HMM_MIN_CONFIDENCE` | 0.70 | 0.50-0.90 | Minimum HMM confidence to trade |
| `HMM_MAX_VOLATILITY` | 0.70 | 0.50-0.90 | Maximum volatility to allow trading |

### 1.2 Neural Network Confidence
| Parameter | Current | Range | Description |
|-----------|---------|-------|-------------|
| `training_min_confidence` | 0.20 | 0.0-0.50 | Minimum NN confidence to trade |
| `training_min_abs_predicted_return` | 0.0008 | 0.0-0.002 | Minimum predicted edge (0.08%) |

### 1.3 Entry Controllers
| Controller | Description | Status |
|------------|-------------|--------|
| `bandit` | Rule-based with HMM thresholds | Default |
| `rl` | Full neural network policy | Available |
| `consensus` | Multi-signal agreement (8 signals) | Available |
| `q_scorer` | Offline Q-regression model | Available |
| `v3_direction` | Direction predictor based | Available |

## 2. Exit Strategy Parameters

### 2.1 Fixed Exits
| Parameter | Current | Range | Description |
|-----------|---------|-------|-------------|
| `stop_loss_pct` | 8.0% | 3-15% | Maximum loss before exit |
| `take_profit_pct` | 12.0% | 8-25% | Target profit for exit |
| `max_hold_minutes` | 45 | 15-120 | Maximum holding time |

### 2.2 Trailing Stop
| Parameter | Current | Range | Description |
|-----------|---------|-------|-------------|
| `trailing_stop_activation_pct` | 4.0% | 2-10% | Profit to activate trailing |
| `trailing_stop_distance_pct` | 2.0% | 1-5% | Distance for trailing stop |

### 2.3 Exit Policies
| Policy | Description | Status |
|--------|-------------|--------|
| `xgboost_exit` | ML-based exit timing | Default |
| `simple_exit` | Fixed rules only | Available |
| `dynamic_exit` | Continuous re-evaluation | Available (hurts performance) |

## 3. Neural Network Architecture

### 3.1 Temporal Encoder
| Type | Description | Status |
|------|-------------|--------|
| `tcn` | Temporal Convolutional Network | Default |
| `transformer` | Attention-based transformer | Available |
| `lstm` | Long Short-Term Memory | Available |

### 3.2 Architecture Components
| Parameter | Current | Options | Description |
|-----------|---------|---------|-------------|
| `NORM_TYPE` | layernorm | layernorm, rmsnorm | Normalization layer |
| `ACTIVATION_TYPE` | gelu | gelu, geglu, swiglu | Activation function |
| `RBF_GATED` | 0 | 0, 1 | Gated RBF kernel |
| `feature_dim` | 50 | 50, 59 | Feature dimension |

### 3.3 Learning Parameters
| Parameter | Current | Range | Description |
|-----------|---------|-------|-------------|
| `learning_rate` | 0.0003 | 0.0001-0.001 | NN learning rate |
| `batch_size` | 128 | 32-256 | Training batch size |
| `direction_loss_weight` | 4.5 | 1.0-10.0 | Direction vs return loss weight |

## 4. Feature Engineering

### 4.1 Feature Sources
| Source | Description | Status |
|--------|-------------|--------|
| `enable_equity_etf` | SPY, QQQ features | Enabled |
| `enable_macro` | TLT, UUP, VIX features | Enabled |
| `enable_extended_macro` | 27 ETF symbols, sectors | Disabled |
| `enable_options_surface` | Options chain features | Disabled |
| `enable_breadth` | Market breadth indicators | Disabled |
| `enable_crypto` | BTC correlation | Disabled |

### 4.2 Technical Indicators
| Indicator | Description | Status |
|-----------|-------------|--------|
| RSI | Relative Strength Index | Included |
| MACD | Moving Average Convergence | Included |
| Bollinger Bands | Volatility bands | Included |
| ATR | Average True Range | Available |
| OBV | On-Balance Volume | Available |

## 5. Risk Management

### 5.1 Position Sizing
| Parameter | Current | Range | Description |
|-----------|---------|-------|-------------|
| `max_risk_per_trade_pct` | 2.0% | 1-5% | Max risk per trade |
| `max_position_size_pct` | 5.0% | 2-10% | Max position size |
| `max_concurrent_positions` | 5 | 1-10 | Max simultaneous positions |

### 5.2 Circuit Breakers
| Parameter | Current | Range | Description |
|-----------|---------|-------|-------------|
| `catastrophic_loss_pct` | -25% | -15 to -50% | Emergency exit threshold |
| `max_consecutive_losses` | 5 | 3-10 | Losses before pause |
| `max_daily_loss_pct` | -20% | -10 to -30% | Daily loss limit |

## 6. Calibration & Gates

### 6.1 PnL Calibration Gate
| Parameter | Current | Range | Description |
|-----------|---------|-------|-------------|
| `PNL_CAL_GATE` | 1 | 0, 1 | Enable P(profit) gate |
| `PNL_CAL_MIN_PROB` | 0.40 | 0.30-0.60 | Minimum P(profit) |

### 6.2 Tradability Gate
| Parameter | Current | Range | Description |
|-----------|---------|-------|-------------|
| `veto_threshold` | 0.25 | 0.15-0.40 | Veto trades below this |
| `downgrade_threshold` | 0.45 | 0.35-0.55 | Reduce position below this |

## 7. Options Parameters

### 7.1 Contract Selection
| Parameter | Current | Range | Description |
|-----------|---------|-------|-------------|
| `min_days_to_expiration` | 7 | 1-14 | Minimum DTE |
| `max_days_to_expiration` | 30 | 14-45 | Maximum DTE |
| `max_strike_deviation_pct` | 15% | 5-25% | OTM/ITM limit |

### 7.2 Liquidity Requirements
| Parameter | Current | Range | Description |
|-----------|---------|-------|-------------|
| `max_bid_ask_spread_pct` | 5.0% | 2-10% | Max spread allowed |
| `min_volume` | 20 | 10-100 | Minimum volume |
| `min_open_interest` | 10 | 5-50 | Minimum OI |

## 8. Data & Timing

### 8.1 Data Intervals
| Parameter | Current | Options | Description |
|-----------|---------|---------|-------------|
| `TT_DATA_INTERVAL` | 1m | 1m, 5m | Bar interval |
| `sequence_length` | 60 | 30-120 | Lookback timesteps |

### 8.2 Timing
| Parameter | Current | Range | Description |
|-----------|---------|-------|-------------|
| `cycle_interval_seconds` | 60 | 30-120 | Cycle frequency |
| `min_time_between_trades_seconds` | 300 | 60-600 | Trade cooldown |

---

## Priority Experiments

### High Impact (try first)
1. `stop_loss_pct`: 5% → 3-10% range
2. `take_profit_pct`: 12% → 8-20% range
3. `max_hold_minutes`: 45 → 20-90 range
4. `HMM_STRONG_BULLISH/BEARISH`: 0.70/0.30 → 0.65/0.35 or 0.75/0.25
5. Entry controller: bandit vs consensus vs q_scorer

### Medium Impact
6. `feature_dim`: 50 vs 59 (with time/gaussian features)
7. `training_min_confidence`: 0.0 vs 0.10 vs 0.20
8. `PNL_CAL_MIN_PROB`: 0.30 vs 0.40 vs 0.50
9. Temporal encoder: TCN vs Transformer
10. `learning_rate`: 0.0001 vs 0.0003 vs 0.0005

### Lower Impact
11. `trailing_stop_activation_pct`: 4% vs 6% vs 8%
12. `max_concurrent_positions`: 3 vs 5 vs 7
13. `batch_size`: 64 vs 128 vs 256
14. Options DTE range: 7-30 vs 14-45
15. `sequence_length`: 30 vs 60 vs 120

---

## 8. CRITICAL FINDING: Pre-trained Model State

**The most important parameter is NOT a config value - it's the neural network's training state.**

### The Discovery

| Configuration | Trade Rate | P&L |
|--------------|------------|-----|
| Fresh neural network | 10-15% | Loses money |
| Pre-trained neural network | **1.4%** | **+1016% P&L** |

### Why Pre-trained State Matters

1. **Conservative Predictions**: Trained NN outputs lower confidence values
2. **Higher Rejection Rate**: More signals fail the bandit_gate (20% conf, 0.08% edge)
3. **Extreme Selectivity**: Only ~1.5% of signals pass through
4. **Quality Over Quantity**: Fewer trades but +$700 per trade

### How to Use Pre-trained State

```bash
# 1. Copy state from profitable run
mkdir -p models/my_run/state
cp -r models/long_run_20k/state/* models/my_run/state/

# 2. Run with LOAD_PRETRAINED flag
LOAD_PRETRAINED=1 MODEL_RUN_DIR=models/my_run python scripts/train_time_travel.py
```

### Best Performing Configuration

| Parameter | Value | Notes |
|-----------|-------|-------|
| Entry Controller | bandit | Default HMM-based entry |
| training_min_confidence | 0.20 | Minimum 20% NN confidence |
| training_min_abs_predicted_return | 0.0008 | Minimum 0.08% edge |
| Pre-trained State | **models/long_run_20k** | **KEY TO SUCCESS** |

**Result**: +1016% P&L, +$705/trade, 1.43% trade rate
