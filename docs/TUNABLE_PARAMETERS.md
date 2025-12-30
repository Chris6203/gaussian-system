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
| `training_min_confidence` | **0.30** | 0.0-0.50 | Minimum NN confidence to trade |
| `training_min_abs_predicted_return` | **0.0013** | 0.0-0.002 | Minimum predicted edge (0.13%) |

**Note**: Phase 27 optimization found 30%/0.13% reduces losses by 97% vs 20%/0.08% defaults.

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

### 3.1 Predictor Architecture (**NEW - Phase 28**)
| Type | P&L | Description | Status |
|------|-----|-------------|--------|
| `v3_multi_horizon` | **+1327%** | Multi-horizon (5m,15m,30m,45m) | **BEST** |
| `v2_slim_bayesian` | baseline | Bayesian heads + RBF kernels | Default |
| `v1_original` | - | Original architecture | Legacy |

**Set via**: `PREDICTOR_ARCH=v3_multi_horizon` environment variable

### 3.2 Temporal Encoder
| Type | P&L | Description | Status |
|------|-----|-------------|--------|
| `transformer` | **+801%** | Attention-based transformer | **2nd Best** |
| `tcn` | baseline | Temporal Convolutional Network | Default |
| `lstm` | - | Long Short-Term Memory | Available |

**Set via**: `TEMPORAL_ENCODER=transformer` environment variable

### 3.3 Architecture Components
| Parameter | Current | Options | Description |
|-----------|---------|---------|-------------|
| `NORM_TYPE` | layernorm | layernorm, rmsnorm | Normalization layer |
| `ACTIVATION_TYPE` | gelu | gelu, geglu, swiglu | Activation function |
| `RBF_GATED` | 0 | 0, 1 | Gated RBF kernel |
| `feature_dim` | 50 | 50, 59 | Feature dimension |

### 3.4 Learning Parameters
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

### HIGH IMPACT - ARCHITECTURE (Phase 28 validated)
1. **`PREDICTOR_ARCH=v3_multi_horizon`** → +1327% P&L (10K validated)
2. **`TEMPORAL_ENCODER=transformer`** → +801% P&L (10K validated)
3. V3 + Transformer combined (untested)

### Medium Impact - Thresholds
4. `training_min_confidence`: 0.30 (optimal) vs 0.20 (default)
5. `training_min_abs_predicted_return`: 0.0013 (optimal) vs 0.0008 (default)
6. `stop_loss_pct`: 5% → 3-10% range
7. `take_profit_pct`: 12% → 8-20% range
8. `max_hold_minutes`: 45 → 20-90 range

### Lower Impact
9. `HMM_STRONG_BULLISH/BEARISH`: 0.70/0.30 → 0.65/0.35 or 0.75/0.25
10. `PNL_CAL_MIN_PROB`: 0.30 vs 0.40 vs 0.50
11. `learning_rate`: 0.0001 vs 0.0003 vs 0.0005
12. `trailing_stop_activation_pct`: 4% vs 6% vs 8%

---

## 9. CRITICAL FINDING: Architecture > Pre-training (Phase 28)

**Phase 28 superseded pre-training findings: V3 architecture achieves +1327% without pre-training.**

### Architecture Comparison (10K cycles, fresh model)

| Configuration | Trade Rate | P&L | Per-Trade |
|--------------|------------|-----|-----------|
| V3 Multi-Horizon | 15.5% | **+1327%** | **+$42.76** |
| Transformer Encoder | 22.6% | +801% | +$17.73 |
| V2 + Optimal Thresholds (30%/0.13%) | 0.87% | -1.8% | -$1.04 |
| V2 Default (20%/0.08%) | 15-17% | -62% to -91% | Loses money |

### Best Performing Configuration (NEW)

| Parameter | Value | Notes |
|-----------|-------|-------|
| **PREDICTOR_ARCH** | **v3_multi_horizon** | **KEY TO SUCCESS** |
| Entry Controller | bandit | Default HMM-based entry |
| training_min_confidence | 0.30 | Minimum 30% NN confidence |
| training_min_abs_predicted_return | 0.0013 | Minimum 0.13% edge |

**Result**: +1327% P&L, +$42.76/trade, 15.5% trade rate (10K cycles)

### How to Run Best Configuration

```bash
# Best configuration - V3 Multi-Horizon
PREDICTOR_ARCH=v3_multi_horizon python scripts/train_time_travel.py

# Alternative - Transformer encoder
TEMPORAL_ENCODER=transformer python scripts/train_time_travel.py

# Experimental - Combined
PREDICTOR_ARCH=v3_multi_horizon TEMPORAL_ENCODER=transformer python scripts/train_time_travel.py
```

### Why V3 Works

1. **Multi-Horizon Predictions**: Outputs at 5m, 15m, 30m, 45m horizons
2. **Solves Horizon Misalignment**: No longer predicting 15min but holding 45min
3. **Backward Compatible**: Default outputs mapped to 15m for existing code
