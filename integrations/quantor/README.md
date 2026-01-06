# Quantor-MTFuzz Integration

**Adapted from Jerry Mahabub & John Draper's [spy-iron-condor-trading](https://github.com/trextrader/spy-iron-condor-trading) system**

## Overview

This integration brings key components from the Quantor-MTFuzz deterministic trading framework into the Gaussian system:

- **Fuzzy Position Sizing** - 9-factor membership functions for intelligent position sizing
- **Market Regime Filter** - 5-regime classification with trading gates
- **Volatility Analytics** - Realized vol, IV skew, VRP calculations
- **Data Alignment** - Chain alignment with confidence decay for backtesting quality

## Credits

**Authors:**
- **Jerry Mahabub** (trextrader) - Primary developer
- **John Draper** - Co-developer, system architecture

**Repository:** [spy-iron-condor-trading](https://github.com/trextrader/spy-iron-condor-trading)
**Integration Date:** January 2026

## Components

### 1. Fuzzy Position Sizer (`fuzzy_sizer.py`)

Calculates position size using 9 fuzzy membership functions:

| Factor | Call Membership | Put Membership |
|--------|-----------------|----------------|
| RSI | Lower is better (oversold) | Higher is better (overbought) |
| ADX | Higher trending = bigger size | Higher trending = bigger size |
| BBands | Near lower band = better | Near upper band = better |
| ATR | Lower vol = bigger size | Lower vol = bigger size |
| Volume | Higher = bigger size | Higher = bigger size |
| MACD | Positive = bigger size | Negative = bigger size |
| Stochastic | Oversold = better | Overbought = better |
| OBV Trend | Positive = bigger size | Negative = bigger size |
| Momentum | Positive = bigger size | Negative = bigger size |

**Usage:**
```python
from integrations.quantor import FuzzyPositionSizer

sizer = FuzzyPositionSizer()
size = sizer.compute_position_size(
    equity=10000,
    max_loss=500,
    direction="CALL",
    rsi=35,
    adx=28,
    bb_position=0.2,  # Near lower band
    atr_pct=0.015,
    volume_ratio=1.5,
    macd=0.5,
    stoch_k=25,
    obv_trend=1.0,
    momentum=0.02
)
```

### 2. Regime Filter (`regime_filter.py`)

Classifies market into 5 regimes and applies trading gates:

| Regime | VIX Range | Trend | Trading Allowed |
|--------|-----------|-------|-----------------|
| CRASH | > 35 | Any | NO (circuit breaker) |
| BULL_TREND | < 20 | Up | CALLS only |
| BEAR_TREND | < 25 | Down | PUTS only |
| HIGH_VOL_RANGE | 25-35 | Sideways | Both (reduced size) |
| LOW_VOL_RANGE | < 18 | Sideways | Both (neutral) |

**Usage:**
```python
from integrations.quantor import RegimeFilter, MarketRegime

filter = RegimeFilter()
regime = filter.classify_regime(
    vix_level=22,
    trend_direction="UP",
    volatility_percentile=0.6
)

if filter.should_block_trade(regime, "CALL"):
    print("Trade blocked by regime filter")

bias = filter.get_direction_bias(regime)  # 1.0 for BULL, -1.0 for BEAR
```

### 3. Volatility Analyzer (`volatility.py`)

Advanced volatility analytics from Jerry's system:

| Metric | Description |
|--------|-------------|
| Realized Vol | 20-day annualized historical volatility |
| IV Skew | (25d Put IV - 25d Call IV) / ATM IV |
| VRP | IV - Realized Vol (volatility risk premium) |
| Vol Regime | LOW/NORMAL/HIGH/EXTREME classification |
| ATR Stop | Dynamic stop loss multiplier based on ATR |

**Usage:**
```python
from integrations.quantor import VolatilityAnalyzer

analyzer = VolatilityAnalyzer()
metrics = analyzer.compute_full_metrics(
    price_history=df['close'],
    atm_iv=0.22,
    put_25d_iv=0.25,
    call_25d_iv=0.20
)

print(f"Realized Vol: {metrics.realized_vol:.1%}")
print(f"IV Skew: {metrics.iv_skew:.3f}")
print(f"VRP: {metrics.vrp:.1%}")
print(f"Regime: {metrics.vol_regime}")
```

### 4. Data Alignment (`data_alignment.py`)

Ensures backtest quality by tracking data freshness:

| Mode | Description | iv_conf |
|------|-------------|---------|
| EXACT | Data timestamp matches request | 1.0 |
| PRIOR | Using earlier data (within tolerance) | Decays with lag |
| STALE | Data too old (beyond max_lag) | < 0.5 |
| NONE | No data available | 0.0 |

**Key Features:**
- `iv_conf` decays exponentially: `0.5^(lag_sec / half_life_sec)`
- Fail-fast mode stops backtests when data quality is poor
- Alignment diagnostics track exact%, prior%, stale%, none%

**Usage:**
```python
from integrations.quantor import DataAligner, AlignmentDiagnosticsTracker, AlignmentConfig

config = AlignmentConfig(
    max_lag_sec=600,           # 10 minutes max
    iv_decay_half_life_sec=300, # Confidence halves every 5 min
    fail_fast_enabled=True,
    fail_fast_stale_threshold=0.3  # Stop if >30% stale
)

tracker = AlignmentDiagnosticsTracker(config)
aligner = DataAligner(config)

# In your backtest loop:
alignment = aligner.align_chain(
    requested_ts=sim_time,
    chain_data=options_chain,
    chain_ts=actual_data_timestamp
)
tracker.record(alignment)

# Use iv_conf as confidence multiplier
adjusted_confidence = raw_confidence * alignment.iv_conf

# Check fail-fast condition
tracker.check_and_raise()  # Raises RuntimeError if data quality too poor
```

### 5. Facade (`facade.py`)

Unified interface combining all components:

```python
from integrations.quantor import QuantorFacade

quantor = QuantorFacade()

# Full analysis
result = quantor.analyze(
    equity=10000,
    max_loss=500,
    direction="CALL",
    vix_level=18,
    trend_direction="UP",
    price_history=df['close'],
    indicators={'rsi': 35, 'adx': 28, ...}
)

print(f"Should trade: {result.should_trade}")
print(f"Position size: {result.position_size}")
print(f"Regime: {result.regime}")
print(f"Direction bias: {result.direction_bias}")
```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `ALIGNMENT_ENABLED` | `1` | Enable data alignment tracking |
| `ALIGNMENT_FAIL_FAST` | `1` | Stop backtests on poor data quality |
| `ALIGNMENT_MAX_LAG_SEC` | `600` | Maximum acceptable data lag (10 min) |
| `ALIGNMENT_IV_DECAY_HALF_LIFE` | `300` | Confidence decay half-life (5 min) |

## Integration with train_time_travel.py

The alignment system is wired into the main training script:

1. **Imports alignment tracking** at startup
2. **Initializes tracker** before main loop
3. **Tracks alignment** each cycle (computes iv_conf)
4. **Applies iv_conf as confidence multiplier** to all entry controllers
5. **Reports alignment stats** in final summary
6. **Fail-fast** stops backtest if data quality degrades

## Testing

```bash
# Run all Quantor integration tests
python -m pytest tests/test_quantor_integration.py -v

# 41 tests covering:
# - Fuzzy position sizing (9 tests)
# - Regime filtering (8 tests)
# - Volatility analytics (8 tests)
# - Facade integration (5 tests)
# - Data alignment (11 tests)
```

## Quantor-MTFuzz Philosophy (Jerry Mahabub & John Draper)

From the Quantor-MTFuzz documentation:

> "Risk-first, capital-bound, liquidity-filtered execution. Every trade is a provable consequence of validated premises - designed for auditability and traceability."

Key principles adapted:
1. **Hard constraints are non-negotiable** - Regime gates veto trades entirely
2. **Soft conditions modulate** - Fuzzy scores adjust position size
3. **2% max risk per trade** - Capital preservation is paramount
4. **Data quality matters** - Alignment tracking ensures backtest reliability

## References

- [spy-iron-condor-trading repo](https://github.com/trextrader/spy-iron-condor-trading) by Jerry Mahabub & John Draper
- `docs/jerry-info/SUMMARY.md` - Full system summary
- `docs/jerry-info/EQUATIONS.md` - Mathematical formulas
- `docs/jerry-info/APPLICABLE_TO_US.md` - Integration recommendations
