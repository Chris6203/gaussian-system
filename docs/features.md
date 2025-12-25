# Feature Families Documentation

## Overview

This document describes all feature families available in the SPY options trading bot's enhanced feature pipeline. Features are organized into seven families, each capturing different aspects of market dynamics.

## Feature Families Summary

| Family | Description | Key Signals |
|--------|-------------|-------------|
| 1. Equity/ETF | Cross-asset price dynamics | Sector rotation, market regime |
| 2. Options Surface | IV and options positioning | Volatility expectations, dealer flow |
| 3. Breadth | Market participation | Risk-on/off, trend confirmation |
| 4. Macro | Rates and dollar | Flight-to-safety, macro regime |
| 5. Option Positioning | Put/call metrics | Sentiment, hedging demand |
| 6. Crypto | Digital asset signals | Risk appetite proxy |
| 7. Meta | Time and regime | Session effects, expiry dynamics |

---

## 1. Cross-Asset Equity/ETF Features

**Module:** `features/equity_etf.py`

**Intuition:** Different market segments (sectors, cap sizes, risk assets) often lead or lag SPY. Cross-asset features capture rotation, relative strength, and broad market health.

### Symbols Used
- **Core Indices:** SPY, QQQ, IWM, DIA
- **Sector ETFs:** XLF (financials), XLK (tech), XLE (energy), XLU (utilities), XLY (discretionary), XLP (staples)
- **Credit:** HYG (high yield), LQD (investment grade)

### Features
| Feature Pattern | Description |
|-----------------|-------------|
| `{SYM}_ret_{h}` | Simple return at horizon h bars |
| `{SYM}_log_ret_{h}` | Log return at horizon h bars |
| `{SYM}_rvol_{w}` | Rolling realized volatility (window w) |
| `{SYM}_price_zscore_{w}` | Price z-score vs rolling mean |
| `{SYM}_ret_zscore_{w}` | Return z-score vs rolling mean |
| `{SYM}_price_vs_ma{n}` | Price position vs MA(n) |
| `{SYM}_ma{s}_over_ma{l}` | Short/long MA ratio |
| `{SYM}_rel_vol_{w}` | Current volume / average volume |

### Example Features
```
eq_SPY_ret_1, eq_SPY_ret_5, eq_SPY_ret_15, eq_SPY_ret_60
eq_QQQ_rvol_20, eq_QQQ_rvol_60
eq_XLF_ma10_over_ma50
eq_IWM_price_zscore_20
```

---

## 2. Volatility & Options Surface Features

**Module:** `features/options_surface.py`

**Intuition:** Options markets are forward-looking. IV levels, skew, and positioning reveal market expectations and dealer hedging flows that often precede price moves.

### Key Concepts
- **ATM IV:** At-the-money implied volatility reflects expected near-term movement
- **IV Skew:** OTM put IV > OTM call IV indicates fear/hedging demand
- **GEX (Gamma Exposure):** Dealer positioning that can pin or accelerate moves

### Features
| Feature | Description |
|---------|-------------|
| `atm_iv_call` | ATM call implied volatility |
| `atm_iv_put` | ATM put implied volatility |
| `atm_iv_avg` | Average ATM IV |
| `iv_skew_25d` | 25-delta put IV minus call IV |
| `iv_skew_otm` | OTM put IV minus OTM call IV |
| `iv_term_slope` | IV slope across expirations |
| `oi_total_calls` | Total call open interest |
| `oi_total_puts` | Total put open interest |
| `oi_put_call_ratio` | Put/call OI ratio |
| `vol_total_calls` | Total call volume |
| `vol_total_puts` | Total put volume |
| `vol_put_call_ratio` | Put/call volume ratio |
| `gex_total` | Total gamma exposure (millions) |
| `gex_near_atm` | Near-ATM gamma exposure |

### Interpretation
- **High `oi_put_call_ratio`:** Elevated hedging, bearish sentiment
- **Positive `iv_skew_otm`:** Fear premium in puts (normal)
- **Large negative `gex_total`:** Dealers short gamma, may amplify moves

---

## 3. Breadth / Cross-Section Features

**Module:** `features/breadth.py`

**Intuition:** Healthy rallies have broad participation. Narrow leadership (few stocks/sectors driving moves) often precedes reversals.

### Features
| Feature | Description |
|---------|-------------|
| `breadth_up_frac_short` | Fraction of symbols with positive short-term return |
| `breadth_up_frac_long` | Fraction with positive longer-term return |
| `breadth_above_ma_frac` | Fraction above their moving average |
| `xsec_ret_mean` | Cross-section mean return |
| `xsec_ret_std` | Cross-section return dispersion |
| `xsec_ret_skew` | Cross-section return skewness |
| `breadth_momentum` | Change in breadth (improving/deteriorating) |
| `breadth_divergence` | SPY return vs. cross-section average |

### Sector Features
| Feature | Description |
|---------|-------------|
| `sector_{name}_ret` | Sector return |
| `sector_{name}_up` | Binary: sector positive |
| `sector_risk_on` | Tech vs defensive spread |
| `sector_cyclical_spread` | Cyclical vs defensive spread |

### Interpretation
- **Low `breadth_up_frac_short` + SPY up:** Narrow rally, potential reversal
- **High `xsec_ret_std`:** High dispersion, sector rotation active
- **Positive `sector_risk_on`:** Risk-seeking behavior

---

## 4. Rates / Macro Proxy Features

**Module:** `features/macro.py`

**Intuition:** Interest rates and dollar strength significantly impact equity valuations and capital flows. Bond-equity correlations signal risk regimes.

### Symbols Used
- **Treasury:** TLT (20+ yr), IEF (7-10 yr), SHY (1-3 yr)
- **Dollar:** UUP (US Dollar Index ETF)

### Features
| Feature | Description |
|---------|-------------|
| `tlt_ret_{h}` | Long-duration treasury return |
| `ief_ret_5` | Intermediate treasury return |
| `shy_ret_5` | Short-duration treasury return |
| `yield_curve_slope` | TLT - SHY return spread (curve proxy) |
| `duration_momentum` | Is long duration outperforming? |
| `tlt_vol_20` | Treasury volatility |
| `uup_ret_{h}` | Dollar index return |
| `dollar_strength` | Dollar momentum indicator |
| `dollar_vol_20` | Dollar volatility |
| `bond_equity_corr` | TLT-SPY correlation |
| `dollar_equity_corr` | UUP-SPY correlation |
| `flight_to_safety` | TLT strength when SPY weak |

### Interpretation
- **Positive `yield_curve_slope`:** Curve steepening (growth expectations)
- **Negative `bond_equity_corr`:** Normal risk-off relationship
- **High `flight_to_safety`:** Active de-risking

---

## 5. Option Positioning Features

**Module:** `features/options_surface.py` (combined with surface features)

**Intuition:** Option positioning reveals institutional hedging and speculative flows that often lead price action.

### Features
| Feature | Description |
|---------|-------------|
| `oi_near_money_calls` | Near-ATM call open interest |
| `oi_near_money_puts` | Near-ATM put open interest |
| `vol_near_money_calls` | Near-ATM call volume |
| `vol_near_money_puts` | Near-ATM put volume |
| `near_money_put_bias` | Put OI / (Put + Call) near ATM |

### Interpretation
- **High `near_money_put_bias`:** Elevated put hedging demand
- **Spike in `vol_near_money_calls`:** Call buying activity

---

## 6. Crypto Risk Proxy Features

**Module:** `features/crypto.py`

**Intuition:** Crypto markets often lead equity risk sentiment, especially in risk-on/off transitions. 24/7 trading provides after-hours signals.

### Symbols Used
- **Primary:** BTC-USD (Bitcoin)
- **Secondary:** ETH-USD (Ethereum)

### Features
| Feature | Description |
|---------|-------------|
| `btc_ret_{h}` | Bitcoin return at horizon h |
| `btc_vol_{w}` | Bitcoin realized volatility |
| `btc_ma_ratio` | BTC short/long MA ratio |
| `btc_zscore` | BTC price z-score |
| `eth_ret_5` | Ethereum return |
| `eth_btc_ratio` | ETH/BTC relative strength |
| `crypto_risk_on` | Combined crypto risk indicator |
| `crypto_momentum` | Crypto momentum score |
| `crypto_vol_regime` | High/low vol regime |
| `btc_spy_corr` | BTC-SPY correlation |
| `btc_qqq_corr` | BTC-QQQ correlation |

### Interpretation
- **Positive `crypto_risk_on`:** Crypto strength = risk appetite
- **High `crypto_vol_regime`:** Elevated crypto volatility (caution)
- **Rising `btc_spy_corr`:** Crypto-equity coupling increasing

---

## 7. Meta / Regime / Time Features

**Module:** `features/meta.py`

**Intuition:** Market behavior varies systematically by time-of-day, day-of-week, and around option expirations.

### Time Features
| Feature | Description |
|---------|-------------|
| `minutes_since_open` | Minutes since 9:30 AM ET |
| `minutes_to_close` | Minutes until 4:00 PM ET |
| `session_progress` | 0 at open, 1 at close |
| `is_first_30min` | First 30 minutes (high vol) |
| `is_last_hour` | Last hour (power hour) |
| `is_midday` | Mid-session (lower vol) |
| `is_power_hour` | Final hour |
| `time_sin`, `time_cos` | Cyclic time encoding |

### Expiry Features
| Feature | Description |
|---------|-------------|
| `days_to_expiry` | DTE for the option |
| `is_0dte` | Same-day expiry |
| `is_1dte` | Next-day expiry |
| `is_weekly` | Within 7 days |
| `is_opex_day` | Monthly options expiration |
| `is_opex_week` | OPEX week |
| `days_to_opex` | Days until monthly OPEX |
| `is_monday` | Monday |
| `is_friday` | Friday |
| `day_of_week` | Day of week (0-1) |

### Vol Regime Features
| Feature | Description |
|---------|-------------|
| `vol_regime_low` | Low volatility regime |
| `vol_regime_normal` | Normal volatility regime |
| `vol_regime_high` | High volatility regime |
| `vol_percentile` | Vol percentile in history |
| `vol_zscore` | Vol z-score |
| `vix_regime_low` | VIX < 15 |
| `vix_regime_elevated` | VIX 15-25 |
| `vix_regime_high` | VIX > 25 |

### Interpretation
- **`is_first_30min`:** Higher volatility, gap fills
- **`is_power_hour`:** Increased activity, directional moves
- **`is_opex_week`:** Pin risk, elevated gamma effects
- **`vol_regime_high`:** Larger moves, adjusted position sizing

---

## Configuration

Features are configured in `config.json`:

```json
{
    "feature_pipeline": {
        "enabled": true,
        "enable_equity_etf": true,
        "enable_options_surface": true,
        "enable_breadth": true,
        "enable_macro": true,
        "enable_crypto": true,
        "enable_meta": true,
        "equity_symbols": ["SPY", "QQQ", "IWM", ...],
        "return_horizons": [1, 5, 15, 60],
        "vol_windows": [20, 60, 120],
        "use_prefix": true
    }
}
```

### Enabling/Disabling Families

Set `enable_{family}: false` to disable specific feature families without affecting others.

### Feature Prefixes

With `use_prefix: true`, features are prefixed by family:
- `eq_` - Equity/ETF
- `opt_` - Options surface
- `brd_` - Breadth
- `mac_` - Macro
- `cry_` - Crypto
- `meta_` - Meta/time

---

## Usage

### Basic Usage
```python
from features.pipeline import FeaturePipeline, FeatureConfig

# Create pipeline
config = FeatureConfig()
pipeline = FeaturePipeline(config)

# Compute features
features = pipeline.compute_features(
    equity_data={'SPY': spy_df, 'QQQ': qqq_df, ...},
    options_chain=spy_chain,
    spot_price=450.0
)
```

### Integration with Existing Bot
```python
from features.integration import EnhancedFeatureManager, integrate_with_bot

# Attach to bot
manager = integrate_with_bot(bot_instance)

# Features are automatically augmented
```

### Getting Feature Array
```python
# For model input
arr = pipeline.get_feature_array(
    equity_data=data,
    spot_price=450.0
)
# Returns: np.ndarray of shape (n_features,)
```

---

## Feature Count Summary

| Family | Approximate Features |
|--------|---------------------|
| Equity/ETF | 100-200 (depends on symbols) |
| Options Surface | 17 |
| Breadth | 12 |
| Macro | 15 |
| Crypto | 13 |
| Meta | 25 |
| **Total** | **~180-280** |

Note: Actual count depends on configuration and available data.




