# Regime-Based Dynamic Configuration Plan

## Goal
Have the bot automatically use different trading configurations based on the detected market regime, learned from our 577 experiments.

---

## Current State

We already have:
- `backend/regime_mapper.py` - Maps HMM + VIX to canonical regimes
- `backend/regime_strategies.py` - Has `REGIME_PARAMS` with predefined configs
- `backend/multi_dimensional_hmm.py` - Detects trend/vol/liquidity states

**Problem**: The configs are hardcoded, not learned from our experiments.

---

## Implementation Plan

### Phase 1: Track Regime Data During Experiments

**Add to experiments database:**
```sql
ALTER TABLE experiments ADD COLUMN dominant_regime TEXT;
ALTER TABLE experiments ADD COLUMN avg_vix REAL;
ALTER TABLE experiments ADD COLUMN avg_hmm_trend REAL;
ALTER TABLE experiments ADD COLUMN avg_hmm_vol REAL;
ALTER TABLE experiments ADD COLUMN regime_breakdown TEXT;  -- JSON: {regime: pct_time}
```

**Modify train_time_travel.py:**
- Track VIX and HMM states during simulation
- Calculate dominant regime (where most trades occurred)
- Save to SUMMARY.txt and database

### Phase 2: Analyze Experiments by Regime

**Create `tools/analyze_regime_performance.py`:**
```python
# For each regime, find best performing configs
# Output: regime → {stop_loss, take_profit, confidence, etc.}
```

**Example output:**
| Regime | Best Config | P&L | Win Rate | Key Settings |
|--------|-------------|-----|----------|--------------|
| HIGH_VOL | IDEA-266 | +201% | 68.1% | wider stops, higher conf |
| LOW_VOL | dec_validation | +519% | 60.6% | tighter stops, more trades |
| TRENDING | v3_jerry | +450% | 27.8% | momentum following |

### Phase 3: Create Learned Regime Config

**New file: `backend/learned_regime_configs.py`:**
```python
LEARNED_REGIME_PARAMS = {
    "HIGH_VOL_TRENDING": {
        "source_experiment": "EXP-0167_IDEA-266",
        "stop_loss_pct": 15.0,
        "take_profit_pct": 25.0,
        "min_confidence": 0.65,
        "max_hold_minutes": 30,
        ...
    },
    "LOW_VOL_SIDEWAYS": {
        "source_experiment": "dec_validation",
        ...
    }
}
```

### Phase 4: Dynamic Config Switching in Bot

**Modify `go_live_only.py` / `unified_options_trading_bot.py`:**
```python
def get_current_config(self):
    # 1. Detect current regime
    regime = self.regime_mapper.get_current_regime(vix, hmm_states)

    # 2. Load learned config for this regime
    config = LEARNED_REGIME_PARAMS.get(regime, DEFAULT_CONFIG)

    # 3. Apply config to trading parameters
    self.apply_config(config)

    logger.info(f"[REGIME] Switched to {regime} config")
```

**Regime change detection:**
```python
# Check regime every N minutes
if self.current_regime != new_regime:
    logger.info(f"[REGIME] Changed: {self.current_regime} → {new_regime}")
    self.current_regime = new_regime
    self.apply_regime_config(new_regime)
```

---

## Regime Categories (6 VIX × 3 Trend = 18 combos)

| VIX Level | Trend | Regime Name |
|-----------|-------|-------------|
| < 12 | Up | ULTRA_LOW_VOL_BULLISH |
| < 12 | Sideways | ULTRA_LOW_VOL_NEUTRAL |
| < 12 | Down | ULTRA_LOW_VOL_BEARISH |
| 12-15 | Up | LOW_VOL_BULLISH |
| 12-15 | Sideways | LOW_VOL_NEUTRAL |
| ... | ... | ... |
| > 35 | Down | EXTREME_VOL_BEARISH |

**Simplification**: Start with 6 regimes (VIX buckets only), add trend later.

---

## Implementation Order

1. **[TODAY]** Add regime tracking to experiments
2. **[TODAY]** Create analysis tool to find best configs per regime
3. **[NEXT]** Generate `learned_regime_configs.py` from analysis
4. **[NEXT]** Modify bot to use dynamic configs
5. **[LATER]** Auto-update configs as new experiments complete

---

## Quick Start Implementation

### Step 1: Analyze existing experiments by VIX

We can estimate regime from the date range of each experiment:

```python
# Get VIX data for experiment date range
# Classify as LOW/NORMAL/HIGH/EXTREME
# Group experiments by regime
# Find best performing configs per regime
```

### Step 2: Create regime selector

```python
class RegimeConfigSelector:
    def __init__(self):
        self.regime_configs = self._load_learned_configs()

    def get_config(self, vix: float, hmm_trend: float) -> dict:
        regime = self._classify_regime(vix, hmm_trend)
        return self.regime_configs.get(regime, self.default_config)
```

---

## Expected Benefits

1. **Better performance in each regime** - Use configs proven to work
2. **Reduced drawdowns** - Conservative in high-vol, aggressive in low-vol
3. **Adaptive trading** - Bot adjusts to market conditions
4. **Data-driven** - Configs based on 577 experiments, not guessing

---

## Files to Create/Modify

| File | Action |
|------|--------|
| `tools/analyze_regime_performance.py` | NEW - Analyze experiments by regime |
| `backend/learned_regime_configs.py` | NEW - Store learned configs |
| `backend/regime_config_selector.py` | NEW - Select config based on regime |
| `scripts/train_time_travel.py` | MODIFY - Track regime data |
| `core/go_live_only.py` | MODIFY - Use dynamic configs |
| `data/experiments.db` | MODIFY - Add regime columns |
