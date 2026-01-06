# Pattern Analysis - COMBO_BEST_20K_VALIDATION Run

## Summary Statistics
- **Total Trades:** 239
- **Wins:** 91 (38.1% WR)
- **Total P&L:** -$167.58
- **Average P&L per trade:** -$0.70

## Already Known Patterns (Confirmed)

| Pattern | Trades | Win Rate | Diff vs Avg | P&L | Action |
|---------|--------|----------|-------------|-----|--------|
| Monday | 51 | 25.5% | -12.6% | -$100.61 | AVOID |
| Last hour (15:xx) | 30 | 13.3% | -24.8% | -$28.60 | AVOID |
| Midday 12:xx | 35 | 60.0% | +21.9% | +$34.69 | PREFER |
| CALLS | 145 | 42.1% | +4.0% | -$53.88 | slight prefer |
| PUTS | 94 | 31.9% | -6.2% | -$113.70 | AVOID |
| FAST_CUT exits | 9 | 0.0% | -38.1% | -$177.43 | AVOID |

---

## NEW PATTERNS DISCOVERED

### 1. Sequential Trading Patterns (Win/Loss Streaks)

| Pattern | Trades | Win Rate | Diff | P&L | Action |
|---------|--------|----------|------|-----|--------|
| After 2 WINS in a row | 44 | 52.3% | +14.2% | -$0.69 | **PREFER** |
| After 1 WIN | 91 | 48.4% | +10.3% | -$3.81 | **PREFER** |
| After 1 LOSS | 147 | 32.0% | -6.1% | -$163.34 | AVOID |
| After 2 LOSSES in a row | 99 | 36.4% | -1.7% | -$24.42 | neutral |

**INSIGHT:** Win streaks tend to continue! After a win, next trade has 48.4% WR vs 32% after a loss. Consider implementing "momentum gating" - be more aggressive after wins, more cautious after losses.

### 2. Volume Spike Correlations

| Pattern | Trades | Win Rate | Diff | P&L | Action |
|---------|--------|----------|------|-----|--------|
| High Volume (1.8-3.0x) | 15 | 53.3% | +15.3% | +$18.12 | **PREFER** |
| Very Low Volume (<0.5x) | 57 | 29.8% | -8.3% | -$51.97 | **AVOID** |
| Low Volume (0.5-0.8x) | 70 | 42.9% | +4.8% | -$83.91 | neutral |
| Normal Volume (0.8-1.2x) | 53 | 37.7% | -0.3% | -$44.35 | neutral |

**INSIGHT:** High volume (1.8-3x) is the sweet spot with 53.3% WR. Very low volume (<0.5x) should be avoided.

### 3. Momentum Alignment (Trade Direction vs Price Trend)

| Pattern | Trades | Win Rate | Diff | P&L | Action |
|---------|--------|----------|------|-----|--------|
| Contrarian (CALL when down, PUT when up) | 114 | 43.9% | +5.8% | -$5.97 | PREFER |
| Aligned (CALL when up, PUT when down) | 121 | 32.2% | -5.8% | -$182.54 | AVOID |

**INSIGHT:** Contrarian trades outperform trend-following! Trading against short-term momentum yields 44% WR vs 32% for aligned trades. This supports mean-reversion strategies.

### 4. Momentum Strength at Entry (5min)

| Pattern | Trades | Win Rate | Diff | P&L | Action |
|---------|--------|----------|------|-----|--------|
| Moderate Positive (0.05-0.2%) | 19 | 47.4% | +9.3% | +$7.58 | **PREFER** |
| Weak Negative (-0.05% to 0) | 88 | 44.3% | +6.2% | +$4.52 | PREFER |
| Weak Positive (0 to 0.05%) | 109 | 31.2% | -6.9% | -$168.30 | AVOID |

**INSIGHT:** Moderate momentum extremes work better than weak/neutral momentum.

### 5. Day + Option Type Combinations

| Combo | Trades | Win Rate | Diff | P&L | Action |
|-------|--------|----------|------|-----|--------|
| Thursday + CALL | 32 | 56.3% | +18.2% | +$11.44 | **PREFER** |
| Tuesday + CALL | 26 | 50.0% | +11.9% | +$0.81 | **PREFER** |
| Friday + PUT | 21 | 47.6% | +9.5% | -$2.65 | PREFER |
| Wednesday + PUT | 12 | 8.3% | -29.8% | -$26.06 | **AVOID** |
| Monday + PUT | 17 | 23.5% | -14.6% | -$54.63 | AVOID |
| Monday + CALL | 34 | 26.5% | -11.6% | -$45.99 | AVOID |

**INSIGHT:** Thursday CALLS are the best combo (56.3% WR). Wednesday/Monday PUTs are worst.

### 6. Hour + Option Type Combinations

| Combo | Trades | Win Rate | Diff | P&L | Action |
|-------|--------|----------|------|-----|--------|
| 12:xx + PUT | 10 | 70.0% | +31.9% | +$24.73 | **PREFER** |
| 14:xx + CALL | 15 | 60.0% | +21.9% | +$37.30 | **PREFER** |
| 12:xx + CALL | 25 | 56.0% | +17.9% | +$9.96 | **PREFER** |
| 13:xx + CALL | 18 | 50.0% | +11.9% | +$3.30 | PREFER |
| 09:xx + PUT | 8 | 12.5% | -25.6% | -$29.06 | **AVOID** |
| 11:xx + PUT | 15 | 26.7% | -11.4% | -$58.12 | AVOID |
| 15:xx + CALL | 20 | 10.0% | -28.1% | -$23.59 | **AVOID** |

**INSIGHT:** Midday (12-14) is best for both CALLS and PUTS. Avoid PUTs in the morning (9-11) and CALLS in the last hour (15:xx).

### 7. Predicted Return at Entry

| Pattern | Trades | Win Rate | Diff | P&L | Action |
|---------|--------|----------|------|-----|--------|
| Small (0-0.1%) | 59 | 45.8% | +7.7% | -$0.86 | PREFER |
| Negative (<0%) | 93 | 32.3% | -5.8% | -$106.34 | AVOID |

**INSIGHT:** Moderate predicted returns (0-0.1%) outperform aggressive predictions.

### 8. Momentum + Direction Detail

| Combo | Trades | Win Rate | P&L | Action |
|-------|--------|----------|-----|--------|
| Negative Momentum + CALL | 12 | 58.3% | +$14.10 | **PREFER** |
| Positive Momentum + PUT | 9 | 44.4% | +$13.73 | PREFER |
| Negative Momentum + PUT | 7 | 0.0% | -$46.40 | **AVOID** |
| Neutral Mom + CALL | 120 | 39.2% | -$90.11 | neutral |

**INSIGHT:** Strong contrarian signal: CALLS when momentum is negative, PUTS when momentum is positive.

### 9. Calibrated Confidence + Option Type

| Combo | Trades | Win Rate | P&L | Action |
|-------|--------|----------|-----|--------|
| High Conf (60-80%) + CALL | 50 | 42.0% | -$19.09 | slight prefer |
| Med Conf (40-60%) + PUT | 48 | 25.0% | -$95.94 | **AVOID** |
| Med Conf (40-60%) + CALL | 71 | 39.4% | -$14.72 | neutral |

**INSIGHT:** Medium confidence PUTs (40-60%) are losing money badly - require higher confidence for PUT trades.

### 10. Best/Worst Day+Hour Combinations

**BEST (WR > 55%):**
| Day + Hour | Trades | Win Rate | P&L |
|------------|--------|----------|-----|
| Monday 12:xx | 7 | 71.4% | +$14.00 |
| Thursday 12:xx | 10 | 60.0% | +$4.01 |
| Tuesday 13:xx | 5 | 60.0% | -$5.00 |

**WORST (WR < 25%):**
| Day + Hour | Trades | Win Rate | P&L |
|------------|--------|----------|-----|
| Friday 15:xx | 6 | 0.0% | -$8.41 |
| Monday 15:xx | 9 | 0.0% | -$11.99 |
| Monday 10:xx | 12 | 16.7% | -$23.03 |
| Monday 11:xx | 6 | 16.7% | -$62.57 |
| Monday 14:xx | 10 | 20.0% | -$19.69 |
| Thursday 09:xx | 5 | 20.0% | -$33.77 |

---

## RECOMMENDED FILTERS (Implementation Priority)

### HIGH PRIORITY - Implement Immediately

1. **Skip Monday trades entirely** - 25.5% WR, -$100.61 loss
2. **Skip last hour (15:xx)** - 13.3% WR, -$28.60 loss
3. **Require volume_spike >= 0.5** - Very low volume (<0.5) has 29.8% WR
4. **Disable FAST_CUT exits** - 0% WR, -$177.43 loss
5. **Require higher confidence for PUTs** - PUTs at 40-60% conf have 25% WR

### MEDIUM PRIORITY - Test Next

6. **Prefer trades after previous win** - 48.4% WR vs 32% after loss
7. **Prefer volume_spike 1.8-3.0** - 53.3% WR, +$18.12 gain
8. **Prefer contrarian setups** - 43.9% WR vs 32.2% for aligned
9. **Focus on midday 12:00-14:00** - 50%+ WR during these hours
10. **Prefer Thursday CALLS** - 56.3% WR, +$11.44 gain

### Specific Combination Filters

11. **Wednesday + PUT = AVOID** - 8.3% WR
12. **Morning (9-11) + PUT = AVOID** - 12.5-26.7% WR
13. **Negative momentum + CALL = PREFER** - 58.3% WR
14. **Negative momentum + PUT = AVOID** - 0% WR

---

## Filter Implementation Code Suggestions

```python
# Volume Spike Gate
if volume_spike < 0.5:
    reject("Low volume - skip trade")

# Sequential Pattern Gate
if last_trade_was_loss:
    increase_confidence_threshold(0.1)  # Be more selective after losses

# Day/Hour Gate
if day == 'Monday':
    reject("Skip Monday entirely")
if hour >= 15:
    reject("Skip last hour")

# Direction + Momentum Alignment
if momentum_5m < -0.0005 and option_type == 'PUT':
    reject("Negative momentum + PUT is anti-pattern")
if momentum_5m > 0.0005 and option_type == 'CALL':
    # This is actually aligned, but has lower WR
    # Consider preferring contrarian instead
    pass

# PUT Confidence Gate
if option_type == 'PUT' and calibrated_confidence < 0.60:
    reject("PUTs require higher confidence")

# Midday Boost
if 12 <= hour <= 14:
    boost_confidence(0.05)  # More aggressive during best hours
```

---

## Summary: Expected Impact of Filters

If we implement the top 5 filters:
- Skip Monday: Avoid 51 trades @ 25.5% WR, save ~$100 in losses
- Skip 15:xx: Avoid 30 trades @ 13.3% WR, save ~$29 in losses
- Volume >= 0.5: Avoid 57 trades @ 29.8% WR, save ~$52 in losses
- Disable FAST_CUT: Avoid 9 trades @ 0% WR, save ~$177 in losses
- Higher conf for PUTs: Improve PUT WR from 32% to potentially 40%+

**Total potential savings: ~$358 in avoided losses**
**Remaining trades would have higher win rates**
