# Jerry's Quantor-MTFuzz System Summary

## Overview

**Quantor-MTFuzz** is a deterministic modus ponens decision engine for SPY options trading (Iron Condors). Every trade is a provable consequence of validated premises - designed for auditability and traceability.

## Documents Index

| Document | Pages | Purpose |
|----------|-------|---------|
| Quantor-mtfuz-zip-README.pdf | 3 | Package overview, reading order |
| SPYOptionTrader-Mathematical-Logical-Decision-Framework.pdf | 43 | Core math spec (equations A.1-A.20) |
| Consolidated-Equation-List-For-Quantor-mtfuz.pdf | 39 | All equations with examples |
| Developer-Focused-Summary-Integrating-Fuzzy-Position-Sizing.pdf | 6 | Python implementation code |
| SPYOptionTrader-Codex-Cross-Reference-of-all-functions.pdf | 30 | Function-to-equation mapping |
| CODE-FRAMEWORK_CROSS_REFERENCE_DIAGRAM.pdf | 2 | System architecture visual |
| Symbolic-Logic-Fuzzy-Flow.pdf | 7 | 8-stage modus ponens decision tree |
| Structural-Integration-and-Optimization.pdf | 15 | Research methods white paper |
| Software-Test-Status.pdf | 11 | Current implementation state |
| structural-integration-diagrams-12-28-2025.pdf | 9 | Flow diagrams for all components |
| Appendix-E-Diagrams.pdf | 5 | Monte Carlo convergence charts |
| fuzzylogic_2.txt | 1 | Graphviz position sizing diagram |

---

## Core Philosophy

1. **Risk-first, capital-bound, liquidity-filtered** execution
2. **Deterministic modus ponens** - every trade is logically provable
3. **Iron Condor strategy** on SPY options (4-leg spread)
4. **Auditability over raw performance**

---

## System Architecture

### Data Flow
```
Market Data → OHLCV per Timeframe → Regime Classification → MTF Consensus
    ↓
Hard Constraints (Veto Gates) → Fuzzy Logic Engine → Position Sizing → Execute/Reject
```

### Multi-Timeframe Weights
| Timeframe | Weight | Purpose |
|-----------|--------|---------|
| Daily | 50% | Dominant trend direction |
| 60-min | 35% | Intermediate structure |
| 5-min | 15% | Entry timing refinement |

**MTF Consensus Formula:**
```
S^MTF_t = Σ w_k × S^(k)_t
```

---

## Decision System

### Hard Constraints (Must Pass - Veto Power)

These are **gates** - if any fail, trade is rejected:

1. **Regime Validity** - Must be Trending, Ranging, or High-Vol (not Disallowed)
2. **MTF Alignment** - Multi-timeframe consensus above threshold
3. **Credit-to-Risk Ratio** - CR ≥ 0.25 (for Iron Condors)
4. **Capital-at-Risk** - Risk_trade / Equity ≤ 2%
5. **Greek Exposure Limits** - Delta, Gamma, Vega, Theta within bounds
6. **Liquidity Gate** - Minimum volume, maximum bid-ask spread
7. **Macro Event Halt** - No trades during CPI, FOMC, etc.

### Soft Conditions (Modulate Confidence - No Veto)

These affect position size but don't reject trades:

| Condition | Symbol | Range | Description |
|-----------|--------|-------|-------------|
| IV Favorability | μ_IV | [0,1] | Is implied volatility favorable? |
| Regime Stability | μ_R | [0,1] | How stable is current regime? |
| MTF Alignment | μ_MTF | [0,1] | How aligned are timeframes? |
| Delta Balance | μ_Δ | [0,1] | Is portfolio delta neutral? |
| Liquidity Quality | μ_Liq | [0,1] | How good is market liquidity? |

---

## 8-Stage Decision Tree (Modus Ponens)

```
S0: System Validity     → Are data feeds OK?
        ↓ TRUE
S1: Risk Soundness      → Within capital limits?
        ↓ TRUE
S2: Structural Validity → Is regime acceptable?
        ↓ TRUE
S3: Context Validity    → Is MTF aligned?
        ↓ TRUE
S4: Fuzzy Confidence    → Is F_t ≥ F_min?
        ↓ TRUE
S5: Volatility Penalty  → Apply σ adjustment
        ↓
S6: Scaling Soundness   → Is final quantity valid?
        ↓ TRUE
S7: Final Decision      → EXECUTE or REJECT
```

Each stage is a logical implication: `P → Q`. If premise P is false, trade is rejected.

---

## Position Sizing

### Formula
```
q = q₀ × g(F_t, σ*_t)

where:
  q₀ = floor(α × Equity / MaxLossPerContract)  -- Base quantity
  α = 0.02 (2% max risk per trade)
  F_t = Σ w_j × μ_j  -- Fuzzy confidence score [0,1]
  σ*_t = normalized volatility [0,1]
  g() = scaling function (reduces size when vol high or confidence low)
```

### Scaling Function
```
g(F_t, σ*_t) = F_t × (1 - β × σ*_t)

where β ∈ [0.3, 0.5] is volatility sensitivity
```

---

## Iron Condor Specifics

### Structure
- **Short Call** at strike K_C (sell)
- **Long Call** at strike K_C + w (buy, protection)
- **Short Put** at strike K_P (sell)
- **Long Put** at strike K_P - w (buy, protection)

### Credit Calculation
```
NetCredit = (ShortCall.mid - LongCall.mid) + (ShortPut.mid - LongPut.mid)
MaxRisk = w - NetCredit
CreditRatio = NetCredit / MaxRisk ≥ 0.25 required
```

---

## Advanced Techniques (Research/Future)

1. **Vanna-Volga Pricing**: V_VV = V_BS + ω₁C_ATM + ω₂C_RR + ω₃C_BF
2. **LSTM-GARCH Volatility**: Hybrid model for vol forecasting
3. **Bayesian Optimization**: Expected Improvement acquisition for parameters
4. **Stochastic Optimal Stopping**: Exit at τ ∈ [0.5, 0.75] of trade duration

---

## Implementation Status (from Software-Test-Status.pdf)

- Core framework: Implemented
- Fuzzy logic engine: Implemented
- Position sizing: Implemented
- Live trading: Zero trades (waiting for data completeness requirements)
- Iron Condor execution: Ready but blocked by data gates

---

## Key Takeaways

1. **Hard constraints are non-negotiable** - they veto trades entirely
2. **Soft conditions modulate** - they adjust position size, not entry
3. **2% max risk per trade** - capital preservation is paramount
4. **MTF consensus weighted** - longer timeframes have more weight
5. **Fuzzy logic is interpretable** - each factor has a [0,1] membership
6. **Exit at 50-75% of duration** - don't hold to expiry

See `EQUATIONS.md` for all mathematical formulas and `APPLICABLE_TO_US.md` for integration recommendations.
