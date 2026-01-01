# Quantor-MTFuzz Equations Reference

All equations from Jerry's Mathematical-Logical-Decision-Framework.

---

## A. Data & Preprocessing

### A.1-A.3: OHLCV Data Structure
```
D^(k)_t = {O^(k)_t, H^(k)_t, L^(k)_t, C^(k)_t, V^(k)_t}

where k ∈ {5min, 60min, daily}
```

### A.4: Returns
```
r^(k)_t = ln(C^(k)_t / C^(k)_{t-1})
```

### A.5: Realized Volatility
```
σ^(k)_t = sqrt(Σ_{i=1}^{n} (r^(k)_{t-i})² / n)
```

---

## B. Regime Classification

### A.6: Regime State
```
R_t ∈ {TRENDING, RANGING, HIGH_VOL, DISALLOWED}
```

### A.7: Trend Indicator
```
Trend_t = sign(EMA_fast - EMA_slow)
```

### A.8: Volatility Regime
```
Vol_regime = {
    LOW     if σ_t < σ_25th
    NORMAL  if σ_25th ≤ σ_t < σ_75th
    HIGH    if σ_t ≥ σ_75th
}
```

---

## C. Multi-Timeframe Consensus

### A.9: MTF Signal Aggregation
```
S^MTF_t = Σ_{k} w_k × S^(k)_t

where:
  w_daily = 0.50
  w_60min = 0.35
  w_5min  = 0.15
  Σ w_k = 1.0
```

### A.10: Direction Bias
```
B_t = sign(S^MTF_t + β_t)

where β_t is a small bias term for tie-breaking
```

---

## D. Fuzzy Logic Inference

### A.11: Membership Functions
```
μ_j(x) : ℝ → [0,1]

Common forms:
  Triangular: μ(x) = max(0, 1 - |x - c| / w)
  Trapezoidal: μ(x) = max(0, min((x-a)/(b-a), 1, (d-x)/(d-c)))
  Gaussian: μ(x) = exp(-(x - c)² / (2σ²))
```

### A.12: Fuzzy Confidence Score
```
F_t = Σ_{j=1}^{J} w_j × μ_j

where:
  μ_IV   = IV favorability membership
  μ_R    = Regime stability membership
  μ_MTF  = MTF alignment membership
  μ_Δ    = Delta balance membership
  μ_Liq  = Liquidity quality membership
  Σ w_j = 1.0
```

### A.13: Defuzzification (Centroid Method)
```
z* = ∫ z × μ(z) dz / ∫ μ(z) dz
```

---

## E. Risk Management

### A.14: Capital-at-Risk Constraint
```
Risk_trade / Equity_t ≤ α_max

where α_max = 0.02 (2%)
```

### A.15: Portfolio Delta
```
Δ_P = Σ_{i} q_i × Δ_i

Target: |Δ_P| ≤ Δ_max (typically ±50)
```

### A.16: Portfolio Gamma
```
Γ_P = Σ_{i} q_i × Γ_i
```

### A.17: Portfolio Vega
```
V_P = Σ_{i} q_i × ν_i
```

### A.18: Portfolio Theta
```
Θ_P = Σ_{i} q_i × Θ_i
```

---

## F. Iron Condor Pricing

### A.19: Net Credit Received
```
NetCredit = (ShortCall.mid - LongCall.mid) + (ShortPut.mid - LongPut.mid)
```

### A.20: Maximum Risk
```
MaxRisk = StrikeWidth - NetCredit
```

### A.21: Credit Ratio
```
CR = NetCredit / MaxRisk ≥ CR_min

where CR_min = 0.25 (25%)
```

### A.22: Breakeven Points
```
Upper_BE = ShortCall.strike + NetCredit
Lower_BE = ShortPut.strike - NetCredit
```

---

## G. Position Sizing

### A.23: Base Quantity
```
q₀ = floor(α × Equity_t / MaxLossPerContract)

where:
  α = 0.02 (2% risk allocation)
  MaxLossPerContract = MaxRisk per spread
```

### A.24: Volatility Normalization
```
σ*_t = (σ_t - σ_min) / (σ_max - σ_min) ∈ [0,1]
```

### A.25: Scaling Function
```
g(F_t, σ*_t) = F_t × (1 - β × σ*_t)

where:
  F_t = fuzzy confidence [0,1]
  σ*_t = normalized volatility [0,1]
  β ∈ [0.3, 0.5] = volatility penalty factor
```

### A.26: Final Position Size
```
q = max(1, round(q₀ × g(F_t, σ*_t)))
```

---

## H. Modus Ponens Logic Chain

### Stage Implications
```
S0: DataValid      → proceed to S1
S1: RiskSound      → proceed to S2
S2: RegimeOK       → proceed to S3
S3: MTFAligned     → proceed to S4
S4: FuzzyConfident → proceed to S5
S5: VolAdjusted    → proceed to S6
S6: QtyValid       → proceed to S7
S7: FinalDecision  → EXECUTE or REJECT
```

### Logical Form
```
For each stage:
  P_i ∧ C_i → P_{i+1}

where:
  P_i = premise i is true
  C_i = condition i passes

If any C_i = FALSE: REJECT trade
If all C_i = TRUE: EXECUTE trade
```

---

## I. Advanced Formulas

### A.27: Vanna-Volga Pricing
```
V_VV = V_BS + ω₁ × C_ATM + ω₂ × C_RR + ω₃ × C_BF

where:
  V_BS = Black-Scholes price
  C_ATM = ATM volatility adjustment
  C_RR = Risk reversal adjustment
  C_BF = Butterfly adjustment
  ω = hedging weights
```

### A.28: LSTM-GARCH Hybrid Volatility
```
σ²_{t+1} = ω + α × ε²_t + β × σ²_t + γ × h_LSTM(X_t)

where:
  h_LSTM = LSTM hidden state output
  X_t = input features
```

### A.29: Expected Improvement (Bayesian Optimization)
```
EI(x) = E[max(f(x) - f(x⁺), 0)]

where x⁺ = current best observation
```

### A.30: Stochastic Optimal Stopping
```
τ* = argmax_{τ ∈ [0.5T, 0.75T]} E[V_τ - V_T]

where:
  T = trade duration
  V_τ = value at stopping time

Optimal exit: 50-75% of expected trade duration
```

---

## J. Liquidity Filters

### A.31: Bid-Ask Spread Check
```
Spread_pct = (Ask - Bid) / Mid × 100 ≤ Spread_max

where Spread_max typically = 5%
```

### A.32: Volume Filter
```
Volume_t ≥ Volume_min

where Volume_min is contract-specific minimum
```

### A.33: Open Interest Filter
```
OI_t ≥ OI_min

Ensures sufficient market depth
```

---

## Quick Reference Table

| Equation | Symbol | Description |
|----------|--------|-------------|
| A.9 | S^MTF_t | Multi-timeframe consensus signal |
| A.12 | F_t | Fuzzy confidence score |
| A.14 | α_max | Max capital-at-risk (2%) |
| A.21 | CR | Credit-to-risk ratio |
| A.23 | q₀ | Base position quantity |
| A.25 | g() | Volatility-adjusted scaling |
| A.26 | q | Final position size |
| A.30 | τ* | Optimal stopping time |
