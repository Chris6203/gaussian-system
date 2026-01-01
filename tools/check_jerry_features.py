#!/usr/bin/env python3
"""
Check Jerry Feature Values

Quick diagnostic to see what Jerry features are being generated.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ['JERRY_FEATURES'] = '1'

from features.jerry_features import compute_jerry_features, JerryFeatures
import numpy as np

print("=" * 60)
print("JERRY FEATURE VALUE CHECK")
print("=" * 60)

# Test with different VIX levels
print("\n1. VIX Level Impact on μ_IV (IV Favorability):")
print("-" * 40)
for vix in [12, 15, 18, 22, 28, 35]:
    result = compute_jerry_features(vix_level=vix)
    print(f"   VIX={vix:2d} → μ_IV={result.mu_iv:.2f}")

# Test with different regime stability
print("\n2. Regime Stability Impact on μ_regime:")
print("-" * 40)
for bars, changes in [(5, 2), (15, 1), (30, 0), (50, 0)]:
    result = compute_jerry_features(vix_level=20, bars_in_regime=bars, regime_changes=changes)
    print(f"   Bars={bars:2d}, Changes={changes} → μ_regime={result.mu_regime:.2f}")

# Test with different volume ratios (liquidity)
print("\n3. Volume Ratio Impact on μ_liq (Liquidity):")
print("-" * 40)
for vol_ratio in [0.3, 0.7, 1.0, 1.5, 2.0, 3.0]:
    result = compute_jerry_features(vix_level=20, volume_ratio=vol_ratio)
    print(f"   Volume Ratio={vol_ratio:.1f}x → μ_liq={result.mu_liq:.2f}")

# Test composite F_t score
print("\n4. Composite F_t Score Examples:")
print("-" * 40)
# Good conditions
good = compute_jerry_features(
    vix_level=14,
    bars_in_regime=40,
    regime_changes=0,
    volume_ratio=1.5,
    bid_ask_spread_pct=0.5
)
print(f"   GOOD conditions: F_t={good.f_t:.2f}")
print(f"      μ_IV={good.mu_iv:.2f}, μ_regime={good.mu_regime:.2f}, μ_mtf={good.mu_mtf:.2f}, μ_liq={good.mu_liq:.2f}")

# Medium conditions
medium = compute_jerry_features(
    vix_level=22,
    bars_in_regime=15,
    regime_changes=1,
    volume_ratio=0.8,
    bid_ask_spread_pct=1.5
)
print(f"   MEDIUM conditions: F_t={medium.f_t:.2f}")
print(f"      μ_IV={medium.mu_iv:.2f}, μ_regime={medium.mu_regime:.2f}, μ_mtf={medium.mu_mtf:.2f}, μ_liq={medium.mu_liq:.2f}")

# Bad conditions
bad = compute_jerry_features(
    vix_level=32,
    bars_in_regime=3,
    regime_changes=3,
    volume_ratio=0.4,
    bid_ask_spread_pct=4.0
)
print(f"   BAD conditions: F_t={bad.f_t:.2f}")
print(f"      μ_IV={bad.mu_iv:.2f}, μ_regime={bad.mu_regime:.2f}, μ_mtf={bad.mu_mtf:.2f}, μ_liq={bad.mu_liq:.2f}")

# Show all 6 features that get added to NN
print("\n5. All 6 Jerry Features (NN Inputs):")
print("-" * 40)
typical = compute_jerry_features(vix_level=20, bars_in_regime=20, volume_ratio=1.0)
for name, value in typical.to_dict().items():
    print(f"   {name}: {value:.3f}")

print("\n" + "=" * 60)
print("DONE - These features are fed to the neural network when JERRY_FEATURES=1")
print("=" * 60)
