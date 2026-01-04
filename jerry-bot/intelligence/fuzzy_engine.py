# intelligence/fuzzy_engine.py
import numpy as np
from typing import Dict

# =============================================================================
# QUANTOR-MTFUZZ™: Fuzzy Position Sizing Logic
# =============================================================================

def compute_base_quantity(
    equity: float,
    max_loss_per_contract: float,
    risk_fraction: float = 0.02
) -> int:
    """
    Compute the hard ceiling on position size (Stage 1).
    q0 = floor((risk_fraction * equity) / max_loss_per_contract)
    """
    if equity <= 0.0:
        return 0

    if max_loss_per_contract <= 0.0:
        return 0

    max_risk = risk_fraction * equity
    q0 = int(max_risk // max_loss_per_contract)

    return max(q0, 0)


def compute_fuzzy_confidence(
    memberships: Dict[str, float],
    weights: Dict[str, float]
) -> float:
    """
    Compute fuzzy confidence score Ft in [0, 1] (Stage 4).
    Ft = sum(w_j * mu_j)
    """
    confidence = 0.0

    for key, mu in memberships.items():
        w = weights.get(key, 0.0)
        confidence += w * mu

    # Hard clamp
    if confidence < 0.0: return 0.0
    if confidence > 1.0: return 1.0

    return confidence


def normalize_volatility(
    realized_vol: float,
    low_vol: float,
    high_vol: float
) -> float:
    """
    Normalize volatility into sigma_star in [0, 1] (Stage 5).
    Higher sigma_star = Higher Risk = Lower Size.
    """
    if high_vol <= low_vol:
        return 1.0

    sigma_star = (realized_vol - low_vol) / (high_vol - low_vol)

    if sigma_star < 0.0: return 0.0
    if sigma_star > 1.0: return 1.0

    return sigma_star


def compute_scaling_factor(
    confidence: float,
    volatility_penalty: float,
    min_scale: float = 0.0
) -> float:
    """
    Compute g(Ft, sigma_star) (Stage 6).
    g = Ft * (1 - sigma_star)
    """
    g = confidence * (1.0 - volatility_penalty)

    if g < min_scale: g = min_scale
    if g > 1.0: g = 1.0

    return g


def compute_position_size(
    equity: float,
    max_loss_per_contract: float,
    memberships: Dict[str, float],
    weights: Dict[str, float],
    realized_vol: float,
    low_vol: float,
    high_vol: float,
    risk_fraction: float = 0.02
) -> int:
    """
    Full sizing pipeline (Stage 7).
    """
    # Step 1: Hard ceiling
    q0 = compute_base_quantity(
        equity=equity,
        max_loss_per_contract=max_loss_per_contract,
        risk_fraction=risk_fraction
    )

    if q0 == 0:
        return 0

    # Step 2: Fuzzy confidence
    Ft = compute_fuzzy_confidence(
        memberships=memberships,
        weights=weights
    )

    # Step 3: Volatility penalty
    sigma_star = normalize_volatility(
        realized_vol=realized_vol,
        low_vol=low_vol,
        high_vol=high_vol
    )

    # Step 4: Scaling factor
    g = compute_scaling_factor(
        confidence=Ft,
        volatility_penalty=sigma_star
    )

    # Step 5: Final size
    q = int(q0 * g)
    
    # DEBUG LOG
    # print(f"[FuzzyDebug] Eq={equity:.0f} Risk={equity*risk_fraction:.0f} Loss={max_loss_per_contract:.0f} -> q0={q0} | Ft={Ft:.2f} Vol={realized_vol:.1f} Sig={sigma_star:.2f} -> g={g:.2f} => Final q={q}")

    return max(q, 0)

# =============================================================================
# Helper Functions for Membership Calculation
# =============================================================================

def calculate_mtf_membership(mtf_snapshot) -> float:
    """
    Calculates MTF Alignment score (0.0 to 1.0).
    Replaces old get_fuzzy_consensus.
    """
    if not mtf_snapshot:
        return 0.5
    
    # Weighting: Daily (50%), Hourly (35%), 5-Min (15%)
    weights = {'D': 0.50, '60': 0.35, '5': 0.15}
    total_score = 0
    total_weight = 0

    for tf, weight in weights.items():
        data = mtf_snapshot.get(tf)
        if data:
            # 1.0 = Bullish (Close > Open), 0.0 = Bearish
            # For Iron Condor, we want NEUTRAL (0.5).
            # But 'Membership' implies quality.
            # If we want alignment with NEUTRALITY:
            #   If Close ~ Open, score = 1.0 (Good for IC)
            #   If Trending, score = 0.0 (Bad for IC)
            
            # Let's align this with "MTF Alignment" where 1.0 = Favorable.
            # Favorable for IC = Neutral price action.
            
            close_price = data['close']
            open_price = data['open']
            
            # Absolute % change
            pct_change = abs(close_price - open_price) / open_price
            
            # Membership Function:
            # 0% change -> 1.0
            # >1% change -> 0.0
            score = max(0.0, 1.0 - (pct_change * 100.0))
            
            total_score += (score * weight)
            total_weight += weight

    if total_weight == 0:
        return 0.5
        
    return round(total_score / total_weight, 2)

def calculate_iv_membership(ivr: float) -> float:
    """
    Map IV Rank to favorability (0.0 to 1.0).
    Higher IVR is better for Short IC (selling premium).
    """
    # 0 IVR -> 0.0
    # 50 IVR -> 0.8
    # 100 IVR -> 1.0
    # Simple linear clip
    return min(1.0, ivr / 60.0)

def calculate_regime_membership(vix: float, threshold: float = 20.0) -> float:
    """
    Map VIX to regime stability.
    Lower VIX = Stable = 1.0
    High VIX = Unstable = 0.0
    """
    # If VIX <= 12 -> 1.0
    # If VIX >= threshold + 10 -> 0.0
    if vix <= 12.0: return 1.0
    
    # Linear decay
    # Range: 12 to 30 (approx 18 pts)
    # penalty = (vix - 12) / 18
    penalty = (vix - 12.0) / 18.0
    return max(0.0, 1.0 - penalty)


# Backward Compatibility
get_fuzzy_consensus = calculate_mtf_membership

