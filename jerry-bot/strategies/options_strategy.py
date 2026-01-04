# strategies/options_strategy.py - Iron Condor with MTF Intelligence
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict
import datetime as dt
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from intelligence.fuzzy_engine import get_fuzzy_consensus
from intelligence.regime_filter import check_liquidity_gate

# === Core Types ===
@dataclass
class OptionQuote:
    symbol: str
    expiration: dt.date
    strike: float
    is_call: bool
    bid: float
    ask: float
    mid: float
    delta: float
    iv: float

@dataclass
class IronCondorLegs:
    short_call: OptionQuote
    long_call: OptionQuote
    short_put: OptionQuote
    long_put: OptionQuote

@dataclass
class PositionState:
    id: str
    legs: IronCondorLegs
    open_time: dt.datetime
    credit_received: float
    quantity: int
    adjustments_done: Dict[str, int]
    mtf_consensus: float = 0.5  # Store entry consensus

# === Pricing Utilities ===
def calc_spread_credit(short: OptionQuote, long: OptionQuote) -> float:
    return max(0.0, short.mid - long.mid)

def calc_condor_credit(legs: IronCondorLegs) -> float:
    call_credit = calc_spread_credit(legs.short_call, legs.long_call)
    put_credit = calc_spread_credit(legs.short_put, legs.long_put)
    return call_credit + put_credit

def max_loss(wing_width: float, total_credit: float) -> float:
    return max(0.0, wing_width - total_credit)

# === Strike Selection ===
def within_delta_band(opt: OptionQuote, target_low: float, target_high: float) -> bool:
    d = abs(opt.delta)
    return target_low <= d <= target_high

def pick_expiration(expirations: List[dt.date], cfg, today: dt.date) -> Optional[dt.date]:
    candidates = [e for e in expirations if cfg.dte_min <= (e - today).days <= cfg.dte_max]
    return sorted(candidates)[0] if candidates else None

def nearest_by_delta(quotes: List[OptionQuote], is_call: bool, 
                     target_low: float, target_high: float) -> Optional[OptionQuote]:
    candidates = [q for q in quotes if q.is_call == is_call 
                  and within_delta_band(q, target_low, target_high)]
    if candidates:
        center = (target_low + target_high) / 2.0
        return sorted(candidates, key=lambda q: abs(abs(q.delta) - center))[0]
    return None

def pick_long_by_width(chain: List[OptionQuote], short_leg: OptionQuote, 
                       is_call: bool, wing_width: float) -> Optional[OptionQuote]:
    target_strike = short_leg.strike + wing_width if is_call else short_leg.strike - wing_width
    candidates = [q for q in chain if q.is_call == is_call 
                  and q.expiration == short_leg.expiration]
    if not candidates:
        return None
    return sorted(candidates, key=lambda q: abs(q.strike - target_strike))[0]

# === Wing Width Determination ===
def regime_wing_width(cfg, iv_rank: float, vix: float) -> float:
    """Dynamic wing width based on volatility regime"""
    base = cfg.wing_width_min
    high_vol = (iv_rank >= cfg.regime_iv_rank_widen) or (vix >= cfg.regime_vix_widen)
    if high_vol:
        return min(cfg.wing_width_max, base + cfg.width_widen_increment)
    return base

def optimize_wing_width(cfg, historical_credit_curve: Dict[float, float]) -> float:
    """Select optimal width from historical performance"""
    feasible = [(w, r) for w, r in historical_credit_curve.items() 
                if r >= cfg.min_credit_to_width 
                and cfg.wing_width_min <= w <= cfg.wing_width_max]
    if not feasible:
        return cfg.wing_width_min
    feasible.sort(key=lambda t: (-t[1], t[0]))  # Best ratio, then smallest width
    return feasible[0][0]

# === Build Condor ===
def build_condor(chain: List[OptionQuote], cfg, today: dt.date, 
                 spot: float, base_wing_width: float) -> Optional[IronCondorLegs]:
    """Construct iron condor from option chain"""
    short_call = nearest_by_delta(chain, True, cfg.target_short_delta_low, cfg.target_short_delta_high)
    short_put = nearest_by_delta(chain, False, cfg.target_short_delta_low, cfg.target_short_delta_high)
    
    if not short_call or not short_put:
        return None
    
    long_call = pick_long_by_width(chain, short_call, True, base_wing_width)
    long_put = pick_long_by_width(chain, short_put, False, base_wing_width)
    
    if not long_call or not long_put:
        return None
    
    return IronCondorLegs(
        short_call=short_call,
        long_call=long_call,
        short_put=short_put,
        long_put=long_put
    )

# === Entry Filters ===
def can_enter_trade(broker, cfg) -> Tuple[bool, str]:
    """Check if we can open a new position"""
    ivr = broker.get_iv_rank(cfg.underlying)
    vix = broker.get_vix()
    
    if ivr < cfg.iv_rank_min:
        return False, f"Skip: IV Rank {ivr:.1f} < {cfg.iv_rank_min}"
    
    if vix > cfg.vix_threshold:
        return False, f"Skip: VIX {vix:.1f} > {cfg.vix_threshold}"
    
    acct = broker.get_account_metrics()
    positions = broker.get_open_positions(cfg.underlying)
    
    if len(positions) >= cfg.max_positions:
        return False, "Skip: max positions reached"
    
    positions_value = acct.get("positions_value", 0.0)
    equity = max(acct.get("equity", 1.0), 1.0)
    
    if positions_value / equity >= cfg.max_portfolio_alloc:
        return False, "Skip: portfolio allocation limit"
    
    return True, "OK"

def check_mtf_alignment(cfg, mtf_snapshot) -> Tuple[bool, str, float]:
    """Check if MTF consensus allows entry"""
    if not cfg.use_mtf_filter or mtf_snapshot is None:
        return True, "MTF filter disabled", 0.5
    
    consensus = get_fuzzy_consensus(mtf_snapshot)
    
    # Iron condors profit from range-bound markets (neutral consensus)
    if consensus < cfg.mtf_consensus_min or consensus > cfg.mtf_consensus_max:
        return False, f"Skip: MTF consensus {consensus:.2f} outside range [{cfg.mtf_consensus_min}, {cfg.mtf_consensus_max}]", consensus
    
    return True, f"MTF consensus {consensus:.2f} favorable for range", consensus

def validate_pricing_and_risk(legs: IronCondorLegs, width: float, 
                              cfg, acct_equity: float) -> Tuple[bool, str, float]:
    """Validate credit and risk constraints"""
    credit = calc_condor_credit(legs)
    credit_ratio = credit / width
    
    if credit_ratio < cfg.min_credit_to_width:
        return False, f"Reject: credit ratio {credit_ratio:.2f} < {cfg.min_credit_to_width}", credit
    
    mloss = max_loss(width, credit)
    
    if mloss > cfg.max_account_risk_per_trade * acct_equity:
        return False, f"Reject: max loss ${mloss:.2f} exceeds per-trade risk", credit
    
    return True, "OK", credit

# === Main Entry Function ===
def select_and_enter_condor(broker, cfg, today: dt.date,
                            historical_credit_curve: Optional[Dict[float, float]] = None,
                            quantity: int = 1,
                            mtf_snapshot=None,
                            liquidity_data=None):
    """
    Select and enter iron condor with MTF filtering.
    
    Returns: PositionState if entered, None otherwise
    """
    # 1. Check Portfolio Limits
    ok, msg = can_enter_trade(broker, cfg)
    if not ok:
        print(f"[Entry Blocked] {msg}")
        return None
    
    # 2. MTF Alignment Check
    mtf_ok, mtf_msg, consensus = check_mtf_alignment(cfg, mtf_snapshot)
    if not mtf_ok:
        print(f"[Entry Blocked] {mtf_msg}")
        return None
    
    # 3. Liquidity Gate
    if cfg.use_liquidity_gate and liquidity_data is not None:
        if not check_liquidity_gate(liquidity_data, spread_estimate=1.0):
            print(f"[Entry Blocked] Failed liquidity gate")
            return None
    
    # 4. Get Market Context
    spot = broker.get_spot(cfg.underlying)
    ivr = broker.get_iv_rank(cfg.underlying)
    vix = broker.get_vix()
    
    # 5. Determine Wing Width
    target_width = regime_wing_width(cfg, ivr, vix)
    
    if historical_credit_curve:
        optimized_width = optimize_wing_width(cfg, historical_credit_curve)
        target_width = max(target_width, optimized_width)  # Use wider of the two
    
    target_width = min(target_width, cfg.wing_width_max)
    
    print(f"[Entry] VIX: {vix:.2f}, IVR: {ivr:.2f}, MTF: {consensus:.2f} -> Wing Width: ${target_width}")
    
    # 6. Expiration Selection
    expirations = broker.get_expirations(cfg.underlying)
    exp = pick_expiration(expirations, cfg, today)
    if not exp:
        print("[Entry Blocked] No valid expiration in DTE window")
        return None
    
    # 7. Build Legs
    chain = broker.get_option_chain(cfg.underlying, exp)
    legs = build_condor(chain, cfg, today, spot, target_width)
    if not legs:
        print("[Entry Blocked] Unable to build condor legs")
        return None
    
    # 8. Validate Pricing & Risk
    acct = broker.get_account_metrics()
    equity = acct.get("equity", 0.0)
    valid, vmsg, credit = validate_pricing_and_risk(legs, target_width, cfg, equity)
    
    if not valid:
        print(f"[Entry Blocked] {vmsg}")
        return None
    
    # 9. Execute
    print(f"[Entering] {cfg.underlying} Iron Condor | Width: ${target_width} | Credit: ${credit:.2f} | MTF: {consensus:.2f}")
    order_id = broker.place_iron_condor(legs, quantity=quantity, limit_price=credit)
    
    if order_id:
        positions = broker.get_open_positions(cfg.underlying)
        if positions:
            # Store MTF consensus with position
            positions[-1].mtf_consensus = consensus
            return positions[-1]
    
    return None

# === Position Management ===
def estimate_pnl(position: PositionState, current_quotes: IronCondorLegs) -> Tuple[float, float]:
    """Calculate unrealized P&L"""
    debit_call = max(0.0, current_quotes.long_call.mid - current_quotes.short_call.mid)
    debit_put = max(0.0, current_quotes.long_put.mid - current_quotes.short_put.mid)
    debit_total = debit_call + debit_put
    
    unreal = (position.credit_received - debit_total) * position.quantity
    max_prof = position.credit_received * position.quantity
    
    return unreal, max_prof

def breached_short_strike(spot: float, legs: IronCondorLegs) -> Optional[str]:
    """Check if price breached short strikes"""
    if spot >= legs.short_call.strike:
        return "call"
    if spot <= legs.short_put.strike:
        return "put"
    return None

def net_position_delta(legs: IronCondorLegs, quantity: int) -> float:
    """Calculate net portfolio delta"""
    net = (-legs.short_call.delta + legs.long_call.delta + 
           legs.short_put.delta - legs.long_put.delta) * quantity
    return net

def maybe_dynamic_delta_hedge(broker, cfg, pos: PositionState, current: IronCondorLegs) -> None:
    """Hedge portfolio delta if threshold exceeded"""
    nd = net_position_delta(current, pos.quantity)
    
    if abs(nd) >= cfg.dynamic_delta_hedge_threshold:
        units = int(round((abs(nd) / 0.01) * cfg.dynamic_delta_hedge_unit))
        if units > 0:
            side = "sell" if nd > 0 else "buy"
            broker.trade_shares(cfg.underlying, quantity=units, side=side)
            print(f"[Delta Hedge] {side.upper()} {units} shares (net delta: {nd:.3f})")

def manage_positions(broker, cfg, today: dt.date) -> None:
    """Manage open positions: exits, rolls, hedges"""
    positions = broker.get_open_positions(cfg.underlying)
    spot = broker.get_spot(cfg.underlying)
    
    for pos in positions:
        chain = broker.get_option_chain(cfg.underlying, pos.legs.short_call.expiration)
        
        # Match current quotes
        def match(q: OptionQuote) -> OptionQuote:
            matches = [x for x in chain if x.strike == q.strike and x.is_call == q.is_call]
            return matches[0] if matches else q
        
        current = IronCondorLegs(
            short_call=match(pos.legs.short_call),
            long_call=match(pos.legs.long_call),
            short_put=match(pos.legs.short_put),
            long_put=match(pos.legs.long_put)
        )
        
        # Calculate P&L
        unreal, max_prof = estimate_pnl(pos, current)
        
        # Exit Rule 1: Profit Target
        if unreal >= cfg.profit_take_pct * max_prof:
            print(f"[Exit] Profit target reached: ${unreal:.2f} (MTF entry: {pos.mtf_consensus:.2f})")
            broker.close_position(pos.id)
            continue
        
        # Exit Rule 2: DTE Threshold
        dte = (pos.legs.short_call.expiration - today).days
        if dte <= cfg.max_hold_days:
            print(f"[Exit] DTE threshold: {dte} days remaining")
            broker.close_position(pos.id)
            continue
        
        # Exit Rule 3: Stop Loss
        loss_threshold = -cfg.loss_close_multiple * pos.credit_received * pos.quantity
        if unreal <= loss_threshold:
            print(f"[Exit] Stop loss triggered: ${unreal:.2f}")
            broker.close_position(pos.id)
            continue
        
        # Adjustment: Roll Breached Side
        breach = breached_short_strike(spot, pos.legs)
        if breach and pos.adjustments_done[breach] == 0 and cfg.allow_one_adjustment_per_side:
            expirations = broker.get_expirations(cfg.underlying)
            later_exps = [e for e in expirations if e > pos.legs.short_call.expiration]
            
            if not later_exps:
                continue
            
            new_exp = later_exps[0]
            new_chain = broker.get_option_chain(cfg.underlying, new_exp)
            
            width_call = pos.legs.long_call.strike - pos.legs.short_call.strike
            width_put = pos.legs.short_put.strike - pos.legs.long_put.strike
            
            # Find new strikes
            def nearest_new(is_call: bool):
                center = (cfg.target_short_delta_low + cfg.target_short_delta_high) / 2.0
                candidates = [q for q in new_chain if q.is_call == is_call]
                if not candidates:
                    return None
                return sorted(candidates, key=lambda q: abs(abs(q.delta) - center))[0]
            
            if breach == "call":
                new_short = nearest_new(True)
                new_long = pick_long_by_width(new_chain, new_short, True, width_call) if new_short else None
            else:
                new_short = nearest_new(False)
                new_long = pick_long_by_width(new_chain, new_short, False, width_put) if new_short else None
            
            if new_short and new_long:
                addl_credit = max(0.0, new_short.mid - new_long.mid)
                print(f"[Roll] {breach.upper()} side breached, rolling for ${addl_credit:.2f} credit")
                broker.roll_vertical(pos.id, breach, new_short, new_long, limit_price=addl_credit)
        
        # Dynamic Delta Hedging
        maybe_dynamic_delta_hedge(broker, cfg, pos, current)

# === Legacy Interface ===
def run_strategy(broker, cfg, historical_credit_curve=None, quantity=1):
    """Main strategy runner for live mode"""
    today = dt.date.today()
    pos = select_and_enter_condor(broker, cfg, today, historical_credit_curve, quantity)
    if pos:
        print(f"[Entered] Position {pos.id} with credit ${pos.credit_received:.2f}")
    manage_positions(broker, cfg, today)