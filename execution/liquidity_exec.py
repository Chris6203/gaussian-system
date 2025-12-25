"""
Liquidity-Aware Execution Engine

Sophisticated execution layer that:
- Screens contracts by OI/volume/spread/delta
- Calculates liquidity scores
- Sizes orders by queue depth
- Works midpoint-pegged orders with price ladder
- Uses IOC probes for price discovery
- Falls back to vertical spreads if single-leg execution is poor
"""

from dataclasses import dataclass
from typing import Optional, Dict, Any, List, Tuple
import math
import time
import logging

log = logging.getLogger(__name__)


@dataclass
class LiquidityRules:
    """Configurable rules for liquidity screening and execution"""
    min_oi: int = 75
    min_day_volume: int = 100
    max_spread_pct: float = 5.0
    target_delta: float = 0.30
    delta_band: float = 0.08
    max_notional_pct_adv: float = 0.05
    min_quote_size: int = 10
    max_work_time_sec: int = 20
    price_tick: float = 0.01
    max_price_lifts: int = 4
    ioc_probe_size: int = 1
    cancel_replace_delay: float = 2.0


@dataclass
class OrderIntent:
    """Order intent with liquidity constraints"""
    symbol: str
    side: str            # 'buy_to_open' | 'sell_to_open' | ...
    qty: int
    expiry: str          # 'YYYY-MM-DD'
    call_put: str        # 'call' | 'put'
    bias: str            # 'bull' | 'bear' | 'neutral'
    max_debit: Optional[float] = None
    min_credit: Optional[float] = None


class LiquidityExecutor:
    """
    Liquidity-aware execution engine
    
    Broker shim expects a tradier_api adapter with:
      get_option_chain(symbol, expiry) -> DataFrame-like rows:
        {symbol, occ_symbol, strike, type, bid, ask, bidsize, asksize, 
         open_interest, volume, delta, adv_notional?}
      get_quote(occ) -> {bid, ask, bidsize, asksize}
      place_limit(occ, side, qty, limit_price, tif) -> order_id
      modify_order(order_id, limit_price) -> None
      cancel_order(order_id) -> None
      place_spread_limit(legs, side, qty, net_price, tif) -> order_id
      filled_qty(order_id) -> int
    """
    
    def __init__(self, tradier_api, rules: LiquidityRules):
        self.api = tradier_api
        self.rules = rules
        log.info(f"[LIQ-EXEC] Initialized with rules: OI>={rules.min_oi}, Vol>={rules.min_day_volume}, Spread<={rules.max_spread_pct}%")
    
    # ---------- helpers ----------
    
    def _spread_pct(self, bid: float, ask: float) -> float:
        """Calculate bid-ask spread as percentage of mid"""
        if bid <= 0 or ask <= 0:
            return 999.0
        mid = (bid + ask) / 2.0
        return ((ask - bid) / mid) * 100.0 if mid > 0 else 999.0
    
    def _liquidity_score(self, row) -> float:
        """
        Calculate liquidity score (0-1) based on:
        - Spread tightness (50%)
        - Market participation (30% - OI + volume)
        - Quote depth (20%)
        """
        bid, ask = float(row.bid), float(row.ask)
        sp = self._spread_pct(bid, ask)
        oi = float(row.open_interest or 0)
        vol = float(row.volume or 0)
        bsz = int(row.bidsize or 0)
        asz = int(row.asksize or 0)
        
        # Spread component (50%)
        tight = max(0.0, 1.0 - sp / self.rules.max_spread_pct)
        
        # Quote depth component (20%)
        depth = min(1.0, (min(bsz, asz) / max(self.rules.min_quote_size, 1)))
        
        # Participation component (30% - weighted OI + volume)
        participation = (min(1.0, (oi / self.rules.min_oi)) * 0.6 + 
                        min(1.0, (vol / self.rules.min_day_volume)) * 0.4)
        
        score = 0.50 * tight + 0.30 * participation + 0.20 * depth
        return score
    
    # ---------- contract selection ----------
    
    def pick_contract(self, symbol: str, expiry: str, call_put: str, 
                     target_delta: float) -> Optional[Dict[str, Any]]:
        """
        Screen options chain and select best contract by liquidity
        
        Returns:
            Best contract dict or None if no suitable contract found
        """
        log.info(f"[LIQ-EXEC] Screening {symbol} {call_put}s for {expiry}, target delta={target_delta}")
        
        chain = self.api.get_option_chain(symbol, expiry)
        if chain is None or len(chain) == 0:
            log.warning(f"[LIQ-EXEC] No chain data for {symbol} {expiry}")
            return None
        
        # Filter by option type
        side = chain[(chain['type'] == call_put)]
        if side.empty:
            log.warning(f"[LIQ-EXEC] No {call_put} options in chain")
            return None
        
        # Apply liquidity filters
        keep = side[
            (side['open_interest'] >= self.rules.min_oi) &
            (side['volume'] >= self.rules.min_day_volume) &
            (side['bidsize'] >= self.rules.min_quote_size) &
            (side['asksize'] >= self.rules.min_quote_size) &
            (side['bid'] > 0) & (side['ask'] > 0)
        ].copy()
        
        if keep.empty:
            log.warning(f"[LIQ-EXEC] No contracts passed liquidity filters")
            return None
        
        log.info(f"[LIQ-EXEC] {len(keep)} contracts passed liquidity filters")
        
        # Calculate delta diff and spread
        keep['delta_diff'] = (keep['delta'].abs() - abs(target_delta)).abs()
        keep['spread_pct'] = ((keep['ask'] - keep['bid']) / ((keep['ask'] + keep['bid'])/2.0)) * 100.0
        
        # Filter by delta band and max spread
        keep = keep[
            (keep['delta_diff'] <= self.rules.delta_band) &
            (keep['spread_pct'] <= self.rules.max_spread_pct)
        ]
        
        if keep.empty:
            log.warning(f"[LIQ-EXEC] No contracts in delta band or spread too wide")
            return None
        
        # Score remaining contracts
        keep['liq_score'] = keep.apply(self._liquidity_score, axis=1)
        
        # Select best by liquidity score, then volume, then OI
        best = keep.sort_values(
            ['liq_score', 'volume', 'open_interest'], 
            ascending=[False, False, False]
        ).iloc[0]
        
        log.info(f"[LIQ-EXEC] Selected: Strike=${best['strike']:.2f}, "
                f"Delta={best['delta']:.3f}, Score={best['liq_score']:.2f}, "
                f"Spread={best['spread_pct']:.1f}%")
        
        return best.to_dict()
    
    # ---------- sizing ----------
    
    def compute_qty(self, occ_symbol: str, limit_price: float, desired_qty: int, 
                   adv_notional: Optional[float] = None) -> int:
        """
        Compute order size based on displayed liquidity and ADV constraints
        
        Args:
            occ_symbol: Option symbol
            limit_price: Estimated fill price
            desired_qty: Desired quantity
            adv_notional: Average daily volume in $ (optional cap)
        
        Returns:
            Adjusted quantity
        """
        q = self.api.get_quote(occ_symbol)
        bsz = int(q.get('bidsize', 0))
        asz = int(q.get('asksize', 0))
        
        # Cap by displayed depth
        depth_cap = max(1, min(bsz, asz))
        size = min(desired_qty, depth_cap)
        
        # Cap by ADV if provided
        if adv_notional:
            notional = limit_price * 100.0 * size  # $100 per contract
            cap = self.rules.max_notional_pct_adv * adv_notional
            if notional > cap:
                size = max(1, math.floor(cap / (limit_price * 100.0)))
        
        if size != desired_qty:
            log.info(f"[LIQ-EXEC] Sized down: {desired_qty} → {size} contracts")
        
        return max(1, size)
    
    # ---------- mid-peg with ladder ----------
    
    def work_limit(self, occ_symbol: str, side: str, qty: int) -> Tuple[int, Optional[str]]:
        """
        Work limit order with midpoint peg and price ladder
        
        Strategy:
          1. IOC probe at mid to test liquidity
          2. Place limit at mid
          3. If no fill, walk price ladder toward market
          4. Cancel after max time or lifts
        
        Returns:
            (filled_qty, order_id)
        """
        log.info(f"[LIQ-EXEC] Working {side} {qty} {occ_symbol}")
        
        q = self.api.get_quote(occ_symbol)
        bid, ask = float(q['bid']), float(q['ask'])
        
        if bid <= 0 or ask <= 0:
            log.error(f"[LIQ-EXEC] Invalid quote: bid={bid}, ask={ask}")
            return 0, None
        
        mid = round((bid + ask) / 2.0, 2)
        improve = self.rules.price_tick
        
        filled_total = 0
        order_id = None
        
        try:
            # IOC probe for price discovery
            if self.rules.ioc_probe_size and qty > 1:
                log.info(f"[LIQ-EXEC] IOC probe: {self.rules.ioc_probe_size} @ ${mid:.2f}")
                probe_id = self.api.place_limit(
                    occ_symbol, side, self.rules.ioc_probe_size, mid, tif='ioc'
                )
                time.sleep(0.5)
                filled_total += self.api.filled_qty(probe_id)
                if filled_total > 0:
                    log.info(f"[LIQ-EXEC] IOC probe filled: {filled_total}")
            
            remaining = max(0, qty - filled_total)
            if remaining == 0:
                return filled_total, None
            
            # Work order with price ladder
            work_px = mid
            log.info(f"[LIQ-EXEC] Placing limit: {remaining} @ ${work_px:.2f}")
            order_id = self.api.place_limit(occ_symbol, side, remaining, work_px, tif='day')
            
            start = time.time()
            lifts = 0
            
            while time.time() - start < self.rules.max_work_time_sec:
                time.sleep(self.rules.cancel_replace_delay)
                
                filled = self.api.filled_qty(order_id)
                if filled >= remaining:
                    filled_total += filled
                    log.info(f"[LIQ-EXEC] ✅ Filled: {filled_total} total")
                    return filled_total, order_id
                
                lifts += 1
                if lifts > self.rules.max_price_lifts:
                    log.info(f"[LIQ-EXEC] Max price lifts reached ({lifts})")
                    break
                
                # Update quote and adjust price
                q = self.api.get_quote(occ_symbol)
                bid, ask = float(q['bid']), float(q['ask'])
                
                if bid <= 0 or ask <= 0:
                    break
                
                # Walk price ladder
                if 'buy' in side:
                    work_px = min(ask, round(work_px + improve, 2))
                else:
                    work_px = max(bid, round(work_px - improve, 2))
                
                log.info(f"[LIQ-EXEC] Price lift #{lifts}: ${work_px:.2f}")
                self.api.modify_order(order_id, work_px)
            
            filled_total += self.api.filled_qty(order_id)
            log.info(f"[LIQ-EXEC] Final fill: {filled_total}/{qty}")
            return filled_total, order_id
            
        finally:
            if order_id:
                try:
                    self.api.cancel_order(order_id)
                except Exception as e:
                    log.warning(f"[LIQ-EXEC] Cancel failed: {e}")
    
    # ---------- fallback to tighter vertical ----------
    
    def _build_vertical(self, intent: OrderIntent, pick_primary: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Build vertical spread as fallback if single-leg has poor execution
        
        Returns:
            Spread definition dict or None
        """
        log.info(f"[LIQ-EXEC] Building vertical spread fallback")
        
        chain = self.api.get_option_chain(intent.symbol, intent.expiry)
        same_side = chain[(chain['type'] == intent.call_put)].copy()
        
        if same_side.empty:
            return None
        
        k0 = float(pick_primary['strike'])
        
        # Find hedge leg (next strike out)
        if intent.call_put == 'call':
            candidates = same_side[same_side['strike'] > k0].nsmallest(3, 'strike')
        else:
            candidates = same_side[same_side['strike'] < k0].nlargest(3, 'strike')
        
        def ok(r):
            return (r['open_interest'] >= self.rules.min_oi and
                    r['volume'] >= self.rules.min_day_volume and
                    self._spread_pct(float(r['bid']), float(r['ask'])) <= self.rules.max_spread_pct)
        
        hedge = None
        for _, r in candidates.iterrows():
            if ok(r):
                hedge = r
                break
        
        if hedge is None:
            log.warning(f"[LIQ-EXEC] No suitable hedge leg found")
            return None
        
        # Calculate net credit/debit
        a_bid, a_ask = float(pick_primary['bid']), float(pick_primary['ask'])
        h_bid, h_ask = float(hedge['bid']), float(hedge['ask'])
        
        long_mid = (a_bid + a_ask) / 2.0
        short_mid = (h_bid + h_ask) / 2.0
        
        if 'buy' in intent.side:
            net_mid = long_mid - short_mid
            legs = [
                {"occ": pick_primary.get('occ_symbol', pick_primary['symbol']), 
                 "side": intent.side, "ratio": 1},
                {"occ": hedge.get('occ_symbol', hedge['symbol']), 
                 "side": "sell_to_open", "ratio": 1},
            ]
            spread_side = 'debit'
        else:
            net_mid = short_mid - long_mid
            legs = [
                {"occ": pick_primary.get('occ_symbol', pick_primary['symbol']), 
                 "side": intent.side, "ratio": 1},
                {"occ": hedge.get('occ_symbol', hedge['symbol']), 
                 "side": "buy_to_open", "ratio": 1},
            ]
            spread_side = 'credit'
        
        log.info(f"[LIQ-EXEC] Vertical: ${k0:.2f}/${float(hedge['strike']):.2f}, "
                f"Net ${spread_side}: ${net_mid:.2f}")
        
        return {"legs": legs, "net_mid": net_mid, "side": spread_side}
    
    def execute_with_fallback(self, intent: OrderIntent) -> Dict[str, Any]:
        """
        Execute order with automatic fallback to vertical spread
        
        Flow:
          1. Screen chain and pick best contract
          2. Check price constraints
          3. Size order by liquidity
          4. Work limit order with price ladder
          5. If poor fill, build and submit vertical spread
        
        Returns:
            Execution result dict with status
        """
        log.info(f"[LIQ-EXEC] Executing {intent.side} {intent.qty} {intent.symbol} "
                f"{intent.call_put} exp={intent.expiry}")
        
        # Select contract
        pick = self.pick_contract(
            intent.symbol, intent.expiry, intent.call_put, 
            target_delta=self.rules.target_delta
        )
        
        if not pick:
            return {"status": "no_contract", "reason": "no liquid contract matched"}
        
        occ = pick.get('occ_symbol') or pick.get('symbol')
        if not occ:
            return {"status": "no_symbol", "reason": "missing OCC symbol"}
        
        bid, ask = float(pick['bid']), float(pick['ask'])
        
        # Check price constraints
        if 'buy' in intent.side and intent.max_debit is not None and ask > intent.max_debit:
            log.warning(f"[LIQ-EXEC] Rejected: ask ${ask:.2f} > max_debit ${intent.max_debit:.2f}")
            return {"status": "rejected", "reason": "ask above max_debit"}
        
        if 'sell' in intent.side and intent.min_credit is not None and bid < intent.min_credit:
            log.warning(f"[LIQ-EXEC] Rejected: bid ${bid:.2f} < min_credit ${intent.min_credit:.2f}")
            return {"status": "rejected", "reason": "bid below min_credit"}
        
        # Size and execute
        mid = round((bid + ask) / 2.0, 2)
        qty = self.compute_qty(occ, mid, intent.qty, adv_notional=pick.get('adv_notional'))
        
        filled, _ = self.work_limit(occ, intent.side, qty)
        
        if filled > 0:
            log.info(f"[LIQ-EXEC] ✅ Single-leg filled: {filled}/{qty}")
            return {
                "status": "filled_partial" if filled < qty else "filled",
                "legs": [{"occ": occ, "side": intent.side, "qty": filled, "px": mid}],
                "contract": pick,
                "filled_qty": filled
            }
        
        # Fallback to vertical spread
        log.info(f"[LIQ-EXEC] Single-leg failed, trying vertical spread")
        vertical = self._build_vertical(intent, pick)
        
        if vertical:
            net_mid = round(vertical['net_mid'], 2)
            order_id = self.api.place_spread_limit(
                vertical['legs'], vertical['side'], qty, net_mid, tif='day'
            )
            log.info(f"[LIQ-EXEC] ✅ Vertical spread submitted: order#{order_id}")
            return {
                "status": "spread_submitted",
                "net_mid": net_mid,
                "legs": vertical['legs'],
                "contract": pick,
                "order_id": order_id
            }
        
        log.error(f"[LIQ-EXEC] ❌ Could not execute: single-leg failed, spread build failed")
        return {"status": "no_fill", "reason": "could not work single-leg; spread build failed"}


