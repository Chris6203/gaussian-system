# core/liquidity_gate.py

def liquidity_ok(
    short_leg,
    long_leg,
    credit: float,
    width: float,
    mode: str,
    config
) -> bool:
    """
    Mode-safe liquidity gate.
    - ALWAYS passes backtests
    - FAILS OPEN on missing data
    - Uses spread % instead of volume/OI
    """

    # ------------------------------------------------------------------
    # 1. Backtests are NEVER liquidity-gated
    # ------------------------------------------------------------------
    if mode == "backtest":
        return True

    # ------------------------------------------------------------------
    # 2. Helper: bid/ask spread percentage
    # ------------------------------------------------------------------
    def spread_pct(opt):
        if opt is None:
            return None
        if opt.bid is None or opt.ask is None:
            return None
        mid = (opt.bid + opt.ask) / 2.0
        if mid <= 0:
            return None
        return (opt.ask - opt.bid) / mid

    short_spread = spread_pct(short_leg)
    long_spread  = spread_pct(long_leg)

    # ------------------------------------------------------------------
    # 3. Fail-open on incomplete data (LIVE SAFETY)
    # ------------------------------------------------------------------
    if short_spread is None or long_spread is None:
        return True

    # ------------------------------------------------------------------
    # 4. Spread sanity checks (LIVE ONLY)
    # ------------------------------------------------------------------
    if short_spread > config.max_short_spread_pct:
        return False

    if long_spread > config.max_long_spread_pct:
        return False

    # ------------------------------------------------------------------
    # 5. Credit efficiency (already core to IC logic)
    # ------------------------------------------------------------------
    if width <= 0:
        return False

    if (credit / width) < config.min_credit_to_width:
        return False

    return True
