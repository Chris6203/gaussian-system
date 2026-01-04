# intelligence/regime_filter.py

def check_liquidity_gate(data, spread_estimate=1.0):
    """
    Checks if the recent price action suggests enough 
    liquidity to fill an Iron Condor.
    """
    try:
        # Check volume of the last 5 bars
        recent_volume = data.volume.get(size=5)
        avg_vol = sum(recent_volume) / len(recent_volume)
        
        # Rule: Minimum volume threshold (adjust based on symbol)
        if avg_vol < 100:
            return False
            
        # Check volatility (True Range)
        # If the range is too massive, spreads usually widen
        highs = data.high.get(size=5)
        lows = data.low.get(size=5)
        avg_range = sum([h - l for h, l in zip(highs, lows)]) / 5
        
        if avg_range > (data.close[0] * 0.02): # 2% swing in 5 mins is too wild
            return False

        return True
    except Exception:
        # If we don't have enough bars yet, skip trading
        return False