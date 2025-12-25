from datetime import datetime, timedelta
from typing import Tuple


def get_market_status() -> Tuple[bool, str]:
    """US equities market hours (approx) 9:30-16:00 ET."""
    try:
        import pytz

        et = pytz.timezone("US/Eastern")
        now = datetime.now(et)

        if now.weekday() >= 5:
            return False, f"Weekend ({now.strftime('%A')})"

        market_open = now.replace(hour=9, minute=30, second=0, microsecond=0)
        market_close = now.replace(hour=16, minute=0, second=0, microsecond=0)

        if now < market_open:
            minutes_until = int((market_open - now).total_seconds() // 60)
            return False, f"Pre-market ({minutes_until} min until open)"
        if now > market_close:
            return False, "After-hours"

        minutes_left = int((market_close - now).total_seconds() // 60)
        return True, f"Open ({minutes_left} min remaining)"

    except Exception:
        # Fallback: rough UTC-5
        now = datetime.utcnow() - timedelta(hours=5)
        if now.weekday() >= 5:
            return False, "Weekend"
        if now.hour < 9 or now.hour > 16:
            return False, "Closed"
        if now.hour == 9 and now.minute < 30:
            return False, "Pre-market"
        if now.hour == 16 and now.minute > 0:
            return False, "After-hours"
        return True, "Open"
