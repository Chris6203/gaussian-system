from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import requests


@dataclass
class TradierCreds:
    token: str


class DataFetcher:
    def __init__(self, tradier_token: Optional[str], logger):
        self.tradier_token = tradier_token
        self.logger = logger

    def get_price_data(self, symbol: str, period: str = "1d", interval: str = "1m") -> Optional[Dict[str, Any]]:
        data = self._fetch_yfinance(symbol, period, interval)
        if data is not None:
            return data
        data = self._fetch_tradier(symbol)
        if data is not None:
            return data
        self.logger.warning(f"[DATA] Could not fetch price for {symbol}")
        return None

    def _fetch_yfinance(self, symbol: str, period: str, interval: str) -> Optional[Dict[str, Any]]:
        try:
            import yfinance as yf

            df = yf.Ticker(symbol).history(period=period, interval=interval)
            if df is None or df.empty:
                return None
            latest = df.iloc[-1]
            ts = df.index[-1]
            return {
                "timestamp": ts.isoformat() if hasattr(ts, "isoformat") else str(ts),
                "open": float(latest.get("Open", 0) or 0),
                "high": float(latest.get("High", 0) or 0),
                "low": float(latest.get("Low", 0) or 0),
                "close": float(latest.get("Close", 0) or 0),
                "volume": int(latest.get("Volume", 0) or 0),
                "source": "yfinance",
                "df": df,
            }
        except Exception as e:
            self.logger.debug(f"[YF] {symbol} error: {e}")
            return None

    def _fetch_tradier(self, symbol: str) -> Optional[Dict[str, Any]]:
        if not self.tradier_token:
            return None
        try:
            r = requests.get(
                "https://api.tradier.com/v1/markets/quotes",
                params={"symbols": symbol},
                headers={"Authorization": f"Bearer {self.tradier_token}", "Accept": "application/json"},
                timeout=10,
            )
            if r.status_code != 200:
                return None
            data = r.json()
            quote = data.get("quotes", {}).get("quote", {})
            if isinstance(quote, list):
                quote = quote[0] if quote else {}
            if not quote or not quote.get("last"):
                return None
            return {
                "timestamp": datetime.utcnow().isoformat(),
                "open": float(quote.get("open") or 0),
                "high": float(quote.get("high") or 0),
                "low": float(quote.get("low") or 0),
                "close": float(quote.get("last") or 0),
                "volume": int(quote.get("volume") or 0),
                "source": "tradier",
            }
        except Exception as e:
            self.logger.debug(f"[TRADIER] {symbol} error: {e}")
            return None


class LiquidityFetcher:
    def __init__(self, tradier_token: Optional[str], logger):
        self.tradier_token = tradier_token
        self.logger = logger
        self.base_url = "https://api.tradier.com/v1"

    def _req(self, endpoint: str, params: dict) -> Optional[Dict[str, Any]]:
        if not self.tradier_token:
            return None
        try:
            r = requests.get(
                f"{self.base_url}{endpoint}",
                params=params,
                headers={"Authorization": f"Bearer {self.tradier_token}", "Accept": "application/json"},
                timeout=15,
            )
            if r.status_code != 200:
                return None
            return r.json()
        except Exception as e:
            self.logger.debug(f"[TRADIER] req error: {e}")
            return None

    def expirations(self, symbol: str) -> List[str]:
        j = self._req("/markets/options/expirations", {"symbol": symbol})
        if not j:
            return []
        return j.get("expirations", {}).get("date", []) or []

    def chain(self, symbol: str, expiration: str) -> List[Dict[str, Any]]:
        j = self._req(
            "/markets/options/chains",
            {"symbol": symbol, "expiration": expiration, "greeks": "true"},
        )
        if not j or not j.get("options"):
            return []
        opts = j["options"].get("option", [])
        return opts if isinstance(opts, list) else [opts]

    def atm_snapshots(self, symbol: str, underlying_price: float, vix_value: Optional[float]) -> List[Dict[str, Any]]:
        if not self.tradier_token:
            return []
        exps = self.expirations(symbol)
        if not exps:
            return []
        target = datetime.utcnow() + timedelta(days=35)
        best = min(exps, key=lambda x: abs((datetime.strptime(x, "%Y-%m-%d") - target).days))
        chain = self.chain(symbol, best)
        if not chain:
            return []

        out: List[Dict[str, Any]] = []
        for opt_type in ("call", "put"):
            subset = [o for o in chain if o.get("option_type") == opt_type]
            if not subset:
                continue
            atm = min(subset, key=lambda o: abs(float(o.get("strike") or 0) - underlying_price))
            bid = float(atm.get("bid") or 0)
            ask = float(atm.get("ask") or 0)
            vol = int(atm.get("volume") or 0)
            oi = int(atm.get("open_interest") or 0)
            if bid <= 0 or ask <= 0:
                continue
            mid = (bid + ask) / 2
            spread_pct = ((ask - bid) / mid) * 100 if mid > 0 else 0
            quality = 0
            quality += max(0, 40 - spread_pct * 8)
            quality += min(30, vol / 10)
            quality += min(30, oi / 50)
            greeks = atm.get("greeks", {}) or {}
            out.append(
                {
                    "timestamp": datetime.utcnow().isoformat(),
                    "symbol": symbol,
                    "option_symbol": atm.get("symbol"),
                    "underlying_price": underlying_price,
                    "strike_price": float(atm.get("strike") or 0),
                    "option_type": opt_type.upper(),
                    "expiration_date": best,
                    "bid": bid,
                    "ask": ask,
                    "spread_pct": spread_pct,
                    "mid_price": mid,
                    "volume": vol,
                    "open_interest": oi,
                    "quality_score": quality,
                    "implied_volatility": float(greeks.get("mid_iv") or 0),
                    "delta": float(greeks.get("delta") or 0),
                    "gamma": float(greeks.get("gamma") or 0),
                    "theta": float(greeks.get("theta") or 0),
                    "vega": float(greeks.get("vega") or 0),
                    "vix_value": vix_value,
                }
            )
        return out

    def full_chain_snapshots(
        self,
        symbol: str,
        underlying_price: float,
        vix_value: Optional[float],
        min_dte: int = 0,
        max_dte: int = 45,
        strike_range_pct: float = 10.0,
        max_expirations: int = 5,
    ) -> List[Dict[str, Any]]:
        """Fetch full options chain data for multiple expirations and strikes."""
        if not self.tradier_token:
            return []

        exps = self.expirations(symbol)
        if not exps:
            return []

        now = datetime.utcnow()
        valid_exps = []
        for exp in exps:
            try:
                exp_date = datetime.strptime(exp, "%Y-%m-%d")
                dte = (exp_date - now).days
                if min_dte <= dte <= max_dte:
                    valid_exps.append((exp, dte))
            except ValueError:
                continue

        # Sort by DTE and take the closest ones
        valid_exps.sort(key=lambda x: x[1])
        selected_exps = [e[0] for e in valid_exps[:max_expirations]]

        if not selected_exps:
            return []

        # Calculate strike range
        min_strike = underlying_price * (1 - strike_range_pct / 100)
        max_strike = underlying_price * (1 + strike_range_pct / 100)

        out: List[Dict[str, Any]] = []
        ts = datetime.utcnow().isoformat()

        for exp in selected_exps:
            chain = self.chain(symbol, exp)
            if not chain:
                continue

            for opt in chain:
                strike = float(opt.get("strike") or 0)
                if strike < min_strike or strike > max_strike:
                    continue

                bid = float(opt.get("bid") or 0)
                ask = float(opt.get("ask") or 0)

                # Skip options with no market
                if bid <= 0 and ask <= 0:
                    continue

                vol = int(opt.get("volume") or 0)
                oi = int(opt.get("open_interest") or 0)
                mid = (bid + ask) / 2 if (bid > 0 and ask > 0) else (bid or ask)
                spread_pct = ((ask - bid) / mid) * 100 if mid > 0 and bid > 0 else 999

                # Quality score
                quality = 0
                quality += max(0, 40 - spread_pct * 8)
                quality += min(30, vol / 10)
                quality += min(30, oi / 50)

                greeks = opt.get("greeks", {}) or {}

                out.append({
                    "timestamp": ts,
                    "symbol": symbol,
                    "option_symbol": opt.get("symbol"),
                    "underlying_price": underlying_price,
                    "strike_price": strike,
                    "option_type": (opt.get("option_type") or "").upper(),
                    "expiration_date": exp,
                    "bid": bid,
                    "ask": ask,
                    "spread_pct": spread_pct,
                    "mid_price": mid,
                    "volume": vol,
                    "open_interest": oi,
                    "quality_score": quality,
                    "implied_volatility": float(greeks.get("mid_iv") or 0),
                    "delta": float(greeks.get("delta") or 0),
                    "gamma": float(greeks.get("gamma") or 0),
                    "theta": float(greeks.get("theta") or 0),
                    "vega": float(greeks.get("vega") or 0),
                    "vix_value": vix_value,
                    "last": float(opt.get("last") or 0),
                    "change": float(opt.get("change") or 0),
                    "change_pct": float(opt.get("change_percentage") or 0),
                    "root_symbol": opt.get("root_symbol"),
                })

        return out
