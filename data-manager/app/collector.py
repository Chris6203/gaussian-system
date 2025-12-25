from __future__ import annotations

import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from .fetchers import DataFetcher, LiquidityFetcher
from .market_hours import get_market_status
from .storage import DataStorage


def _get_symbols(cfg: Dict[str, Any]) -> List[str]:
    symbols = cfg.get("data_fetching", {}).get("symbols", ["SPY", "^VIX", "QQQ", "UUP"])
    primary = cfg.get("trading", {}).get("symbol", "SPY")
    if primary not in symbols:
        symbols = [primary] + list(symbols)
    return symbols


def _get_trading_symbol(cfg: Dict[str, Any]) -> str:
    return cfg.get("trading", {}).get("symbol", "SPY")


def _get_tradier_token(cfg: Dict[str, Any]) -> Optional[str]:
    t = cfg.get("credentials", {}).get("tradier", {})
    data_token = t.get("data_api_token")
    sandbox = t.get("sandbox", {})
    live = t.get("live", {})
    access = sandbox.get("access_token") or live.get("access_token")
    return data_token or access


class Collector:
    def __init__(self, cfg: Dict[str, Any], storage: DataStorage, logger):
        self.cfg = cfg
        self.storage = storage
        self.logger = logger
        self.cycle = 0
        self.last_vix: Optional[float] = None

        token = _get_tradier_token(cfg)
        self.price_fetcher = DataFetcher(token, logger)
        self.liq_fetcher = LiquidityFetcher(token, logger)

    def backfill(self, days: int) -> int:
        import yfinance as yf

        total = 0
        for sym in _get_symbols(self.cfg):
            try:
                df = yf.Ticker(sym).history(period=f"{days}d", interval="1m")
                total += self.storage.save_price_dataframe(sym, df)
                time.sleep(1)
            except Exception as e:
                self.logger.error(f"[BACKFILL] {sym} failed: {e}")
        return total

    def collect_once(self) -> Tuple[int, int]:
        self.cycle += 1
        t0 = time.time()

        is_open, status = get_market_status()
        symbols = _get_symbols(self.cfg)
        trading_symbol = _get_trading_symbol(self.cfg)

        self.logger.info(f"[CYCLE {self.cycle}] Market: {status}")

        price_saved = 0
        liq_saved = 0

        for sym in symbols:
            try:
                data = self.price_fetcher.get_price_data(sym)
                if not data:
                    continue
                if self.storage.save_price_point(sym.lstrip("^"), data):
                    price_saved += 1
                    if "VIX" in sym.upper():
                        self.last_vix = float(data.get("close") or 0) if data.get("close") else self.last_vix
            except Exception as e:
                self.logger.error(f"[PRICE] {sym} failed: {e}")

        enable_liq = bool(self.cfg.get("collector", {}).get("enable_options_liquidity", True))
        options_cfg = self.cfg.get("options_chain", {})
        full_chain_enabled = bool(options_cfg.get("enabled", False))

        if is_open and (enable_liq or full_chain_enabled):
            try:
                px = self.price_fetcher.get_price_data(trading_symbol)
                if px:
                    underlying = float(px.get("close") or 0)

                    if full_chain_enabled:
                        # Collect full options chain
                        chain_symbols = options_cfg.get("symbols", [trading_symbol])
                        min_dte = int(options_cfg.get("min_dte", 0))
                        max_dte = int(options_cfg.get("max_dte", 45))
                        strike_range = float(options_cfg.get("strike_range_pct", 10))
                        max_exps = int(options_cfg.get("expirations_to_collect", 5))

                        for sym in chain_symbols:
                            sym_px = self.price_fetcher.get_price_data(sym) if sym != trading_symbol else px
                            sym_underlying = float(sym_px.get("close") or 0) if sym_px else underlying

                            snaps = self.liq_fetcher.full_chain_snapshots(
                                sym,
                                sym_underlying,
                                self.last_vix,
                                min_dte=min_dte,
                                max_dte=max_dte,
                                strike_range_pct=strike_range,
                                max_expirations=max_exps,
                            )
                            for s in snaps:
                                if self.storage.save_liquidity_snapshot(s):
                                    liq_saved += 1
                    elif enable_liq:
                        # Legacy ATM-only collection
                        snaps = self.liq_fetcher.atm_snapshots(trading_symbol, underlying, self.last_vix)
                        for s in snaps:
                            if self.storage.save_liquidity_snapshot(s):
                                liq_saved += 1
            except Exception as e:
                self.logger.error(f"[LIQ] failed: {e}")

        dur_ms = int((time.time() - t0) * 1000)
        self.storage.log_cycle(symbols, price_saved, liq_saved, is_open, dur_ms)
        self.logger.info(f"[CYCLE {self.cycle}] Saved {price_saved} price, {liq_saved} liquidity ({dur_ms}ms)")

        return price_saved, liq_saved

    def run_forever(self) -> None:
        interval = int(self.cfg.get("collector", {}).get("interval_seconds", 60))
        integ_every = int(self.cfg.get("collector", {}).get("integrity_check_interval_cycles", 60))

        self.logger.info("=" * 70)
        self.logger.info("DATA-MANAGER COLLECTOR STARTED")
        self.logger.info(f"Symbols: {', '.join(_get_symbols(self.cfg))}")
        self.logger.info(f"Interval: {interval}s")
        self.logger.info("=" * 70)

        while True:
            start = time.time()
            self.collect_once()

            if self.cycle % max(1, integ_every) == 0:
                chk = self.storage.check_integrity()
                issues = chk.get("issues") or []
                if issues:
                    self.logger.warning(f"[INTEGRITY] Issues: {issues}")

            elapsed = time.time() - start
            time.sleep(max(0, interval - elapsed))
