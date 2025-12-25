import sqlite3
import threading
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union


class DataStorage:
    def __init__(self, db_path: Path, logger):
        self.db_path = db_path
        self.logger = logger
        self._lock = threading.Lock()
        self._ensure_database()

    def _ensure_database(self) -> None:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        conn = sqlite3.connect(str(self.db_path))
        cur = conn.cursor()

        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS historical_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                timestamp DATETIME NOT NULL,
                open_price REAL,
                high_price REAL,
                low_price REAL,
                close_price REAL,
                volume INTEGER,
                UNIQUE(symbol, timestamp)
            )
            """
        )
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_symbol_timestamp ON historical_data(symbol, timestamp)"
        )

        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS liquidity_snapshots (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME NOT NULL,
                symbol TEXT NOT NULL,
                option_symbol TEXT,
                underlying_price REAL,
                strike_price REAL,
                option_type TEXT,
                expiration_date TEXT,
                bid REAL,
                ask REAL,
                spread_pct REAL,
                mid_price REAL,
                volume INTEGER,
                open_interest INTEGER,
                quality_score REAL,
                implied_volatility REAL,
                delta REAL,
                gamma REAL,
                theta REAL,
                vega REAL,
                vix_value REAL,
                last_price REAL,
                change_value REAL,
                change_pct REAL,
                root_symbol TEXT,
                UNIQUE(symbol, option_symbol, timestamp)
            )
            """
        )

        # Add new columns if they don't exist (migration)
        try:
            cur.execute("ALTER TABLE liquidity_snapshots ADD COLUMN last_price REAL")
        except Exception:
            pass
        try:
            cur.execute("ALTER TABLE liquidity_snapshots ADD COLUMN change_value REAL")
        except Exception:
            pass
        try:
            cur.execute("ALTER TABLE liquidity_snapshots ADD COLUMN change_pct REAL")
        except Exception:
            pass
        try:
            cur.execute("ALTER TABLE liquidity_snapshots ADD COLUMN root_symbol TEXT")
        except Exception:
            pass
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_liquidity_symbol_timestamp ON liquidity_snapshots(symbol, timestamp)"
        )

        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS collection_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME NOT NULL,
                symbols_collected TEXT,
                price_records INTEGER,
                liquidity_records INTEGER,
                market_open INTEGER,
                duration_ms INTEGER,
                notes TEXT
            )
            """
        )

        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS sentiment (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME NOT NULL,
                symbol TEXT,
                sentiment_type TEXT NOT NULL,
                value REAL NOT NULL,
                source TEXT,
                notes TEXT,
                metadata TEXT,
                headline TEXT,
                url TEXT,
                confidence REAL,
                model TEXT
            )
            """
        )
        # Add new columns if they don't exist (migration)
        for col, coltype in [("headline", "TEXT"), ("url", "TEXT"), ("confidence", "REAL"), ("model", "TEXT")]:
            try:
                cur.execute(f"ALTER TABLE sentiment ADD COLUMN {col} {coltype}")
            except Exception:
                pass
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_sentiment_timestamp ON sentiment(timestamp)"
        )
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_sentiment_symbol ON sentiment(symbol, timestamp)"
        )
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_sentiment_source ON sentiment(source, timestamp)"
        )
        cur.execute(
            "CREATE UNIQUE INDEX IF NOT EXISTS idx_sentiment_headline_dedup ON sentiment(source, headline, symbol) WHERE headline IS NOT NULL"
        )

        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS api_keys (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                key_hash TEXT NOT NULL UNIQUE,
                name TEXT NOT NULL,
                created_at DATETIME NOT NULL,
                last_used DATETIME,
                permissions TEXT,
                active INTEGER DEFAULT 1
            )
            """
        )

        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS bots (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                owner TEXT NOT NULL,
                api_key_id INTEGER,
                description TEXT,
                bot_type TEXT DEFAULT 'trading',
                status TEXT DEFAULT 'active',
                created_at DATETIME NOT NULL,
                config TEXT,
                config_hash TEXT,
                config_version INTEGER DEFAULT 1,
                config_summary TEXT,
                FOREIGN KEY (api_key_id) REFERENCES api_keys(id)
            )
            """
        )
        cur.execute("CREATE INDEX IF NOT EXISTS idx_bots_owner ON bots(owner)")

        # Add new columns if they don't exist (migration)
        for col, coltype in [
            ("config_hash", "TEXT"),
            ("config_version", "INTEGER DEFAULT 1"),
            ("config_summary", "TEXT"),
        ]:
            try:
                cur.execute(f"ALTER TABLE bots ADD COLUMN {col} {coltype}")
            except Exception:
                pass

        # Bot config history table for tracking changes over time
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS bot_config_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                bot_id INTEGER NOT NULL,
                timestamp DATETIME NOT NULL,
                config TEXT NOT NULL,
                config_hash TEXT NOT NULL,
                config_version INTEGER NOT NULL,
                previous_hash TEXT,
                change_summary TEXT,
                FOREIGN KEY (bot_id) REFERENCES bots(id)
            )
            """
        )
        cur.execute("CREATE INDEX IF NOT EXISTS idx_config_history_bot ON bot_config_history(bot_id, timestamp)")

        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS bot_performance (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                bot_id INTEGER NOT NULL,
                timestamp DATETIME NOT NULL,
                metric_type TEXT NOT NULL,
                value REAL NOT NULL,
                metadata TEXT,
                FOREIGN KEY (bot_id) REFERENCES bots(id)
            )
            """
        )
        cur.execute("CREATE INDEX IF NOT EXISTS idx_bot_perf_bot ON bot_performance(bot_id, timestamp)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_bot_perf_metric ON bot_performance(metric_type, timestamp)")

        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS bot_trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                bot_id INTEGER NOT NULL,
                timestamp DATETIME NOT NULL,
                symbol TEXT NOT NULL,
                action TEXT NOT NULL,
                quantity REAL NOT NULL,
                price REAL NOT NULL,
                pnl REAL,
                pnl_pct REAL,
                notes TEXT,
                metadata TEXT,
                config_version INTEGER,
                FOREIGN KEY (bot_id) REFERENCES bots(id)
            )
            """
        )
        cur.execute("CREATE INDEX IF NOT EXISTS idx_bot_trades_bot ON bot_trades(bot_id, timestamp)")

        # Add config_version column if it doesn't exist (migration)
        try:
            cur.execute("ALTER TABLE bot_trades ADD COLUMN config_version INTEGER")
        except Exception:
            pass

        conn.commit()
        conn.close()

    def save_price_point(self, symbol: str, point: Dict[str, Any]) -> bool:
        try:
            with self._lock:
                conn = sqlite3.connect(str(self.db_path))
                cur = conn.cursor()
                cur.execute(
                    """
                    INSERT OR REPLACE INTO historical_data
                    (symbol, timestamp, open_price, high_price, low_price, close_price, volume)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        symbol,
                        point.get("timestamp"),
                        point.get("open", 0),
                        point.get("high", 0),
                        point.get("low", 0),
                        point.get("close", 0),
                        point.get("volume", 0),
                    ),
                )
                conn.commit()
                conn.close()
            return True
        except Exception as e:
            self.logger.error(f"[DB] save_price_point failed: {e}")
            return False

    def save_price_dataframe(self, symbol: str, df) -> int:
        if df is None or getattr(df, "empty", True):
            return 0

        saved = 0
        try:
            with self._lock:
                conn = sqlite3.connect(str(self.db_path))
                cur = conn.cursor()
                for idx, row in df.iterrows():
                    try:
                        ts = idx.isoformat() if hasattr(idx, "isoformat") else str(idx)
                        cur.execute(
                            """
                            INSERT OR REPLACE INTO historical_data
                            (symbol, timestamp, open_price, high_price, low_price, close_price, volume)
                            VALUES (?, ?, ?, ?, ?, ?, ?)
                            """,
                            (
                                symbol,
                                ts,
                                float(row.get("Open", row.get("open", 0)) or 0),
                                float(row.get("High", row.get("high", 0)) or 0),
                                float(row.get("Low", row.get("low", 0)) or 0),
                                float(row.get("Close", row.get("close", 0)) or 0),
                                int(row.get("Volume", row.get("volume", 0)) or 0),
                            ),
                        )
                        saved += 1
                    except Exception:
                        pass
                conn.commit()
                conn.close()
        except Exception as e:
            self.logger.error(f"[DB] save_price_dataframe failed: {e}")

        return saved

    def save_liquidity_snapshot(self, s: Dict[str, Any]) -> bool:
        try:
            with self._lock:
                conn = sqlite3.connect(str(self.db_path))
                cur = conn.cursor()
                cur.execute(
                    """
                    INSERT OR REPLACE INTO liquidity_snapshots
                    (timestamp, symbol, option_symbol, underlying_price, strike_price,
                     option_type, expiration_date, bid, ask, spread_pct, mid_price,
                     volume, open_interest, quality_score, implied_volatility,
                     delta, gamma, theta, vega, vix_value, last_price, change_value, change_pct, root_symbol)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        s.get("timestamp"),
                        s.get("symbol"),
                        s.get("option_symbol"),
                        s.get("underlying_price"),
                        s.get("strike_price"),
                        s.get("option_type"),
                        s.get("expiration_date"),
                        s.get("bid"),
                        s.get("ask"),
                        s.get("spread_pct"),
                        s.get("mid_price"),
                        s.get("volume"),
                        s.get("open_interest"),
                        s.get("quality_score"),
                        s.get("implied_volatility"),
                        s.get("delta"),
                        s.get("gamma"),
                        s.get("theta"),
                        s.get("vega"),
                        s.get("vix_value"),
                        s.get("last"),
                        s.get("change"),
                        s.get("change_pct"),
                        s.get("root_symbol"),
                    ),
                )
                conn.commit()
                conn.close()
            return True
        except Exception as e:
            self.logger.debug(f"[DB] save_liquidity_snapshot failed: {e}")
            return False

    def log_cycle(
        self,
        symbols: List[str],
        price_records: int,
        liquidity_records: int,
        market_open: bool,
        duration_ms: int,
        notes: str | None = None,
    ) -> None:
        try:
            conn = sqlite3.connect(str(self.db_path))
            cur = conn.cursor()
            cur.execute(
                """
                INSERT INTO collection_log
                (timestamp, symbols_collected, price_records, liquidity_records, market_open, duration_ms, notes)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    datetime.utcnow().isoformat(),
                    ",".join(symbols),
                    price_records,
                    liquidity_records,
                    1 if market_open else 0,
                    duration_ms,
                    notes,
                ),
            )
            conn.commit()
            conn.close()
        except Exception as e:
            self.logger.error(f"[DB] log_cycle failed: {e}")

    def stats(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {"symbols": {}, "liquidity": {}, "collection": {}}
        if not self.db_path.exists():
            return out

        try:
            conn = sqlite3.connect(str(self.db_path))
            cur = conn.cursor()

            cur.execute(
                "SELECT symbol, COUNT(*), MIN(timestamp), MAX(timestamp) FROM historical_data GROUP BY symbol"
            )
            for sym, cnt, mn, mx in cur.fetchall():
                out["symbols"][sym] = {"records": cnt, "first": mn, "last": mx}

            cur.execute("SELECT COUNT(*), MIN(timestamp), MAX(timestamp), AVG(quality_score) FROM liquidity_snapshots")
            row = cur.fetchone() or (0, None, None, None)
            out["liquidity"] = {
                "total_snapshots": row[0],
                "first": row[1],
                "last": row[2],
                "avg_quality": round(row[3], 1) if row[3] else 0,
            }

            cur.execute(
                """
                SELECT COUNT(*), SUM(price_records), SUM(liquidity_records)
                FROM collection_log
                WHERE timestamp > datetime('now', '-24 hours')
                """
            )
            row = cur.fetchone() or (0, 0, 0)
            out["collection"] = {
                "cycles_24h": row[0],
                "price_records_24h": row[1] or 0,
                "liquidity_records_24h": row[2] or 0,
            }

            conn.close()
        except Exception as e:
            out["error"] = str(e)

        return out

    def check_integrity(self) -> Dict[str, Any]:
        results: Dict[str, Any] = {
            "duplicates": 0,
            "null_values": 0,
            "invalid_prices": 0,
            "liquidity_duplicates": 0,
            "issues": [],
        }

        try:
            conn = sqlite3.connect(str(self.db_path))
            cur = conn.cursor()

            cur.execute(
                """
                SELECT symbol, timestamp, COUNT(*) as cnt
                FROM historical_data
                GROUP BY symbol, timestamp
                HAVING cnt > 1
                """
            )
            dups = cur.fetchall()
            results["duplicates"] = len(dups)
            if dups:
                results["issues"].append(f"Found {len(dups)} duplicate price records")

            cur.execute(
                "SELECT COUNT(*) FROM historical_data WHERE close_price IS NULL OR timestamp IS NULL"
            )
            results["null_values"] = (cur.fetchone() or (0,))[0]
            if results["null_values"]:
                results["issues"].append(f"Found {results['null_values']} NULL price rows")

            cur.execute(
                "SELECT COUNT(*) FROM historical_data WHERE close_price <= 0 OR close_price > 100000"
            )
            results["invalid_prices"] = (cur.fetchone() or (0,))[0]
            if results["invalid_prices"]:
                results["issues"].append(f"Found {results['invalid_prices']} invalid price rows")

            cur.execute(
                """
                SELECT symbol, option_symbol, timestamp, COUNT(*) as cnt
                FROM liquidity_snapshots
                GROUP BY symbol, option_symbol, timestamp
                HAVING cnt > 1
                """
            )
            ldups = cur.fetchall()
            results["liquidity_duplicates"] = len(ldups)
            if ldups:
                results["issues"].append(f"Found {len(ldups)} duplicate liquidity rows")

            conn.close()
        except Exception as e:
            results["issues"].append(f"Integrity check failed: {e}")

        return results

    def remove_duplicates(self) -> Dict[str, Any]:
        out = {"price_removed": 0, "liquidity_removed": 0}
        try:
            conn = sqlite3.connect(str(self.db_path))
            cur = conn.cursor()

            cur.execute(
                """
                SELECT symbol, timestamp, COUNT(*) as cnt, MAX(id) as keep_id
                FROM historical_data
                GROUP BY symbol, timestamp
                HAVING cnt > 1
                """
            )
            for sym, ts, cnt, keep_id in cur.fetchall():
                cur.execute(
                    "DELETE FROM historical_data WHERE symbol = ? AND timestamp = ? AND id != ?",
                    (sym, ts, keep_id),
                )
                out["price_removed"] += int(cnt) - 1

            cur.execute(
                """
                SELECT symbol, option_symbol, timestamp, COUNT(*) as cnt, MAX(id) as keep_id
                FROM liquidity_snapshots
                GROUP BY symbol, option_symbol, timestamp
                HAVING cnt > 1
                """
            )
            for sym, opt, ts, cnt, keep_id in cur.fetchall():
                cur.execute(
                    "DELETE FROM liquidity_snapshots WHERE symbol = ? AND option_symbol = ? AND timestamp = ? AND id != ?",
                    (sym, opt, ts, keep_id),
                )
                out["liquidity_removed"] += int(cnt) - 1

            conn.commit()
            conn.close()
        except Exception as e:
            out["error"] = str(e)

        return out

    def optimize(self) -> bool:
        try:
            conn = sqlite3.connect(str(self.db_path))
            cur = conn.cursor()
            cur.execute("VACUUM")
            cur.execute("REINDEX")
            conn.commit()
            conn.close()
            return True
        except Exception as e:
            self.logger.error(f"[DB] optimize failed: {e}")
            return False

    def save_sentiment(self, record: Dict[str, Any]) -> int | None:
        """Save a sentiment record. Returns the record ID or None on failure.

        Fields:
            symbol: Stock symbol (optional)
            sentiment_type: e.g., "bullish", "bearish", "neutral", "fear", "greed"
            value: Numeric score, typically -100 to 100
            source: Where this came from, e.g., "reuters", "twitter", "reddit"
            notes: Free text notes
            metadata: JSON object for extra data
            headline: The headline text being analyzed
            url: Source URL
            confidence: Model confidence 0.0-1.0
            model: Which model produced this, e.g., "gpt-4", "finbert"
        """
        try:
            import json

            with self._lock:
                conn = sqlite3.connect(str(self.db_path))
                cur = conn.cursor()
                metadata = record.get("metadata")
                if isinstance(metadata, dict):
                    metadata = json.dumps(metadata)
                cur.execute(
                    """
                    INSERT OR IGNORE INTO sentiment
                    (timestamp, symbol, sentiment_type, value, source, notes, metadata,
                     headline, url, confidence, model)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        record.get("timestamp") or datetime.utcnow().isoformat(),
                        record.get("symbol"),
                        record.get("sentiment_type", "general"),
                        float(record.get("value", 0)),
                        record.get("source"),
                        record.get("notes"),
                        metadata,
                        record.get("headline"),
                        record.get("url"),
                        float(record.get("confidence")) if record.get("confidence") is not None else None,
                        record.get("model"),
                    ),
                )
                record_id = cur.lastrowid if cur.rowcount > 0 else None
                conn.commit()
                conn.close()
            return record_id
        except Exception as e:
            self.logger.error(f"[DB] save_sentiment failed: {e}")
            return None

    def get_sentiment(
        self,
        symbol: str | None = None,
        sentiment_type: str | None = None,
        source: str | None = None,
        model: str | None = None,
        start_time: str | None = None,
        end_time: str | None = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """Retrieve sentiment records with optional filters."""
        try:
            import json

            conn = sqlite3.connect(str(self.db_path))
            cur = conn.cursor()

            query = """SELECT id, timestamp, symbol, sentiment_type, value, source, notes, metadata,
                              headline, url, confidence, model
                       FROM sentiment WHERE 1=1"""
            params: List[Any] = []

            if symbol:
                query += " AND symbol = ?"
                params.append(symbol)
            if sentiment_type:
                query += " AND sentiment_type = ?"
                params.append(sentiment_type)
            if source:
                query += " AND source = ?"
                params.append(source)
            if model:
                query += " AND model = ?"
                params.append(model)
            if start_time:
                query += " AND timestamp >= ?"
                params.append(start_time)
            if end_time:
                query += " AND timestamp <= ?"
                params.append(end_time)

            query += " ORDER BY timestamp DESC LIMIT ?"
            params.append(limit)

            cur.execute(query, params)
            rows = cur.fetchall()
            conn.close()

            results = []
            for row in rows:
                metadata = row[7]
                if metadata:
                    try:
                        metadata = json.loads(metadata)
                    except Exception:
                        pass
                results.append({
                    "id": row[0],
                    "timestamp": row[1],
                    "symbol": row[2],
                    "sentiment_type": row[3],
                    "value": row[4],
                    "source": row[5],
                    "notes": row[6],
                    "metadata": metadata,
                    "headline": row[8],
                    "url": row[9],
                    "confidence": row[10],
                    "model": row[11],
                })
            return results
        except Exception as e:
            self.logger.error(f"[DB] get_sentiment failed: {e}")
            return []

    # ==================== API KEY MANAGEMENT ====================

    def create_api_key(self, name: str, permissions: str = "read,write") -> str | None:
        """Create a new API key. Returns the plain key (only shown once)."""
        import hashlib
        import secrets

        plain_key = f"dm_{secrets.token_hex(24)}"
        key_hash = hashlib.sha256(plain_key.encode()).hexdigest()

        try:
            conn = sqlite3.connect(str(self.db_path))
            cur = conn.cursor()
            cur.execute(
                """
                INSERT INTO api_keys (key_hash, name, created_at, permissions, active)
                VALUES (?, ?, ?, ?, 1)
                """,
                (key_hash, name, datetime.utcnow().isoformat(), permissions),
            )
            conn.commit()
            conn.close()
            return plain_key
        except Exception as e:
            self.logger.error(f"[DB] create_api_key failed: {e}")
            return None

    def validate_api_key(self, plain_key: str) -> Dict[str, Any] | None:
        """Validate an API key. Returns key info if valid, None otherwise."""
        import hashlib

        key_hash = hashlib.sha256(plain_key.encode()).hexdigest()

        try:
            conn = sqlite3.connect(str(self.db_path))
            cur = conn.cursor()
            cur.execute(
                "SELECT id, name, permissions, active FROM api_keys WHERE key_hash = ?",
                (key_hash,),
            )
            row = cur.fetchone()
            if row and row[3] == 1:
                # Update last_used
                cur.execute(
                    "UPDATE api_keys SET last_used = ? WHERE id = ?",
                    (datetime.utcnow().isoformat(), row[0]),
                )
                conn.commit()
                conn.close()
                return {"id": row[0], "name": row[1], "permissions": row[2].split(",")}
            conn.close()
            return None
        except Exception as e:
            self.logger.error(f"[DB] validate_api_key failed: {e}")
            return None

    def list_api_keys(self) -> List[Dict[str, Any]]:
        """List all API keys (without hashes)."""
        try:
            conn = sqlite3.connect(str(self.db_path))
            cur = conn.cursor()
            cur.execute(
                "SELECT id, name, created_at, last_used, permissions, active FROM api_keys ORDER BY created_at DESC"
            )
            rows = cur.fetchall()
            conn.close()
            return [
                {
                    "id": r[0],
                    "name": r[1],
                    "created_at": r[2],
                    "last_used": r[3],
                    "permissions": r[4],
                    "active": bool(r[5]),
                }
                for r in rows
            ]
        except Exception as e:
            self.logger.error(f"[DB] list_api_keys failed: {e}")
            return []

    def revoke_api_key(self, key_id: int) -> bool:
        """Revoke an API key."""
        try:
            conn = sqlite3.connect(str(self.db_path))
            cur = conn.cursor()
            cur.execute("UPDATE api_keys SET active = 0 WHERE id = ?", (key_id,))
            conn.commit()
            conn.close()
            return True
        except Exception as e:
            self.logger.error(f"[DB] revoke_api_key failed: {e}")
            return False

    def delete_api_key(self, key_id: int) -> bool:
        """Delete an API key."""
        try:
            conn = sqlite3.connect(str(self.db_path))
            cur = conn.cursor()
            cur.execute("DELETE FROM api_keys WHERE id = ?", (key_id,))
            conn.commit()
            conn.close()
            return True
        except Exception as e:
            self.logger.error(f"[DB] delete_api_key failed: {e}")
            return False

    # ==================== BULK DATA RETRIEVAL ====================

    def get_price_history(
        self,
        symbol: str,
        start_time: str | None = None,
        end_time: str | None = None,
        limit: int = 10000,
    ) -> List[Dict[str, Any]]:
        """Get historical price data for a symbol."""
        try:
            conn = sqlite3.connect(str(self.db_path))
            cur = conn.cursor()

            query = "SELECT timestamp, open_price, high_price, low_price, close_price, volume FROM historical_data WHERE symbol = ?"
            params: List[Any] = [symbol]

            if start_time:
                query += " AND timestamp >= ?"
                params.append(start_time)
            if end_time:
                query += " AND timestamp <= ?"
                params.append(end_time)

            query += " ORDER BY timestamp ASC LIMIT ?"
            params.append(limit)

            cur.execute(query, params)
            rows = cur.fetchall()
            conn.close()

            return [
                {
                    "timestamp": r[0],
                    "open": r[1],
                    "high": r[2],
                    "low": r[3],
                    "close": r[4],
                    "volume": r[5],
                }
                for r in rows
            ]
        except Exception as e:
            self.logger.error(f"[DB] get_price_history failed: {e}")
            return []

    def get_liquidity_history(
        self,
        symbol: str,
        start_time: str | None = None,
        end_time: str | None = None,
        option_type: str | None = None,
        limit: int = 10000,
    ) -> List[Dict[str, Any]]:
        """Get historical liquidity/options data for a symbol."""
        try:
            conn = sqlite3.connect(str(self.db_path))
            cur = conn.cursor()

            query = """SELECT timestamp, option_symbol, underlying_price, strike_price, option_type,
                              expiration_date, bid, ask, spread_pct, mid_price, volume, open_interest,
                              implied_volatility, delta, gamma, theta, vega
                       FROM liquidity_snapshots WHERE symbol = ?"""
            params: List[Any] = [symbol]

            if start_time:
                query += " AND timestamp >= ?"
                params.append(start_time)
            if end_time:
                query += " AND timestamp <= ?"
                params.append(end_time)
            if option_type:
                query += " AND option_type = ?"
                params.append(option_type.upper())

            query += " ORDER BY timestamp ASC LIMIT ?"
            params.append(limit)

            cur.execute(query, params)
            rows = cur.fetchall()
            conn.close()

            return [
                {
                    "timestamp": r[0],
                    "option_symbol": r[1],
                    "underlying_price": r[2],
                    "strike_price": r[3],
                    "option_type": r[4],
                    "expiration_date": r[5],
                    "bid": r[6],
                    "ask": r[7],
                    "spread_pct": r[8],
                    "mid_price": r[9],
                    "volume": r[10],
                    "open_interest": r[11],
                    "implied_volatility": r[12],
                    "delta": r[13],
                    "gamma": r[14],
                    "theta": r[15],
                    "vega": r[16],
                }
                for r in rows
            ]
        except Exception as e:
            self.logger.error(f"[DB] get_liquidity_history failed: {e}")
            return []

    def get_latest_prices(self, symbols: List[str] | None = None) -> Dict[str, Dict[str, Any]]:
        """Get the latest price for each symbol."""
        try:
            conn = sqlite3.connect(str(self.db_path))
            cur = conn.cursor()

            if symbols:
                placeholders = ",".join("?" * len(symbols))
                cur.execute(
                    f"""SELECT symbol, timestamp, open_price, high_price, low_price, close_price, volume
                        FROM historical_data
                        WHERE symbol IN ({placeholders})
                        AND timestamp = (SELECT MAX(timestamp) FROM historical_data h2 WHERE h2.symbol = historical_data.symbol)""",
                    symbols,
                )
            else:
                cur.execute(
                    """SELECT symbol, timestamp, open_price, high_price, low_price, close_price, volume
                       FROM historical_data
                       WHERE timestamp = (SELECT MAX(timestamp) FROM historical_data h2 WHERE h2.symbol = historical_data.symbol)"""
                )

            rows = cur.fetchall()
            conn.close()

            return {
                r[0]: {
                    "timestamp": r[1],
                    "open": r[2],
                    "high": r[3],
                    "low": r[4],
                    "close": r[5],
                    "volume": r[6],
                }
                for r in rows
            }
        except Exception as e:
            self.logger.error(f"[DB] get_latest_prices failed: {e}")
            return {}

    # ==================== BOT MANAGEMENT ====================

    def register_bot(self, name: str, owner: str, api_key_id: int | None = None,
                     description: str = "", bot_type: str = "trading", config: Dict = None,
                     config_hash: str = None, config_version: int = 1, config_summary: Dict = None) -> int | None:
        """Register a new bot with full config capture. Returns bot ID."""
        import json
        try:
            conn = sqlite3.connect(str(self.db_path))
            cur = conn.cursor()
            cur.execute(
                """INSERT INTO bots (name, owner, api_key_id, description, bot_type, status, created_at,
                   config, config_hash, config_version, config_summary)
                   VALUES (?, ?, ?, ?, ?, 'active', ?, ?, ?, ?, ?)""",
                (name, owner, api_key_id, description, bot_type,
                 datetime.utcnow().isoformat(),
                 json.dumps(config) if config else None,
                 config_hash,
                 config_version,
                 json.dumps(config_summary) if config_summary else None),
            )
            bot_id = cur.lastrowid

            # Also save to config history
            if bot_id and config:
                cur.execute(
                    """INSERT INTO bot_config_history (bot_id, timestamp, config, config_hash, config_version)
                       VALUES (?, ?, ?, ?, ?)""",
                    (bot_id, datetime.utcnow().isoformat(), json.dumps(config), config_hash, config_version)
                )

            conn.commit()
            conn.close()
            return bot_id
        except Exception as e:
            self.logger.error(f"[DB] register_bot failed: {e}")
            return None

    def get_bot(self, bot_id: int) -> Dict[str, Any] | None:
        """Get bot by ID including config info."""
        import json
        try:
            conn = sqlite3.connect(str(self.db_path))
            cur = conn.cursor()
            cur.execute("SELECT id, name, owner, api_key_id, description, bot_type, status, created_at, config, config_hash, config_version, config_summary FROM bots WHERE id = ?", (bot_id,))
            row = cur.fetchone()
            conn.close()
            if row:
                config = row[8]
                if config:
                    try:
                        config = json.loads(config)
                    except:
                        pass
                config_summary = row[11]
                if config_summary:
                    try:
                        config_summary = json.loads(config_summary)
                    except:
                        pass
                return {
                    "id": row[0], "name": row[1], "owner": row[2], "api_key_id": row[3],
                    "description": row[4], "bot_type": row[5], "status": row[6],
                    "created_at": row[7], "config": config,
                    "config_hash": row[9], "config_version": row[10], "config_summary": config_summary,
                }
            return None
        except Exception as e:
            self.logger.error(f"[DB] get_bot failed: {e}")
            return None

    def get_bot_by_api_key(self, api_key_id: int) -> Dict[str, Any] | None:
        """Get bot associated with an API key."""
        import json
        try:
            conn = sqlite3.connect(str(self.db_path))
            cur = conn.cursor()
            cur.execute("SELECT * FROM bots WHERE api_key_id = ?", (api_key_id,))
            row = cur.fetchone()
            conn.close()
            if row:
                config = row[8]
                if config:
                    try:
                        config = json.loads(config)
                    except:
                        pass
                return {
                    "id": row[0], "name": row[1], "owner": row[2], "api_key_id": row[3],
                    "description": row[4], "bot_type": row[5], "status": row[6],
                    "created_at": row[7], "config": config,
                }
            return None
        except Exception as e:
            self.logger.error(f"[DB] get_bot_by_api_key failed: {e}")
            return None

    def list_bots(self, owner: str | None = None) -> List[Dict[str, Any]]:
        """List bots, optionally filtered by owner."""
        try:
            conn = sqlite3.connect(str(self.db_path))
            cur = conn.cursor()
            if owner:
                cur.execute("SELECT id, name, owner, bot_type, status, created_at, description FROM bots WHERE owner = ? ORDER BY created_at DESC", (owner,))
            else:
                cur.execute("SELECT id, name, owner, bot_type, status, created_at, description FROM bots ORDER BY created_at DESC")
            rows = cur.fetchall()
            conn.close()
            return [{"id": r[0], "name": r[1], "owner": r[2], "bot_type": r[3], "status": r[4], "created_at": r[5], "description": r[6]} for r in rows]
        except Exception as e:
            self.logger.error(f"[DB] list_bots failed: {e}")
            return []

    def update_bot_status(self, bot_id: int, status: str) -> bool:
        """Update bot status (active, paused, stopped)."""
        try:
            conn = sqlite3.connect(str(self.db_path))
            cur = conn.cursor()
            cur.execute("UPDATE bots SET status = ? WHERE id = ?", (status, bot_id))
            conn.commit()
            conn.close()
            return True
        except Exception as e:
            self.logger.error(f"[DB] update_bot_status failed: {e}")
            return False

    def record_bot_metric(self, bot_id: int, metric_type: str, value: float, metadata: Dict = None) -> bool:
        """Record a performance metric for a bot."""
        import json
        try:
            conn = sqlite3.connect(str(self.db_path))
            cur = conn.cursor()
            cur.execute(
                "INSERT INTO bot_performance (bot_id, timestamp, metric_type, value, metadata) VALUES (?, ?, ?, ?, ?)",
                (bot_id, datetime.utcnow().isoformat(), metric_type, value, json.dumps(metadata) if metadata else None),
            )
            conn.commit()
            conn.close()
            return True
        except Exception as e:
            self.logger.error(f"[DB] record_bot_metric failed: {e}")
            return False

    def record_bot_trade(self, bot_id: int, symbol: str, action: str, quantity: float, price: float,
                         pnl: float = None, pnl_pct: float = None, notes: str = None, metadata: Dict = None) -> int | None:
        """Record a trade made by a bot."""
        import json
        try:
            conn = sqlite3.connect(str(self.db_path))
            cur = conn.cursor()
            cur.execute(
                """INSERT INTO bot_trades (bot_id, timestamp, symbol, action, quantity, price, pnl, pnl_pct, notes, metadata)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (bot_id, datetime.utcnow().isoformat(), symbol.upper(), action.upper(), quantity, price,
                 pnl, pnl_pct, notes, json.dumps(metadata) if metadata else None),
            )
            trade_id = cur.lastrowid
            conn.commit()
            conn.close()
            return trade_id
        except Exception as e:
            self.logger.error(f"[DB] record_bot_trade failed: {e}")
            return None

    def get_bot_trades(self, bot_id: int, limit: int = 100, start_time: str = None) -> List[Dict[str, Any]]:
        """Get trades for a bot."""
        import json
        try:
            conn = sqlite3.connect(str(self.db_path))
            cur = conn.cursor()
            query = "SELECT * FROM bot_trades WHERE bot_id = ?"
            params: List[Any] = [bot_id]
            if start_time:
                query += " AND timestamp >= ?"
                params.append(start_time)
            query += " ORDER BY timestamp DESC LIMIT ?"
            params.append(limit)
            cur.execute(query, params)
            rows = cur.fetchall()
            conn.close()
            results = []
            for r in rows:
                meta = r[10]
                if meta:
                    try:
                        meta = json.loads(meta)
                    except:
                        pass
                results.append({
                    "id": r[0], "bot_id": r[1], "timestamp": r[2], "symbol": r[3],
                    "action": r[4], "quantity": r[5], "price": r[6], "pnl": r[7],
                    "pnl_pct": r[8], "notes": r[9], "metadata": meta,
                })
            return results
        except Exception as e:
            self.logger.error(f"[DB] get_bot_trades failed: {e}")
            return []

    def get_bot_performance(self, bot_id: int, metric_type: str = None, hours: int = 24) -> List[Dict[str, Any]]:
        """Get performance metrics for a bot."""
        import json
        try:
            conn = sqlite3.connect(str(self.db_path))
            cur = conn.cursor()
            query = f"SELECT * FROM bot_performance WHERE bot_id = ? AND timestamp > datetime('now', '-{hours} hours')"
            params: List[Any] = [bot_id]
            if metric_type:
                query += " AND metric_type = ?"
                params.append(metric_type)
            query += " ORDER BY timestamp DESC"
            cur.execute(query, params)
            rows = cur.fetchall()
            conn.close()
            results = []
            for r in rows:
                meta = r[5]
                if meta:
                    try:
                        meta = json.loads(meta)
                    except:
                        pass
                results.append({"id": r[0], "bot_id": r[1], "timestamp": r[2], "metric_type": r[3], "value": r[4], "metadata": meta})
            return results
        except Exception as e:
            self.logger.error(f"[DB] get_bot_performance failed: {e}")
            return []

    def get_leaderboard(self, metric_type: str = "total_pnl", period_hours: int = 24, limit: int = 20) -> List[Dict[str, Any]]:
        """Get bot leaderboard ranked by a metric."""
        try:
            conn = sqlite3.connect(str(self.db_path))
            cur = conn.cursor()

            if metric_type == "total_pnl":
                cur.execute(f"""
                    SELECT b.id, b.name, b.owner, b.bot_type,
                           COALESCE(SUM(t.pnl), 0) as total_pnl,
                           COUNT(t.id) as trade_count,
                           COALESCE(AVG(t.pnl_pct), 0) as avg_pnl_pct
                    FROM bots b
                    LEFT JOIN bot_trades t ON b.id = t.bot_id AND t.timestamp > datetime('now', '-{period_hours} hours')
                    WHERE b.status = 'active'
                    GROUP BY b.id
                    ORDER BY total_pnl DESC
                    LIMIT ?
                """, (limit,))
            elif metric_type == "win_rate":
                cur.execute(f"""
                    SELECT b.id, b.name, b.owner, b.bot_type,
                           CAST(SUM(CASE WHEN t.pnl > 0 THEN 1 ELSE 0 END) AS REAL) / NULLIF(COUNT(t.id), 0) * 100 as win_rate,
                           COUNT(t.id) as trade_count,
                           COALESCE(SUM(t.pnl), 0) as total_pnl
                    FROM bots b
                    LEFT JOIN bot_trades t ON b.id = t.bot_id AND t.timestamp > datetime('now', '-{period_hours} hours')
                    WHERE b.status = 'active'
                    GROUP BY b.id
                    HAVING trade_count > 0
                    ORDER BY win_rate DESC
                    LIMIT ?
                """, (limit,))
            else:
                cur.execute(f"""
                    SELECT b.id, b.name, b.owner, b.bot_type,
                           COALESCE(AVG(p.value), 0) as metric_value,
                           COUNT(p.id) as data_points
                    FROM bots b
                    LEFT JOIN bot_performance p ON b.id = p.bot_id
                        AND p.metric_type = ? AND p.timestamp > datetime('now', '-{period_hours} hours')
                    WHERE b.status = 'active'
                    GROUP BY b.id
                    ORDER BY metric_value DESC
                    LIMIT ?
                """, (metric_type, limit))

            rows = cur.fetchall()
            conn.close()

            if metric_type == "total_pnl":
                return [{"rank": i+1, "bot_id": r[0], "name": r[1], "owner": r[2], "bot_type": r[3],
                         "total_pnl": round(r[4], 2), "trade_count": r[5], "avg_pnl_pct": round(r[6], 2)} for i, r in enumerate(rows)]
            elif metric_type == "win_rate":
                return [{"rank": i+1, "bot_id": r[0], "name": r[1], "owner": r[2], "bot_type": r[3],
                         "win_rate": round(r[4], 1) if r[4] else 0, "trade_count": r[5], "total_pnl": round(r[6], 2)} for i, r in enumerate(rows)]
            else:
                return [{"rank": i+1, "bot_id": r[0], "name": r[1], "owner": r[2], "bot_type": r[3],
                         "metric_value": round(r[4], 2), "data_points": r[5]} for i, r in enumerate(rows)]
        except Exception as e:
            self.logger.error(f"[DB] get_leaderboard failed: {e}")
            return []

    def get_bot_summary(self, bot_id: int, hours: int = 24) -> Dict[str, Any]:
        """Get summary stats for a bot."""
        try:
            conn = sqlite3.connect(str(self.db_path))
            cur = conn.cursor()

            cur.execute(f"""
                SELECT COUNT(*), COALESCE(SUM(pnl), 0), COALESCE(AVG(pnl_pct), 0),
                       SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END),
                       SUM(CASE WHEN pnl < 0 THEN 1 ELSE 0 END)
                FROM bot_trades WHERE bot_id = ? AND timestamp > datetime('now', '-{hours} hours')
            """, (bot_id,))
            trade_row = cur.fetchone()

            cur.execute(f"""
                SELECT metric_type, AVG(value) FROM bot_performance
                WHERE bot_id = ? AND timestamp > datetime('now', '-{hours} hours')
                GROUP BY metric_type
            """, (bot_id,))
            metrics = {r[0]: round(r[1], 4) for r in cur.fetchall()}

            conn.close()

            total_trades = trade_row[0] or 0
            wins = trade_row[3] or 0

            return {
                "period_hours": hours,
                "total_trades": total_trades,
                "total_pnl": round(trade_row[1] or 0, 2),
                "avg_pnl_pct": round(trade_row[2] or 0, 2),
                "wins": wins,
                "losses": trade_row[4] or 0,
                "win_rate": round((wins / total_trades * 100) if total_trades > 0 else 0, 1),
                "metrics": metrics,
            }
        except Exception as e:
            self.logger.error(f"[DB] get_bot_summary failed: {e}")
            return {}

    # ==================== BOT CONFIG MANAGEMENT ====================

    def save_bot_config(self, bot_id: int, config: Dict, config_hash: str,
                        config_version: int, previous_hash: str = None,
                        config_summary: Dict = None) -> bool:
        """Save a new config version for a bot."""
        import json
        try:
            conn = sqlite3.connect(str(self.db_path))
            cur = conn.cursor()

            # Update current config on bot
            cur.execute(
                """UPDATE bots SET config = ?, config_hash = ?, config_version = ?, config_summary = ?
                   WHERE id = ?""",
                (json.dumps(config), config_hash, config_version,
                 json.dumps(config_summary) if config_summary else None, bot_id)
            )

            # Add to history
            cur.execute(
                """INSERT INTO bot_config_history
                   (bot_id, timestamp, config, config_hash, config_version, previous_hash)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (bot_id, datetime.utcnow().isoformat(), json.dumps(config),
                 config_hash, config_version, previous_hash)
            )

            conn.commit()
            conn.close()
            return True
        except Exception as e:
            self.logger.error(f"[DB] save_bot_config failed: {e}")
            return False

    def get_bot_config(self, bot_id: int) -> Dict[str, Any] | None:
        """Get current config for a bot."""
        import json
        try:
            conn = sqlite3.connect(str(self.db_path))
            cur = conn.cursor()
            cur.execute(
                "SELECT config, config_hash, config_version, config_summary FROM bots WHERE id = ?",
                (bot_id,)
            )
            row = cur.fetchone()
            conn.close()

            if row:
                config = row[0]
                if config:
                    try:
                        config = json.loads(config)
                    except:
                        pass
                config_summary = row[3]
                if config_summary:
                    try:
                        config_summary = json.loads(config_summary)
                    except:
                        pass
                return {
                    "config": config,
                    "config_hash": row[1],
                    "config_version": row[2],
                    "config_summary": config_summary,
                }
            return None
        except Exception as e:
            self.logger.error(f"[DB] get_bot_config failed: {e}")
            return None

    def get_bot_config_history(self, bot_id: int, limit: int = 50) -> List[Dict[str, Any]]:
        """Get config version history for a bot."""
        import json
        try:
            conn = sqlite3.connect(str(self.db_path))
            cur = conn.cursor()
            cur.execute(
                """SELECT id, timestamp, config, config_hash, config_version, previous_hash, change_summary
                   FROM bot_config_history WHERE bot_id = ? ORDER BY timestamp DESC LIMIT ?""",
                (bot_id, limit)
            )
            rows = cur.fetchall()
            conn.close()

            results = []
            for r in rows:
                config = r[2]
                if config:
                    try:
                        config = json.loads(config)
                    except:
                        pass
                change_summary = r[6]
                if change_summary:
                    try:
                        change_summary = json.loads(change_summary)
                    except:
                        pass
                results.append({
                    "id": r[0],
                    "timestamp": r[1],
                    "config": config,
                    "config_hash": r[3],
                    "config_version": r[4],
                    "previous_hash": r[5],
                    "change_summary": change_summary,
                })
            return results
        except Exception as e:
            self.logger.error(f"[DB] get_bot_config_history failed: {e}")
            return []

    def get_config_by_version(self, bot_id: int, version: int) -> Dict[str, Any] | None:
        """Get a specific config version for a bot."""
        import json
        try:
            conn = sqlite3.connect(str(self.db_path))
            cur = conn.cursor()
            cur.execute(
                """SELECT timestamp, config, config_hash FROM bot_config_history
                   WHERE bot_id = ? AND config_version = ?""",
                (bot_id, version)
            )
            row = cur.fetchone()
            conn.close()

            if row:
                config = row[1]
                if config:
                    try:
                        config = json.loads(config)
                    except:
                        pass
                return {
                    "timestamp": row[0],
                    "config": config,
                    "config_hash": row[2],
                    "config_version": version,
                }
            return None
        except Exception as e:
            self.logger.error(f"[DB] get_config_by_version failed: {e}")
            return None

    def export_bot_config(self, bot_id: int) -> Dict[str, Any] | None:
        """Export a bot's current config for replication.

        Returns a dict suitable for creating a new bot with identical settings.
        """
        import json
        try:
            conn = sqlite3.connect(str(self.db_path))
            cur = conn.cursor()
            cur.execute(
                """SELECT b.name, b.config, b.config_hash, b.config_version,
                          (SELECT COUNT(*) FROM bot_trades WHERE bot_id = b.id) as trade_count,
                          (SELECT COALESCE(SUM(pnl), 0) FROM bot_trades WHERE bot_id = b.id) as total_pnl
                   FROM bots b WHERE b.id = ?""",
                (bot_id,)
            )
            row = cur.fetchone()
            conn.close()

            if row:
                config = row[1]
                if config:
                    try:
                        config = json.loads(config)
                    except:
                        pass
                return {
                    "source_bot_name": row[0],
                    "config": config,
                    "config_hash": row[2],
                    "config_version": row[3],
                    "source_stats": {
                        "trade_count": row[4],
                        "total_pnl": round(row[5], 2) if row[5] else 0,
                    },
                    "exported_at": datetime.utcnow().isoformat() + "Z",
                }
            return None
        except Exception as e:
            self.logger.error(f"[DB] export_bot_config failed: {e}")
            return None

    def update_bot_config(self, bot_id: int, config: Dict, config_hash: str = None,
                          config_summary: Dict = None) -> bool:
        """Update bot's current config (for initial registration update)."""
        import json
        try:
            conn = sqlite3.connect(str(self.db_path))
            cur = conn.cursor()
            cur.execute(
                """UPDATE bots SET config = ?, config_hash = ?, config_summary = ?
                   WHERE id = ?""",
                (json.dumps(config), config_hash,
                 json.dumps(config_summary) if config_summary else None, bot_id)
            )
            conn.commit()
            conn.close()
            return True
        except Exception as e:
            self.logger.error(f"[DB] update_bot_config failed: {e}")
            return False

    def get_sentiment_summary(self, symbol: str | None = None, hours: int = 24) -> Dict[str, Any]:
        """Get sentiment summary statistics."""
        try:
            conn = sqlite3.connect(str(self.db_path))
            cur = conn.cursor()

            base_query = f"FROM sentiment WHERE timestamp > datetime('now', '-{hours} hours')"
            if symbol:
                base_query += f" AND symbol = ?"
                params: Tuple = (symbol,)
            else:
                params = ()

            cur.execute(f"SELECT COUNT(*), AVG(value), MIN(value), MAX(value) {base_query}", params)
            row = cur.fetchone() or (0, None, None, None)

            cur.execute(f"SELECT sentiment_type, COUNT(*), AVG(value) {base_query} GROUP BY sentiment_type", params)
            by_type = {r[0]: {"count": r[1], "avg": r[2]} for r in cur.fetchall()}

            conn.close()

            return {
                "period_hours": hours,
                "symbol": symbol,
                "total_records": row[0],
                "avg_value": round(row[1], 2) if row[1] else None,
                "min_value": row[2],
                "max_value": row[3],
                "by_type": by_type,
            }
        except Exception as e:
            self.logger.error(f"[DB] get_sentiment_summary failed: {e}")
            return {"error": str(e)}
