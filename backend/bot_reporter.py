#!/usr/bin/env python3
"""
Bot Reporter - Reports trades and metrics to Data Manager

Integrates with the remote Data Manager server to:
- Register the bot on startup with FULL configuration capture
- Report trades as they happen (non-blocking async queue)
- Report performance metrics periodically
- Track configuration changes and report new versions
- Enable leaderboard tracking and config replication

Usage:
    from backend.bot_reporter import BotReporter

    reporter = BotReporter(config)
    reporter.register()  # Call once on startup - sends full sanitized config

    # Report trades (non-blocking - uses async queue)
    reporter.report_trade_async(
        symbol="SPY",
        action="BUY",
        quantity=1,
        price=450.50,
        pnl=25.00,
        pnl_pct=5.5,
        metadata={"signal": "BUY_CALL", "confidence": 0.72}
    )

    # Report metrics (non-blocking)
    reporter.report_metrics_async(
        total_pnl=1250.00,
        win_rate=0.65,
        sharpe_ratio=1.45,
        max_drawdown=-8.5
    )

    # Check for config changes (call each cycle)
    reporter.check_config_changed(config)
"""

import os
import json
import logging
import queue
import threading
import time
from datetime import datetime
from typing import Dict, Any, Optional, List, Tuple

logger = logging.getLogger(__name__)

# Import config schema for full config capture
try:
    from backend.config_schema import (
        prepare_config_for_reporting,
        compute_config_hash,
        get_config_summary,
    )
    HAS_CONFIG_SCHEMA = True
except ImportError:
    HAS_CONFIG_SCHEMA = False
    logger.debug("[REPORTER] config_schema not available, using minimal config capture")

# Try to import requests
try:
    import requests
except ImportError:
    requests = None


class BotReporter:
    """Reports bot activity to Data Manager server with async queue."""

    # Queue item types
    QUEUE_TRADE = 'trade'
    QUEUE_METRIC = 'metric'
    QUEUE_CONFIG = 'config'

    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the bot reporter.

        Args:
            config: Configuration dict with data_manager and bot settings
        """
        self.config = config or {}
        self.enabled = False
        self.bot_id = None
        self.session = None

        # Get Data Manager config
        dm_config = self.config.get('data_manager', {})
        self.base_url = dm_config.get('base_url', '').rstrip('/')
        self.api_key = dm_config.get('api_key', '')

        # Get bot identity config
        bot_config = self.config.get('bot_reporter', {})
        self.bot_name = bot_config.get('name', 'GaussianBot')
        self.bot_owner = bot_config.get('owner', 'default')
        self.bot_description = bot_config.get('description', 'Gaussian Options Trading Bot')
        self.report_interval = bot_config.get('metrics_interval_seconds', 300)  # 5 min default

        # State file to persist bot_id and config hash across restarts
        self.state_file = bot_config.get('state_file', 'data/bot_reporter_state.json')

        # Config tracking
        self.config_hash: Optional[str] = None
        self.config_version: int = 1

        # Async reporting queue (non-blocking)
        self._queue: queue.Queue = queue.Queue(maxsize=1000)
        self._queue_thread: Optional[threading.Thread] = None
        self._queue_stop = threading.Event()

        # Retry settings for failed reports
        self._max_retries = 3
        self._retry_delay = 1.0  # seconds

        # Metrics buffer for batching
        self._trade_buffer: List[Dict] = []
        self._buffer_lock = threading.Lock()

        # Background thread for periodic reporting
        self._report_thread = None
        self._stop_event = threading.Event()

        # Statistics
        self._stats = {
            'trades_queued': 0,
            'trades_sent': 0,
            'trades_failed': 0,
            'metrics_queued': 0,
            'metrics_sent': 0,
            'metrics_failed': 0,
            'config_updates': 0,
        }

        # Initialize if configured
        if self.base_url and self.api_key:
            self._init_session()
            self._load_state()
            self._start_queue_processor()
            self.enabled = True
            logger.info(f"[REPORTER] Initialized for {self.base_url}")
        else:
            logger.info("[REPORTER] Not configured (missing base_url or api_key)")

    def _init_session(self):
        """Initialize HTTP session."""
        if requests is None:
            logger.warning("[REPORTER] requests library not available")
            return

        self.session = requests.Session()
        self.session.headers['X-API-Key'] = self.api_key
        self.session.headers['Content-Type'] = 'application/json'
        self.session.timeout = 10

    def _load_state(self):
        """Load persisted state (bot_id, config_hash, config_version)."""
        try:
            if os.path.exists(self.state_file):
                with open(self.state_file, 'r') as f:
                    state = json.load(f)
                    self.bot_id = state.get('bot_id')
                    self.config_hash = state.get('config_hash')
                    self.config_version = state.get('config_version', 1)
                    if self.bot_id:
                        logger.info(f"[REPORTER] Loaded bot_id: {self.bot_id}, config_v{self.config_version}")
        except Exception as e:
            logger.debug(f"[REPORTER] Could not load state: {e}")

    def _save_state(self):
        """Persist state (bot_id, config_hash, config_version)."""
        try:
            os.makedirs(os.path.dirname(self.state_file), exist_ok=True)
            with open(self.state_file, 'w') as f:
                json.dump({
                    'bot_id': self.bot_id,
                    'config_hash': self.config_hash,
                    'config_version': self.config_version,
                }, f)
        except Exception as e:
            logger.debug(f"[REPORTER] Could not save state: {e}")

    def _start_queue_processor(self):
        """Start background thread to process async queue."""
        if self._queue_thread is not None:
            return

        self._queue_stop.clear()

        def process_queue():
            while not self._queue_stop.is_set():
                try:
                    # Wait for item with timeout
                    item = self._queue.get(timeout=1.0)
                    if item is None:
                        continue

                    item_type, data = item
                    success = False

                    # Process based on type with retries
                    for attempt in range(self._max_retries):
                        try:
                            if item_type == self.QUEUE_TRADE:
                                success = self._send_trade(data)
                            elif item_type == self.QUEUE_METRIC:
                                success = self._send_metric(data)
                            elif item_type == self.QUEUE_CONFIG:
                                success = self._send_config_update(data)

                            if success:
                                break
                            else:
                                time.sleep(self._retry_delay * (attempt + 1))
                        except Exception as e:
                            logger.debug(f"[REPORTER] Queue item failed (attempt {attempt + 1}): {e}")
                            time.sleep(self._retry_delay)

                    # Update stats
                    if item_type == self.QUEUE_TRADE:
                        if success:
                            self._stats['trades_sent'] += 1
                        else:
                            self._stats['trades_failed'] += 1
                    elif item_type == self.QUEUE_METRIC:
                        if success:
                            self._stats['metrics_sent'] += 1
                        else:
                            self._stats['metrics_failed'] += 1

                    self._queue.task_done()

                except queue.Empty:
                    continue
                except Exception as e:
                    logger.debug(f"[REPORTER] Queue processor error: {e}")

        self._queue_thread = threading.Thread(target=process_queue, daemon=True, name="BotReporterQueue")
        self._queue_thread.start()
        logger.debug("[REPORTER] Started async queue processor")

    def _stop_queue_processor(self):
        """Stop the queue processor thread."""
        self._queue_stop.set()
        if self._queue_thread:
            self._queue_thread.join(timeout=5)
            self._queue_thread = None

    def _request(self, method: str, endpoint: str, **kwargs) -> Optional[Dict]:
        """Make API request with error handling."""
        if not self.session:
            return None

        url = f"{self.base_url}{endpoint}"
        try:
            response = self.session.request(method, url, timeout=10, **kwargs)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.Timeout:
            logger.warning(f"[REPORTER] Request timeout: {endpoint}")
        except requests.exceptions.HTTPError as e:
            logger.warning(f"[REPORTER] HTTP error {e.response.status_code}: {endpoint}")
        except requests.exceptions.RequestException as e:
            logger.debug(f"[REPORTER] Request failed: {e}")
        except Exception as e:
            logger.debug(f"[REPORTER] Unexpected error: {e}")

        return None

    def _prepare_full_config(self) -> Tuple[Dict[str, Any], str]:
        """
        Prepare full sanitized config for reporting.

        Returns:
            Tuple of (config_dict, config_hash)
        """
        if HAS_CONFIG_SCHEMA:
            # Use full config schema (extracts trading-relevant sections, removes secrets)
            full_config = prepare_config_for_reporting(self.config)
            config_hash = compute_config_hash(full_config)
        else:
            # Fallback to minimal config
            full_config = {
                'entry_controller': self.config.get('entry_controller', {}).get('type', 'unknown'),
                'trading_symbol': self.config.get('trading', {}).get('symbol', 'SPY'),
                'initial_balance': self.config.get('trading', {}).get('initial_balance', 5000),
                'max_positions': self.config.get('trading', {}).get('max_positions', 3),
            }
            # Simple hash for fallback
            import hashlib
            config_hash = hashlib.sha256(
                json.dumps(full_config, sort_keys=True).encode()
            ).hexdigest()[:16]

        return full_config, config_hash

    def register(self) -> bool:
        """
        Register the bot with Data Manager.

        Sends FULL sanitized configuration for reproducibility.

        Returns:
            True if registration successful or already registered
        """
        if not self.enabled:
            return False

        # Prepare full config
        full_config, new_hash = self._prepare_full_config()

        # If we already have a bot_id, verify it's still valid
        if self.bot_id:
            result = self._request('GET', f'/api/v1/bots/{self.bot_id}')
            if result and result.get('id') == self.bot_id:
                # Check if config changed since last registration
                if self.config_hash and new_hash != self.config_hash:
                    logger.info(f"[REPORTER] Config changed (hash: {self.config_hash[:8]} -> {new_hash[:8]})")
                    self._report_config_change(full_config, new_hash)
                else:
                    logger.info(f"[REPORTER] Bot already registered: {self.bot_id}")
                return True
            else:
                logger.info("[REPORTER] Previous bot_id invalid, re-registering")
                self.bot_id = None

        # Store config hash
        self.config_hash = new_hash

        # Get config summary for quick viewing
        config_summary = {}
        if HAS_CONFIG_SCHEMA:
            config_summary = get_config_summary(full_config)

        # Register new bot with FULL config
        result = self._request('POST', '/api/v1/bots/register', json={
            'name': self.bot_name,
            'owner': self.bot_owner,
            'description': self.bot_description,
            'bot_type': 'trading',
            'config': full_config,
            'config_hash': self.config_hash,
            'config_version': self.config_version,
            'config_summary': config_summary,
        })

        if result and result.get('ok'):
            self.bot_id = result.get('bot_id')
            self._save_state()
            logger.info(f"[REPORTER] Bot registered: {self.bot_id} (config hash: {self.config_hash[:8]})")
            return True

        logger.warning("[REPORTER] Bot registration failed")
        return False

    def _report_config_change(self, new_config: Dict[str, Any], new_hash: str) -> bool:
        """
        Report a configuration change to the server.

        Args:
            new_config: New sanitized config
            new_hash: New config hash

        Returns:
            True if update successful
        """
        if not self.bot_id:
            return False

        self.config_version += 1
        old_hash = self.config_hash
        self.config_hash = new_hash

        result = self._request('POST', f'/api/v1/bots/{self.bot_id}/config', json={
            'config': new_config,
            'config_hash': new_hash,
            'config_version': self.config_version,
            'previous_hash': old_hash,
        })

        if result and result.get('ok'):
            self._save_state()
            self._stats['config_updates'] += 1
            logger.info(f"[REPORTER] Config updated to v{self.config_version}")
            return True

        # Rollback version on failure
        self.config_version -= 1
        self.config_hash = old_hash
        return False

    def check_config_changed(self, config: Dict[str, Any]) -> bool:
        """
        Check if config has changed and report if so.

        Call this periodically (e.g., each trading cycle) to detect
        hot-reloaded config changes.

        Args:
            config: Current full config dict

        Returns:
            True if config was changed and reported
        """
        if not self.enabled or not self.bot_id:
            return False

        # Update internal config reference
        self.config = config

        # Prepare and hash new config
        new_config, new_hash = self._prepare_full_config()

        if new_hash != self.config_hash:
            logger.info(f"[REPORTER] Detected config change: {self.config_hash[:8]} -> {new_hash[:8]}")
            return self._report_config_change(new_config, new_hash)

        return False

    def _send_trade(self, data: Dict[str, Any]) -> bool:
        """Internal: Send trade data to server (called by queue processor)."""
        if not self.bot_id:
            return False

        result = self._request('POST', f'/api/v1/bots/{self.bot_id}/trade', json=data)
        if result and result.get('ok'):
            trade_id = result.get('trade_id')
            logger.debug(f"[REPORTER] Trade sent: {trade_id}")
            return True
        return False

    def _send_metric(self, data: Dict[str, Any]) -> bool:
        """Internal: Send metric data to server (called by queue processor)."""
        if not self.bot_id:
            return False

        result = self._request('POST', f'/api/v1/bots/{self.bot_id}/metric', json=data)
        return result is not None and result.get('ok', False)

    def _send_config_update(self, data: Dict[str, Any]) -> bool:
        """Internal: Send config update to server (called by queue processor)."""
        if not self.bot_id:
            return False

        result = self._request('POST', f'/api/v1/bots/{self.bot_id}/config', json=data)
        return result is not None and result.get('ok', False)

    def _build_trade_data(
        self,
        symbol: str,
        action: str,
        quantity: float,
        price: float,
        pnl: Optional[float] = None,
        pnl_pct: Optional[float] = None,
        notes: Optional[str] = None,
        metadata: Optional[Dict] = None,
        config_version: Optional[int] = None
    ) -> Dict[str, Any]:
        """Build trade data dict."""
        data = {
            'symbol': symbol.upper(),
            'action': action.upper(),
            'quantity': quantity,
            'price': price,
            'timestamp': datetime.utcnow().isoformat() + 'Z',
        }

        if pnl is not None:
            data['pnl'] = pnl
        if pnl_pct is not None:
            data['pnl_pct'] = pnl_pct
        if notes:
            data['notes'] = notes
        if metadata:
            data['metadata'] = metadata
        if config_version is not None:
            data['config_version'] = config_version

        return data

    def report_trade(
        self,
        symbol: str,
        action: str,
        quantity: float,
        price: float,
        pnl: Optional[float] = None,
        pnl_pct: Optional[float] = None,
        notes: Optional[str] = None,
        metadata: Optional[Dict] = None
    ) -> bool:
        """
        Report a trade to Data Manager (BLOCKING).

        For non-blocking, use report_trade_async() instead.

        Args:
            symbol: Trading symbol (e.g., "SPY")
            action: Trade action (BUY, SELL, BUY_CALL, BUY_PUT, EXIT, etc.)
            quantity: Number of contracts/shares
            price: Execution price
            pnl: Profit/loss in dollars (for exits)
            pnl_pct: Profit/loss percentage (for exits)
            notes: Optional trade notes
            metadata: Additional metadata (signal type, confidence, etc.)

        Returns:
            True if trade reported successfully
        """
        if not self.enabled or not self.bot_id:
            return False

        data = self._build_trade_data(
            symbol, action, quantity, price,
            pnl, pnl_pct, notes, metadata,
            config_version=self.config_version
        )

        return self._send_trade(data)

    def report_trade_async(
        self,
        symbol: str,
        action: str,
        quantity: float,
        price: float,
        pnl: Optional[float] = None,
        pnl_pct: Optional[float] = None,
        notes: Optional[str] = None,
        metadata: Optional[Dict] = None
    ) -> bool:
        """
        Report a trade to Data Manager (NON-BLOCKING).

        Trade is queued and sent by background thread.

        Args:
            symbol: Trading symbol (e.g., "SPY")
            action: Trade action (BUY, SELL, BUY_CALL, BUY_PUT, EXIT, etc.)
            quantity: Number of contracts/shares
            price: Execution price
            pnl: Profit/loss in dollars (for exits)
            pnl_pct: Profit/loss percentage (for exits)
            notes: Optional trade notes
            metadata: Additional metadata (signal type, confidence, etc.)

        Returns:
            True if trade was queued successfully
        """
        if not self.enabled or not self.bot_id:
            return False

        data = self._build_trade_data(
            symbol, action, quantity, price,
            pnl, pnl_pct, notes, metadata,
            config_version=self.config_version
        )

        try:
            self._queue.put_nowait((self.QUEUE_TRADE, data))
            self._stats['trades_queued'] += 1
            return True
        except queue.Full:
            logger.warning("[REPORTER] Trade queue full, dropping trade")
            return False

    def report_metric(
        self,
        metric_type: str,
        value: float,
        metadata: Optional[Dict] = None
    ) -> bool:
        """
        Report a single performance metric.

        Args:
            metric_type: Metric name (e.g., "sharpe_ratio", "win_rate", "max_drawdown")
            value: Metric value
            metadata: Additional context

        Returns:
            True if metric reported successfully
        """
        if not self.enabled or not self.bot_id:
            return False

        data = {
            'metric_type': metric_type,
            'value': value,
        }
        if metadata:
            data['metadata'] = metadata

        result = self._request('POST', f'/api/v1/bots/{self.bot_id}/metric', json=data)
        return result is not None and result.get('ok', False)

    def _build_metrics_dict(
        self,
        total_pnl: Optional[float] = None,
        win_rate: Optional[float] = None,
        sharpe_ratio: Optional[float] = None,
        max_drawdown: Optional[float] = None,
        total_trades: Optional[int] = None,
        current_balance: Optional[float] = None,
        **extra_metrics
    ) -> Dict[str, float]:
        """Build metrics dict from arguments."""
        metrics = {}
        if total_pnl is not None:
            metrics['total_pnl'] = total_pnl
        if win_rate is not None:
            metrics['win_rate'] = win_rate
        if sharpe_ratio is not None:
            metrics['sharpe_ratio'] = sharpe_ratio
        if max_drawdown is not None:
            metrics['max_drawdown'] = max_drawdown
        if total_trades is not None:
            metrics['total_trades'] = total_trades
        if current_balance is not None:
            metrics['current_balance'] = current_balance
        metrics.update(extra_metrics)
        return metrics

    def report_metrics(
        self,
        total_pnl: Optional[float] = None,
        win_rate: Optional[float] = None,
        sharpe_ratio: Optional[float] = None,
        max_drawdown: Optional[float] = None,
        total_trades: Optional[int] = None,
        current_balance: Optional[float] = None,
        **extra_metrics
    ) -> int:
        """
        Report multiple performance metrics at once (BLOCKING).

        For non-blocking, use report_metrics_async() instead.

        Args:
            total_pnl: Total profit/loss in dollars
            win_rate: Win rate as decimal (0.65 = 65%)
            sharpe_ratio: Sharpe ratio
            max_drawdown: Maximum drawdown as percentage
            total_trades: Total number of trades
            current_balance: Current account balance
            **extra_metrics: Any additional metrics

        Returns:
            Number of metrics successfully reported
        """
        if not self.enabled or not self.bot_id:
            return 0

        metrics = self._build_metrics_dict(
            total_pnl, win_rate, sharpe_ratio, max_drawdown,
            total_trades, current_balance, **extra_metrics
        )

        count = 0
        for metric_type, value in metrics.items():
            if self.report_metric(metric_type, value):
                count += 1

        if count > 0:
            logger.debug(f"[REPORTER] Reported {count} metrics")

        return count

    def report_metrics_async(
        self,
        total_pnl: Optional[float] = None,
        win_rate: Optional[float] = None,
        sharpe_ratio: Optional[float] = None,
        max_drawdown: Optional[float] = None,
        total_trades: Optional[int] = None,
        current_balance: Optional[float] = None,
        **extra_metrics
    ) -> int:
        """
        Report multiple performance metrics (NON-BLOCKING).

        Metrics are queued and sent by background thread.

        Args:
            total_pnl: Total profit/loss in dollars
            win_rate: Win rate as decimal (0.65 = 65%)
            sharpe_ratio: Sharpe ratio
            max_drawdown: Maximum drawdown as percentage
            total_trades: Total number of trades
            current_balance: Current account balance
            **extra_metrics: Any additional metrics

        Returns:
            Number of metrics queued successfully
        """
        if not self.enabled or not self.bot_id:
            return 0

        metrics = self._build_metrics_dict(
            total_pnl, win_rate, sharpe_ratio, max_drawdown,
            total_trades, current_balance, **extra_metrics
        )

        count = 0
        for metric_type, value in metrics.items():
            data = {'metric_type': metric_type, 'value': value}
            try:
                self._queue.put_nowait((self.QUEUE_METRIC, data))
                self._stats['metrics_queued'] += 1
                count += 1
            except queue.Full:
                logger.warning(f"[REPORTER] Metrics queue full, dropping {metric_type}")

        return count

    def get_summary(self, hours: int = 24) -> Optional[Dict]:
        """
        Get bot performance summary from Data Manager.

        Args:
            hours: Time period in hours

        Returns:
            Summary dict or None
        """
        if not self.enabled or not self.bot_id:
            return None

        return self._request('GET', f'/api/v1/bots/{self.bot_id}/summary', params={'hours': hours})

    def get_leaderboard(self, metric: str = 'total_pnl', hours: int = 24, limit: int = 20) -> List[Dict]:
        """
        Get bot leaderboard from Data Manager.

        Args:
            metric: Ranking metric (total_pnl, win_rate)
            hours: Time period
            limit: Max results

        Returns:
            List of ranked bots
        """
        if not self.enabled:
            return []

        result = self._request('GET', '/api/v1/leaderboard', params={
            'metric': metric,
            'hours': hours,
            'limit': limit
        })

        return result.get('leaderboard', []) if result else []

    def start_periodic_reporting(self, callback=None):
        """
        Start background thread for periodic metrics reporting.

        Args:
            callback: Function that returns metrics dict to report
        """
        if not self.enabled:
            return

        self._stop_event.clear()

        def report_loop():
            while not self._stop_event.wait(self.report_interval):
                try:
                    if callback:
                        metrics = callback()
                        if metrics:
                            self.report_metrics(**metrics)
                except Exception as e:
                    logger.debug(f"[REPORTER] Periodic report error: {e}")

        self._report_thread = threading.Thread(target=report_loop, daemon=True)
        self._report_thread.start()
        logger.info(f"[REPORTER] Started periodic reporting (every {self.report_interval}s)")

    def stop_periodic_reporting(self):
        """Stop background reporting thread."""
        self._stop_event.set()
        if self._report_thread:
            self._report_thread.join(timeout=5)
            self._report_thread = None

    def get_config(self) -> Optional[Dict]:
        """
        Get the bot's current config from Data Manager.

        Returns:
            Config dict or None
        """
        if not self.enabled or not self.bot_id:
            return None

        return self._request('GET', f'/api/v1/bots/{self.bot_id}/config')

    def get_config_history(self) -> List[Dict]:
        """
        Get the bot's config version history.

        Returns:
            List of config versions with timestamps
        """
        if not self.enabled or not self.bot_id:
            return []

        result = self._request('GET', f'/api/v1/bots/{self.bot_id}/config/history')
        return result.get('history', []) if result else []

    def get_stats(self) -> Dict[str, int]:
        """
        Get reporter statistics.

        Returns:
            Dict with queue and send statistics
        """
        return {
            **self._stats,
            'queue_size': self._queue.qsize(),
            'enabled': self.enabled,
            'bot_id': self.bot_id,
            'config_version': self.config_version,
            'config_hash': self.config_hash[:8] if self.config_hash else None,
        }

    def close(self):
        """Cleanup resources."""
        self.stop_periodic_reporting()
        self._stop_queue_processor()

        # Wait for queue to drain (max 5 seconds)
        try:
            self._queue.join()
        except Exception:
            pass

        if self.session:
            self.session.close()

        logger.info(f"[REPORTER] Closed. Stats: {self._stats}")


# Singleton instance for global access
_reporter_instance: Optional[BotReporter] = None


def get_reporter() -> Optional[BotReporter]:
    """Get the global reporter instance."""
    return _reporter_instance


def init_reporter(config: Dict) -> BotReporter:
    """Initialize the global reporter instance."""
    global _reporter_instance
    _reporter_instance = BotReporter(config)
    return _reporter_instance


__all__ = [
    'BotReporter',
    'get_reporter',
    'init_reporter',
    'HAS_CONFIG_SCHEMA',
]
