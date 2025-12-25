"""
Data Manager Python Client SDK

Use this client to connect your trading bots to the Data Manager API.

Example usage:
    from client import DataManagerClient

    # Initialize client
    client = DataManagerClient(
        base_url="http://your-server:5050",
        api_key="dm_your_api_key_here"
    )

    # Get historical prices for training
    prices = client.get_prices("SPY", start="2025-01-01", limit=50000)

    # Get latest prices
    latest = client.get_latest_prices(["SPY", "QQQ", "VIX"])

    # Get options data
    options = client.get_options("SPY", option_type="CALL")

    # Get sentiment
    sentiment = client.get_sentiment(symbol="SPY", limit=100)

    # Post sentiment from your analysis bot
    client.post_sentiment(
        symbol="SPY",
        sentiment_type="bullish",
        value=65,
        headline="Market showing strength...",
        confidence=0.85,
        model="my-model"
    )

    # Real-time streaming (WebSocket)
    def on_price_update(data):
        print(f"Price update: {data}")

    def on_sentiment_update(data):
        print(f"Sentiment update: {data}")

    client.subscribe_prices(on_price_update)
    client.subscribe_sentiment(on_sentiment_update)
    client.listen()  # Blocking
"""

from __future__ import annotations

import json
from typing import Any, Callable, Dict, List, Optional
from datetime import datetime

try:
    import requests
except ImportError:
    requests = None

try:
    import socketio
except ImportError:
    socketio = None


class DataManagerClient:
    """Client for connecting trading bots to Data Manager."""

    def __init__(self, base_url: str, api_key: str):
        """
        Initialize the client.

        Args:
            base_url: The Data Manager server URL (e.g., "http://localhost:5050")
            api_key: Your API key (generate from dashboard)
        """
        if requests is None:
            raise ImportError("requests library required: pip install requests")

        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.session = requests.Session()
        self.session.headers["X-API-Key"] = api_key
        self.session.headers["Content-Type"] = "application/json"

        self._sio: Optional[Any] = None
        self._callbacks: Dict[str, List[Callable]] = {
            "prices": [],
            "sentiment": [],
            "options": [],
        }

    def _request(self, method: str, endpoint: str, **kwargs) -> Dict[str, Any]:
        """Make an API request."""
        url = f"{self.base_url}{endpoint}"
        try:
            response = self.session.request(method, url, **kwargs)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.HTTPError as e:
            error_data = {}
            try:
                error_data = e.response.json()
            except Exception:
                pass
            raise APIError(
                f"API error: {e.response.status_code}",
                status_code=e.response.status_code,
                response=error_data,
            )
        except requests.exceptions.RequestException as e:
            raise ConnectionError(f"Connection error: {e}")

    # ==================== Price Data ====================

    def get_prices(
        self,
        symbol: str,
        start: Optional[str] = None,
        end: Optional[str] = None,
        limit: int = 10000,
    ) -> List[Dict[str, Any]]:
        """
        Get historical price data for a symbol.

        Args:
            symbol: Stock symbol (e.g., "SPY")
            start: Start datetime (ISO format)
            end: End datetime (ISO format)
            limit: Maximum records to return

        Returns:
            List of price records with timestamp, open, high, low, close, volume
        """
        params = {"limit": limit}
        if start:
            params["start"] = start
        if end:
            params["end"] = end

        result = self._request("GET", f"/api/v1/prices/{symbol.upper()}", params=params)
        return result.get("data", [])

    def get_latest_prices(
        self, symbols: Optional[List[str]] = None
    ) -> Dict[str, Dict[str, Any]]:
        """
        Get the latest price for each symbol.

        Args:
            symbols: List of symbols (None for all tracked symbols)

        Returns:
            Dict mapping symbol to latest price data
        """
        params = {}
        if symbols:
            params["symbols"] = ",".join(s.upper() for s in symbols)

        result = self._request("GET", "/api/v1/prices", params=params)
        return result.get("data", {})

    # ==================== Options Data ====================

    def get_options(
        self,
        symbol: str,
        start: Optional[str] = None,
        end: Optional[str] = None,
        option_type: Optional[str] = None,
        limit: int = 10000,
    ) -> List[Dict[str, Any]]:
        """
        Get historical options/liquidity data for a symbol.

        Args:
            symbol: Underlying symbol (e.g., "SPY")
            start: Start datetime (ISO format)
            end: End datetime (ISO format)
            option_type: "CALL" or "PUT" (None for both)
            limit: Maximum records to return

        Returns:
            List of options records with greeks, bid/ask, volume, etc.
        """
        params = {"limit": limit}
        if start:
            params["start"] = start
        if end:
            params["end"] = end
        if option_type:
            params["option_type"] = option_type.upper()

        result = self._request("GET", f"/api/v1/options/{symbol.upper()}", params=params)
        return result.get("data", [])

    # ==================== Sentiment Data ====================

    def get_sentiment(
        self,
        symbol: Optional[str] = None,
        sentiment_type: Optional[str] = None,
        source: Optional[str] = None,
        model: Optional[str] = None,
        start: Optional[str] = None,
        end: Optional[str] = None,
        limit: int = 1000,
    ) -> List[Dict[str, Any]]:
        """
        Get sentiment data.

        Args:
            symbol: Filter by symbol
            sentiment_type: Filter by type (bullish, bearish, etc.)
            source: Filter by source (reuters, twitter, etc.)
            model: Filter by model (finbert, gpt-4, etc.)
            start: Start datetime
            end: End datetime
            limit: Maximum records

        Returns:
            List of sentiment records
        """
        params = {"limit": limit}
        if symbol:
            params["symbol"] = symbol.upper()
        if sentiment_type:
            params["sentiment_type"] = sentiment_type
        if source:
            params["source"] = source
        if model:
            params["model"] = model
        if start:
            params["start"] = start
        if end:
            params["end"] = end

        result = self._request("GET", "/api/v1/sentiment", params=params)
        return result.get("data", [])

    def post_sentiment(
        self,
        value: float,
        symbol: Optional[str] = None,
        sentiment_type: str = "general",
        headline: Optional[str] = None,
        url: Optional[str] = None,
        confidence: Optional[float] = None,
        model: Optional[str] = None,
        source: Optional[str] = None,
        notes: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Post sentiment data from your bot.

        Args:
            value: Sentiment value (-100 to 100)
            symbol: Associated symbol (optional)
            sentiment_type: Type (bullish, bearish, neutral, fear, greed)
            headline: The analyzed headline text
            url: Source URL
            confidence: Model confidence (0.0 to 1.0)
            model: Model name that produced this
            source: Source name (defaults to API key name)
            notes: Additional notes
            metadata: Additional JSON metadata

        Returns:
            Response with record ID or duplicate indicator
        """
        data = {
            "value": value,
            "sentiment_type": sentiment_type,
        }
        if symbol:
            data["symbol"] = symbol.upper()
        if headline:
            data["headline"] = headline
        if url:
            data["url"] = url
        if confidence is not None:
            data["confidence"] = confidence
        if model:
            data["model"] = model
        if source:
            data["source"] = source
        if notes:
            data["notes"] = notes
        if metadata:
            data["metadata"] = metadata

        return self._request("POST", "/api/v1/sentiment", json=data)

    # ==================== Symbols ====================

    def get_symbols(self) -> List[str]:
        """Get list of tracked symbols."""
        result = self._request("GET", "/api/v1/symbols")
        return result.get("symbols", [])

    # ==================== WebSocket Streaming ====================

    def _ensure_socketio(self):
        """Initialize Socket.IO client if needed."""
        if socketio is None:
            raise ImportError("python-socketio required: pip install python-socketio")

        if self._sio is None:
            self._sio = socketio.Client()

            @self._sio.on("price_update")
            def on_price(data):
                for cb in self._callbacks["prices"]:
                    cb(data)

            @self._sio.on("sentiment_update")
            def on_sentiment(data):
                for cb in self._callbacks["sentiment"]:
                    cb(data)

            @self._sio.on("options_update")
            def on_options(data):
                for cb in self._callbacks["options"]:
                    cb(data)

    def connect_websocket(self):
        """Connect to WebSocket for real-time updates."""
        self._ensure_socketio()
        if not self._sio.connected:
            self._sio.connect(
                self.base_url,
                headers={"X-API-Key": self.api_key},
                transports=["websocket"],
            )

    def disconnect_websocket(self):
        """Disconnect from WebSocket."""
        if self._sio and self._sio.connected:
            self._sio.disconnect()

    def subscribe_prices(self, callback: Callable[[Dict[str, Any]], None]):
        """Subscribe to real-time price updates."""
        self._callbacks["prices"].append(callback)

    def subscribe_sentiment(self, callback: Callable[[Dict[str, Any]], None]):
        """Subscribe to real-time sentiment updates."""
        self._callbacks["sentiment"].append(callback)

    def subscribe_options(self, callback: Callable[[Dict[str, Any]], None]):
        """Subscribe to real-time options updates."""
        self._callbacks["options"].append(callback)

    def listen(self):
        """Start listening for WebSocket events (blocking)."""
        self.connect_websocket()
        self._sio.wait()

    # ==================== Utilities ====================

    def health_check(self) -> bool:
        """Check if the server is healthy."""
        try:
            result = self._request("GET", "/api/health")
            return result.get("ok", False)
        except Exception:
            return False

    def get_stats(self) -> Dict[str, Any]:
        """Get collection statistics."""
        return self._request("GET", "/api/v1/stats")

    # ==================== Bot Management ====================

    def register_bot(
        self,
        name: str,
        owner: str,
        description: str = "",
        bot_type: str = "trading",
        config: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Register a new bot.

        Args:
            name: Bot name
            owner: Owner/user identifier
            description: Bot description
            bot_type: Type (trading, analysis, etc.)
            config: Optional configuration dict

        Returns:
            Response with bot_id
        """
        data = {
            "name": name,
            "owner": owner,
            "description": description,
            "bot_type": bot_type,
        }
        if config:
            data["config"] = config
        return self._request("POST", "/api/v1/bots/register", json=data)

    def get_bot(self, bot_id: int) -> Dict[str, Any]:
        """Get bot details."""
        return self._request("GET", f"/api/v1/bots/{bot_id}")

    def get_bot_summary(self, bot_id: int, hours: int = 24) -> Dict[str, Any]:
        """Get bot performance summary."""
        return self._request("GET", f"/api/v1/bots/{bot_id}/summary", params={"hours": hours})

    def record_trade(
        self,
        bot_id: int,
        symbol: str,
        action: str,
        quantity: float,
        price: float,
        pnl: Optional[float] = None,
        pnl_pct: Optional[float] = None,
        notes: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Record a trade for a bot.

        Args:
            bot_id: Bot ID
            symbol: Symbol traded
            action: BUY or SELL
            quantity: Quantity traded
            price: Execution price
            pnl: Profit/loss in dollars
            pnl_pct: Profit/loss percentage
            notes: Optional notes
            metadata: Additional data

        Returns:
            Response with trade_id
        """
        data = {
            "symbol": symbol.upper(),
            "action": action.upper(),
            "quantity": quantity,
            "price": price,
        }
        if pnl is not None:
            data["pnl"] = pnl
        if pnl_pct is not None:
            data["pnl_pct"] = pnl_pct
        if notes:
            data["notes"] = notes
        if metadata:
            data["metadata"] = metadata
        return self._request("POST", f"/api/v1/bots/{bot_id}/trade", json=data)

    def get_trades(self, bot_id: int, limit: int = 100) -> List[Dict[str, Any]]:
        """Get trades for a bot."""
        result = self._request("GET", f"/api/v1/bots/{bot_id}/trades", params={"limit": limit})
        return result.get("trades", [])

    def record_metric(
        self,
        bot_id: int,
        metric_type: str,
        value: float,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Record a performance metric for a bot.

        Args:
            bot_id: Bot ID
            metric_type: Metric type (e.g., "sharpe_ratio", "max_drawdown")
            value: Metric value
            metadata: Additional data
        """
        data = {"metric_type": metric_type, "value": value}
        if metadata:
            data["metadata"] = metadata
        return self._request("POST", f"/api/v1/bots/{bot_id}/metric", json=data)

    def get_leaderboard(
        self,
        metric: str = "total_pnl",
        hours: int = 24,
        limit: int = 20,
    ) -> List[Dict[str, Any]]:
        """
        Get bot leaderboard.

        Args:
            metric: Ranking metric (total_pnl, win_rate)
            hours: Time period
            limit: Max results

        Returns:
            List of ranked bots
        """
        result = self._request(
            "GET", "/api/v1/leaderboard",
            params={"metric": metric, "hours": hours, "limit": limit}
        )
        return result.get("leaderboard", [])


class APIError(Exception):
    """API error with status code and response."""

    def __init__(self, message: str, status_code: int = 0, response: Dict = None):
        super().__init__(message)
        self.status_code = status_code
        self.response = response or {}


# ==================== Quick Test ====================

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 3:
        print("Usage: python client.py <base_url> <api_key>")
        print("Example: python client.py http://localhost:5050 dm_abc123...")
        sys.exit(1)

    base_url = sys.argv[1]
    api_key = sys.argv[2]

    client = DataManagerClient(base_url, api_key)

    print("Testing connection...")
    if client.health_check():
        print("Server is healthy")
    else:
        print("Server is not responding!")
        sys.exit(1)

    print("\nTracked symbols:")
    print(client.get_symbols())

    print("\nLatest prices:")
    prices = client.get_latest_prices()
    for symbol, data in prices.items():
        print(f"  {symbol}: ${data['close']:.2f}")

    print("\nStats:")
    stats = client.get_stats()
    if "note" in stats:
        print(f"  {stats['note']}")
    else:
        print(f"  Symbols: {len(stats.get('symbols', {}))}")
        print(f"  Price records (24h): {stats.get('collection', {}).get('price_records_24h', 0)}")

    print("\nClient test successful!")
