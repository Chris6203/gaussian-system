from __future__ import annotations

import os
import subprocess
from functools import wraps
from pathlib import Path

from flask import Flask, jsonify, request, render_template_string, session, redirect, url_for
from flask_cors import CORS

from .auth import UserStore, generate_secret_key
from .settings import load_config, save_config
from .dashboard_template import DASHBOARD_HTML


def create_app(base_dir: Path, settings, storage, logger) -> Flask:
    app = Flask(__name__)
    CORS(app, supports_credentials=True)

    # Setup secret key for sessions
    secret_key_file = base_dir / "data" / ".secret_key"
    if secret_key_file.exists():
        app.secret_key = secret_key_file.read_text().strip()
    else:
        secret_key_file.parent.mkdir(parents=True, exist_ok=True)
        app.secret_key = generate_secret_key()
        secret_key_file.write_text(app.secret_key)

    app.config["SESSION_COOKIE_HTTPONLY"] = True
    app.config["SESSION_COOKIE_SAMESITE"] = "Lax"

    # User store
    users = UserStore(base_dir / "data" / "users.json")
    env_path = base_dir / ".env"

    def login_required(f):
        @wraps(f)
        def decorated(*args, **kwargs):
            if not session.get("user"):
                if request.is_json or request.path.startswith("/api/"):
                    return jsonify({"error": "Unauthorized"}), 401
                return redirect(url_for("login"))
            return f(*args, **kwargs)
        return decorated

    def _read_env() -> dict:
        env_vars = {}
        if env_path.exists():
            with open(env_path, "r") as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#") and "=" in line:
                        key, val = line.split("=", 1)
                        env_vars[key.strip()] = val.strip()
        return env_vars

    def _write_env(env_vars: dict) -> None:
        lines = []
        existing_keys = set()
        if env_path.exists():
            with open(env_path, "r") as f:
                for line in f:
                    stripped = line.strip()
                    if stripped.startswith("#") or not stripped:
                        lines.append(line.rstrip())
                    elif "=" in stripped:
                        key = stripped.split("=", 1)[0].strip()
                        existing_keys.add(key)
                        if key in env_vars:
                            lines.append(f"{key}={env_vars[key]}")
                        else:
                            lines.append(line.rstrip())
        for key, val in env_vars.items():
            if key not in existing_keys:
                lines.append(f"{key}={val}")
        with open(env_path, "w") as f:
            f.write("\n".join(lines) + "\n")

    # ==================== AUTH ROUTES ====================

    @app.get("/login")
    def login():
        if not users.has_users():
            return redirect(url_for("setup"))
        if session.get("user"):
            return redirect(url_for("index"))
        return render_template_string(LOGIN_HTML)

    @app.post("/login")
    def login_post():
        data = request.form if request.form else request.get_json(force=True)
        email = data.get("email", "").strip()
        password = data.get("password", "")

        if users.verify_password(email, password):
            session["user"] = email
            if request.is_json:
                return jsonify({"ok": True})
            return redirect(url_for("index"))

        if request.is_json:
            return jsonify({"error": "Invalid credentials"}), 401
        return render_template_string(LOGIN_HTML, error="Invalid email or password")

    @app.get("/logout")
    def logout():
        session.pop("user", None)
        return redirect(url_for("login"))

    @app.get("/setup")
    def setup():
        if users.has_users():
            return redirect(url_for("login"))
        return render_template_string(SETUP_HTML)

    @app.post("/setup")
    def setup_post():
        if users.has_users():
            return redirect(url_for("login"))

        data = request.form if request.form else request.get_json(force=True)
        email = data.get("email", "").strip()
        password = data.get("password", "")
        confirm = data.get("confirm", "")

        if not email or not password:
            return render_template_string(SETUP_HTML, error="Email and password required")
        if password != confirm:
            return render_template_string(SETUP_HTML, error="Passwords do not match")
        if len(password) < 6:
            return render_template_string(SETUP_HTML, error="Password must be at least 6 characters")

        users.create_user(email, password)
        session["user"] = email
        return redirect(url_for("index"))

    @app.get("/api/auth/status")
    def auth_status():
        return jsonify({
            "authenticated": bool(session.get("user")),
            "user": session.get("user"),
            "setup_required": not users.has_users()
        })

    # ==================== PROTECTED API ROUTES ====================

    @app.get("/api/health")
    def health():
        return jsonify({"ok": True})

    @app.get("/api/config")
    @login_required
    def get_config():
        cfg = load_config(settings)
        t = cfg.get("credentials", {}).get("tradier", {})
        if t.get("data_api_token"):
            t["data_api_token"] = "***redacted***"
        for k in ("sandbox", "live"):
            if isinstance(t.get(k), dict) and t[k].get("access_token"):
                t[k]["access_token"] = "***redacted***"
        return jsonify(cfg)

    @app.post("/api/config")
    @login_required
    def post_config():
        cfg = request.get_json(force=True) or {}
        save_config(settings, cfg)
        return jsonify({"ok": True})

    @app.get("/api/symbols")
    @login_required
    def get_symbols():
        cfg = load_config(settings)
        symbols = cfg.get("data_fetching", {}).get("symbols", [])
        return jsonify({"symbols": symbols})

    @app.post("/api/symbols")
    @login_required
    def post_symbols():
        body = request.get_json(force=True) or {}
        symbols = body.get("symbols", [])
        cfg = load_config(settings)
        cfg.setdefault("data_fetching", {})["symbols"] = symbols
        save_config(settings, cfg)
        return jsonify({"ok": True, "symbols": symbols})

    @app.get("/api/collector")
    @login_required
    def get_collector():
        cfg = load_config(settings)
        collector = cfg.get("collector", {})
        return jsonify(collector)

    @app.post("/api/collector")
    @login_required
    def post_collector():
        body = request.get_json(force=True) or {}
        cfg = load_config(settings)
        cfg["collector"] = body
        save_config(settings, cfg)
        return jsonify({"ok": True})

    @app.get("/api/options_chain")
    @login_required
    def get_options_chain():
        cfg = load_config(settings)
        options = cfg.get("options_chain", {
            "enabled": False,
            "min_dte": 0,
            "max_dte": 45,
            "strike_range_pct": 10,
            "collect_greeks": True,
            "expirations_to_collect": 5,
            "symbols": ["SPY"]
        })
        return jsonify(options)

    @app.post("/api/options_chain")
    @login_required
    def post_options_chain():
        body = request.get_json(force=True) or {}
        cfg = load_config(settings)
        cfg["options_chain"] = body
        save_config(settings, cfg)
        return jsonify({"ok": True})

    @app.get("/api/credentials")
    @login_required
    def get_credentials():
        env_vars = _read_env()
        return jsonify({
            "tradier": {
                "data_api_token": "***" + env_vars.get("TRADIER_DATA_API_TOKEN", "")[-4:] if env_vars.get("TRADIER_DATA_API_TOKEN") else "",
                "access_token": "***" + env_vars.get("TRADIER_ACCESS_TOKEN", "")[-4:] if env_vars.get("TRADIER_ACCESS_TOKEN") else "",
                "account_number": env_vars.get("TRADIER_ACCOUNT_NUMBER", ""),
                "is_sandbox": env_vars.get("TRADIER_IS_SANDBOX", "true"),
            },
            "polygon": {
                "api_key": "***" + env_vars.get("POLYGON_API_KEY", "")[-4:] if env_vars.get("POLYGON_API_KEY") else "",
            }
        })

    @app.post("/api/credentials")
    @login_required
    def post_credentials():
        body = request.get_json(force=True) or {}
        env_vars = _read_env()

        tradier = body.get("tradier", {})
        if tradier.get("data_api_token") and not tradier["data_api_token"].startswith("***"):
            env_vars["TRADIER_DATA_API_TOKEN"] = tradier["data_api_token"]
        if tradier.get("access_token") and not tradier["access_token"].startswith("***"):
            env_vars["TRADIER_ACCESS_TOKEN"] = tradier["access_token"]
        if tradier.get("account_number"):
            env_vars["TRADIER_ACCOUNT_NUMBER"] = tradier["account_number"]
        if tradier.get("is_sandbox") is not None:
            env_vars["TRADIER_IS_SANDBOX"] = str(tradier["is_sandbox"]).lower()

        polygon = body.get("polygon", {})
        if polygon.get("api_key") and not polygon["api_key"].startswith("***"):
            env_vars["POLYGON_API_KEY"] = polygon["api_key"]

        _write_env(env_vars)
        return jsonify({"ok": True})

    @app.get("/api/stats")
    @login_required
    def stats():
        return jsonify(storage.stats())

    @app.get("/api/integrity")
    @login_required
    def integrity():
        return jsonify(storage.check_integrity())

    @app.post("/api/integrity/dedup")
    @login_required
    def dedup():
        out = storage.remove_duplicates()
        if (out.get("price_removed", 0) + out.get("liquidity_removed", 0)) > 0:
            storage.optimize()
        return jsonify(out)

    @app.post("/api/backfill")
    @login_required
    def backfill():
        body = request.get_json(force=True) or {}
        days = int(body.get("days", 7))
        cmd = [str(base_dir / "venv" / "bin" / "python"), str(base_dir / "run.py"), "backfill", str(days)]
        subprocess.Popen(cmd, cwd=str(base_dir), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return jsonify({"ok": True, "started": True, "days": days})

    # ==================== SENTIMENT API ====================

    @app.post("/api/sentiment")
    @login_required
    def post_sentiment():
        """Record a sentiment entry.

        Body: {
            "symbol": "SPY" (optional),
            "sentiment_type": "bullish|bearish|neutral|fear|greed|custom",
            "value": -100 to 100 (required),
            "source": "reuters|bloomberg|twitter|reddit|manual|etc" (optional),
            "notes": "free text" (optional),
            "metadata": {} (optional JSON object),
            "headline": "Fed raises rates..." (the analyzed text),
            "url": "https://..." (source URL),
            "confidence": 0.0-1.0 (model confidence),
            "model": "gpt-4|finbert|vader|etc" (which model produced this)
        }

        Duplicate headlines (same source + headline + symbol) are ignored.
        """
        body = request.get_json(force=True) or {}
        if "value" not in body:
            return jsonify({"error": "value is required"}), 400
        try:
            value = float(body["value"])
        except (ValueError, TypeError):
            return jsonify({"error": "value must be a number"}), 400

        record_id = storage.save_sentiment(body)
        if record_id:
            return jsonify({"ok": True, "id": record_id})
        return jsonify({"error": "Failed to save sentiment"}), 500

    @app.get("/api/sentiment")
    @login_required
    def get_sentiment():
        """Get sentiment records with optional filters.

        Query params: symbol, sentiment_type, source, model, start_time, end_time, limit
        """
        records = storage.get_sentiment(
            symbol=request.args.get("symbol"),
            sentiment_type=request.args.get("sentiment_type"),
            source=request.args.get("source"),
            model=request.args.get("model"),
            start_time=request.args.get("start_time"),
            end_time=request.args.get("end_time"),
            limit=int(request.args.get("limit", 100)),
        )
        return jsonify({"records": records, "count": len(records)})

    @app.get("/api/sentiment/summary")
    @login_required
    def get_sentiment_summary():
        """Get sentiment summary statistics.

        Query params: symbol (optional), hours (default 24)
        """
        summary = storage.get_sentiment_summary(
            symbol=request.args.get("symbol"),
            hours=int(request.args.get("hours", 24)),
        )
        return jsonify(summary)

    @app.post("/api/service/restart")
    @login_required
    def restart_service():
        body = request.get_json(force=True) or {}
        service = body.get("service", "collector")
        service_name = f"data_manager_{service}"
        try:
            subprocess.run(["systemctl", "restart", service_name], check=True)
            return jsonify({"ok": True, "service": service_name})
        except Exception as e:
            return jsonify({"ok": False, "error": str(e)}), 500

    @app.get("/api/service/status")
    @login_required
    def service_status():
        services = {}
        for svc in ["data_manager_collector", "data_manager_web"]:
            try:
                result = subprocess.run(["systemctl", "is-active", svc], capture_output=True, text=True)
                services[svc] = result.stdout.strip()
            except Exception:
                services[svc] = "unknown"
        return jsonify(services)

    @app.get("/")
    @login_required
    def index():
        return render_template_string(DASHBOARD_HTML, user=session.get("user"))

    # ==================== BOT API (API Key Auth) ====================

    def api_key_required(f):
        """Decorator for API key authentication (for bots)."""
        @wraps(f)
        def decorated(*args, **kwargs):
            api_key = request.headers.get("X-API-Key") or request.args.get("api_key")
            if not api_key:
                return jsonify({"error": "API key required"}), 401
            key_info = storage.validate_api_key(api_key)
            if not key_info:
                return jsonify({"error": "Invalid API key"}), 401
            request.api_key_info = key_info
            return f(*args, **kwargs)
        return decorated

    def api_key_or_login(f):
        """Decorator that allows either API key or session auth."""
        @wraps(f)
        def decorated(*args, **kwargs):
            # Check API key first
            api_key = request.headers.get("X-API-Key") or request.args.get("api_key")
            if api_key:
                key_info = storage.validate_api_key(api_key)
                if key_info:
                    request.api_key_info = key_info
                    return f(*args, **kwargs)
            # Fall back to session auth
            if session.get("user"):
                return f(*args, **kwargs)
            return jsonify({"error": "Authentication required"}), 401
        return decorated

    # API Key Management (dashboard only)
    @app.get("/api/keys")
    @login_required
    def list_api_keys():
        """List all API keys."""
        return jsonify({"keys": storage.list_api_keys()})

    @app.post("/api/keys")
    @login_required
    def create_api_key():
        """Create a new API key."""
        body = request.get_json(force=True) or {}
        name = body.get("name", "").strip()
        if not name:
            return jsonify({"error": "Name is required"}), 400
        permissions = body.get("permissions", "read,write")
        key = storage.create_api_key(name, permissions)
        if key:
            return jsonify({"ok": True, "key": key, "message": "Save this key - it won't be shown again!"})
        return jsonify({"error": "Failed to create key"}), 500

    @app.delete("/api/keys/<int:key_id>")
    @login_required
    def delete_api_key(key_id):
        """Delete an API key."""
        if storage.delete_api_key(key_id):
            return jsonify({"ok": True})
        return jsonify({"error": "Failed to delete key"}), 500

    @app.post("/api/keys/<int:key_id>/revoke")
    @login_required
    def revoke_api_key(key_id):
        """Revoke an API key."""
        if storage.revoke_api_key(key_id):
            return jsonify({"ok": True})
        return jsonify({"error": "Failed to revoke key"}), 500

    # Bot Data Endpoints
    @app.get("/api/v1/prices/<symbol>")
    @api_key_or_login
    def get_price_history(symbol):
        """Get historical price data for a symbol.

        Query params: start, end, limit (default 10000)
        """
        data = storage.get_price_history(
            symbol=symbol.upper(),
            start_time=request.args.get("start"),
            end_time=request.args.get("end"),
            limit=int(request.args.get("limit", 10000)),
        )
        return jsonify({"symbol": symbol.upper(), "count": len(data), "data": data})

    @app.get("/api/v1/prices")
    @api_key_or_login
    def get_latest_prices():
        """Get latest prices for all or specified symbols.

        Query params: symbols (comma-separated, optional)
        """
        symbols_param = request.args.get("symbols")
        symbols = [s.strip().upper() for s in symbols_param.split(",")] if symbols_param else None
        data = storage.get_latest_prices(symbols)
        return jsonify({"count": len(data), "data": data})

    @app.get("/api/v1/options/<symbol>")
    @api_key_or_login
    def get_options_history(symbol):
        """Get historical options/liquidity data for a symbol.

        Query params: start, end, option_type (CALL/PUT), limit (default 10000)
        """
        data = storage.get_liquidity_history(
            symbol=symbol.upper(),
            start_time=request.args.get("start"),
            end_time=request.args.get("end"),
            option_type=request.args.get("option_type"),
            limit=int(request.args.get("limit", 10000)),
        )
        return jsonify({"symbol": symbol.upper(), "count": len(data), "data": data})

    @app.get("/api/v1/sentiment")
    @api_key_or_login
    def get_sentiment_for_bots():
        """Get sentiment data for bots.

        Query params: symbol, sentiment_type, source, model, start, end, limit
        """
        data = storage.get_sentiment(
            symbol=request.args.get("symbol"),
            sentiment_type=request.args.get("sentiment_type"),
            source=request.args.get("source"),
            model=request.args.get("model"),
            start_time=request.args.get("start"),
            end_time=request.args.get("end"),
            limit=int(request.args.get("limit", 1000)),
        )
        return jsonify({"count": len(data), "data": data})

    @app.post("/api/v1/sentiment")
    @api_key_required
    def post_sentiment_from_bot():
        """Post sentiment from external bot.

        Requires 'write' permission in API key.
        """
        key_info = request.api_key_info
        if "write" not in key_info.get("permissions", []):
            return jsonify({"error": "Write permission required"}), 403

        body = request.get_json(force=True) or {}
        if "value" not in body:
            return jsonify({"error": "value is required"}), 400

        body["source"] = body.get("source") or key_info.get("name", "bot")
        record_id = storage.save_sentiment(body)
        if record_id:
            return jsonify({"ok": True, "id": record_id})
        if record_id is None:
            return jsonify({"ok": True, "duplicate": True, "message": "Duplicate entry ignored"})
        return jsonify({"error": "Failed to save sentiment"}), 500

    @app.get("/api/v1/symbols")
    @api_key_or_login
    def get_symbols_for_bots():
        """Get list of tracked symbols."""
        cfg = load_config(settings)
        symbols = cfg.get("data_fetching", {}).get("symbols", [])
        return jsonify({"symbols": symbols})

    @app.get("/api/v1/stats")
    @api_key_or_login
    def get_stats_for_bots():
        """Get collection statistics for bots."""
        return jsonify(storage.stats())

    # ==================== BOT MANAGEMENT API ====================

    @app.post("/api/v1/bots/register")
    @api_key_required
    def register_bot():
        """Register a new bot with full config capture.

        Body: {
            name: str (required),
            owner: str (required),
            description: str,
            bot_type: str,
            config: dict (full sanitized trading config),
            config_hash: str (16-char hash),
            config_version: int,
            config_summary: dict (key settings for quick viewing)
        }
        """
        key_info = request.api_key_info
        body = request.get_json(force=True) or {}

        name = body.get("name", "").strip()
        if not name:
            return jsonify({"error": "name is required"}), 400

        owner = body.get("owner", "").strip()
        if not owner:
            return jsonify({"error": "owner is required"}), 400

        bot_id = storage.register_bot(
            name=name,
            owner=owner,
            api_key_id=key_info.get("id"),
            description=body.get("description", ""),
            bot_type=body.get("bot_type", "trading"),
            config=body.get("config"),
            config_hash=body.get("config_hash"),
            config_version=body.get("config_version", 1),
            config_summary=body.get("config_summary"),
        )
        if bot_id:
            return jsonify({"ok": True, "bot_id": bot_id})
        return jsonify({"error": "Failed to register bot"}), 500

    @app.get("/api/v1/bots")
    @api_key_or_login
    def list_bots():
        """List all bots. Query param: owner (optional)."""
        owner = request.args.get("owner")
        bots = storage.list_bots(owner)
        return jsonify({"bots": bots, "count": len(bots)})

    @app.get("/api/v1/bots/<int:bot_id>")
    @api_key_or_login
    def get_bot(bot_id):
        """Get bot details."""
        bot = storage.get_bot(bot_id)
        if bot:
            return jsonify(bot)
        return jsonify({"error": "Bot not found"}), 404

    @app.get("/api/v1/bots/<int:bot_id>/summary")
    @api_key_or_login
    def get_bot_summary(bot_id):
        """Get bot performance summary."""
        hours = int(request.args.get("hours", 24))
        summary = storage.get_bot_summary(bot_id, hours)
        return jsonify(summary)

    @app.post("/api/v1/bots/<int:bot_id>/trade")
    @api_key_required
    def record_trade(bot_id):
        """Record a trade for a bot.

        Body: {symbol, action, quantity, price, pnl, pnl_pct, notes, metadata}
        """
        body = request.get_json(force=True) or {}

        required = ["symbol", "action", "quantity", "price"]
        for field in required:
            if field not in body:
                return jsonify({"error": f"{field} is required"}), 400

        trade_id = storage.record_bot_trade(
            bot_id=bot_id,
            symbol=body["symbol"],
            action=body["action"],
            quantity=float(body["quantity"]),
            price=float(body["price"]),
            pnl=float(body["pnl"]) if body.get("pnl") is not None else None,
            pnl_pct=float(body["pnl_pct"]) if body.get("pnl_pct") is not None else None,
            notes=body.get("notes"),
            metadata=body.get("metadata"),
        )
        if trade_id:
            return jsonify({"ok": True, "trade_id": trade_id})
        return jsonify({"error": "Failed to record trade"}), 500

    @app.get("/api/v1/bots/<int:bot_id>/trades")
    @api_key_or_login
    def get_bot_trades(bot_id):
        """Get trades for a bot."""
        limit = int(request.args.get("limit", 100))
        start = request.args.get("start")
        trades = storage.get_bot_trades(bot_id, limit, start)
        return jsonify({"trades": trades, "count": len(trades)})

    @app.post("/api/v1/bots/<int:bot_id>/metric")
    @api_key_required
    def record_metric(bot_id):
        """Record a performance metric for a bot.

        Body: {metric_type, value, metadata}
        """
        body = request.get_json(force=True) or {}

        if "metric_type" not in body or "value" not in body:
            return jsonify({"error": "metric_type and value are required"}), 400

        success = storage.record_bot_metric(
            bot_id=bot_id,
            metric_type=body["metric_type"],
            value=float(body["value"]),
            metadata=body.get("metadata"),
        )
        if success:
            return jsonify({"ok": True})
        return jsonify({"error": "Failed to record metric"}), 500

    @app.get("/api/v1/bots/<int:bot_id>/performance")
    @api_key_or_login
    def get_bot_performance(bot_id):
        """Get performance metrics for a bot."""
        hours = int(request.args.get("hours", 24))
        metric_type = request.args.get("metric_type")
        data = storage.get_bot_performance(bot_id, metric_type, hours)
        return jsonify({"data": data, "count": len(data)})

    # ==================== BOT CONFIG API ====================

    @app.post("/api/v1/bots/<int:bot_id>/config")
    @api_key_required
    def update_bot_config(bot_id):
        """Update bot config (creates new version).

        Body: {
            config: dict (required, full sanitized config),
            config_hash: str (required),
            config_version: int (required),
            previous_hash: str (optional),
            config_summary: dict (optional)
        }
        """
        body = request.get_json(force=True) or {}

        config = body.get("config")
        if not config:
            return jsonify({"error": "config is required"}), 400

        config_hash = body.get("config_hash")
        if not config_hash:
            return jsonify({"error": "config_hash is required"}), 400

        config_version = body.get("config_version")
        if config_version is None:
            return jsonify({"error": "config_version is required"}), 400

        success = storage.save_bot_config(
            bot_id=bot_id,
            config=config,
            config_hash=config_hash,
            config_version=config_version,
            previous_hash=body.get("previous_hash"),
            config_summary=body.get("config_summary"),
        )
        if success:
            return jsonify({"ok": True, "config_version": config_version})
        return jsonify({"error": "Failed to save config"}), 500

    @app.get("/api/v1/bots/<int:bot_id>/config")
    @api_key_or_login
    def get_bot_config(bot_id):
        """Get bot's current config.

        Returns: {config, config_hash, config_version, config_summary}
        """
        data = storage.get_bot_config(bot_id)
        if data:
            return jsonify(data)
        return jsonify({"error": "Bot or config not found"}), 404

    @app.get("/api/v1/bots/<int:bot_id>/config/history")
    @api_key_or_login
    def get_bot_config_history(bot_id):
        """Get bot's config version history.

        Query params: limit (default 50)
        """
        limit = int(request.args.get("limit", 50))
        history = storage.get_bot_config_history(bot_id, limit)
        return jsonify({"history": history, "count": len(history)})

    @app.get("/api/v1/bots/<int:bot_id>/config/version/<int:version>")
    @api_key_or_login
    def get_bot_config_version(bot_id, version):
        """Get a specific config version."""
        data = storage.get_config_by_version(bot_id, version)
        if data:
            return jsonify(data)
        return jsonify({"error": "Config version not found"}), 404

    @app.get("/api/v1/bots/<int:bot_id>/config/export")
    @api_key_or_login
    def export_bot_config(bot_id):
        """Export bot config for replication.

        Returns a downloadable JSON with config and source bot stats.
        """
        data = storage.export_bot_config(bot_id)
        if data:
            response = jsonify(data)
            response.headers["Content-Disposition"] = f"attachment; filename=bot_{bot_id}_config.json"
            return response
        return jsonify({"error": "Bot not found"}), 404

    @app.get("/api/v1/leaderboard")
    @api_key_or_login
    def get_leaderboard():
        """Get bot leaderboard.

        Query params: metric (total_pnl, win_rate), hours (default 24), limit (default 20)
        """
        metric = request.args.get("metric", "total_pnl")
        hours = int(request.args.get("hours", 24))
        limit = int(request.args.get("limit", 20))
        data = storage.get_leaderboard(metric, hours, limit)
        return jsonify({"metric": metric, "period_hours": hours, "leaderboard": data})

    # Dashboard endpoints for bot/leaderboard management
    @app.get("/api/bots")
    @login_required
    def list_bots_dashboard():
        """List all bots (dashboard)."""
        return jsonify({"bots": storage.list_bots()})

    @app.get("/api/leaderboard")
    @login_required
    def get_leaderboard_dashboard():
        """Get leaderboard (dashboard)."""
        metric = request.args.get("metric", "total_pnl")
        hours = int(request.args.get("hours", 24))
        return jsonify({"leaderboard": storage.get_leaderboard(metric, hours)})

    return app


# ==================== HTML TEMPLATES ====================

AUTH_STYLES = '''
<style>
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; background: #0f172a; color: #e2e8f0; min-height: 100vh; display: flex; align-items: center; justify-content: center; }
    .auth-card { background: #1e293b; border-radius: 16px; padding: 40px; width: 100%; max-width: 400px; }
    .auth-card h1 { color: #38bdf8; font-size: 24px; margin-bottom: 8px; text-align: center; }
    .auth-card p { color: #94a3b8; font-size: 14px; margin-bottom: 24px; text-align: center; }
    .form-group { margin-bottom: 16px; }
    .form-group label { display: block; font-size: 13px; color: #94a3b8; margin-bottom: 6px; }
    input { width: 100%; padding: 12px 14px; border: 1px solid #334155; border-radius: 8px; background: #0f172a; color: #e2e8f0; font-size: 14px; }
    input:focus { outline: none; border-color: #38bdf8; }
    button { width: 100%; padding: 12px; border: none; border-radius: 8px; background: #0284c7; color: white; font-size: 14px; font-weight: 600; cursor: pointer; margin-top: 8px; }
    button:hover { background: #0369a1; }
    .error { background: #7f1d1d; color: #fca5a5; padding: 10px 14px; border-radius: 8px; font-size: 13px; margin-bottom: 16px; }
    .logo { text-align: center; margin-bottom: 20px; font-size: 40px; }
</style>
'''

LOGIN_HTML = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Login - Data Manager</title>
''' + AUTH_STYLES + '''
</head>
<body>
    <div class="auth-card">
        <div class="logo">📊</div>
        <h1>Data Manager</h1>
        <p>Sign in to continue</p>
        {% if error %}
        <div class="error">{{ error }}</div>
        {% endif %}
        <form method="POST">
            <div class="form-group">
                <label>Email</label>
                <input type="email" name="email" required autofocus>
            </div>
            <div class="form-group">
                <label>Password</label>
                <input type="password" name="password" required>
            </div>
            <button type="submit">Sign In</button>
        </form>
    </div>
</body>
</html>
'''

SETUP_HTML = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Setup - Data Manager</title>
''' + AUTH_STYLES + '''
</head>
<body>
    <div class="auth-card">
        <div class="logo">📊</div>
        <h1>Welcome!</h1>
        <p>Create your admin account</p>
        {% if error %}
        <div class="error">{{ error }}</div>
        {% endif %}
        <form method="POST">
            <div class="form-group">
                <label>Email</label>
                <input type="email" name="email" required autofocus>
            </div>
            <div class="form-group">
                <label>Password</label>
                <input type="password" name="password" required minlength="6">
            </div>
            <div class="form-group">
                <label>Confirm Password</label>
                <input type="password" name="confirm" required minlength="6">
            </div>
            <button type="submit">Create Account</button>
        </form>
    </div>
</body>
</html>
'''

