#!/usr/bin/env python3
"""
Configuration Loader for Standalone Trading Bot

Loads and validates configuration from config.json
Loads server configuration from server_config.json with localhost fallback
"""

import json
import logging
import socket
from pathlib import Path
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

# =============================================================================
# SERVER CONFIGURATION (with localhost fallback)
# =============================================================================

_server_config: Optional[Dict[str, Any]] = None
_active_server: Optional[str] = None

def load_server_config(config_path: str = "server_config.json") -> Dict[str, Any]:
    """Load server configuration from JSON file."""
    global _server_config
    if _server_config is not None:
        return _server_config

    path = Path(config_path)
    if not path.exists():
        # Default config if file doesn't exist
        _server_config = {
            "primary": {"ip": "192.168.20.235"},
            "fallback": {"ip": "localhost"},
            "dashboard": {"host": "0.0.0.0", "port": 5000},
            "training_dashboard": {"host": "0.0.0.0", "port": 5001},
            "data_manager": {"port": 5050, "timeout_seconds": 3}
        }
        return _server_config

    try:
        with open(path, 'r') as f:
            _server_config = json.load(f)
        return _server_config
    except Exception as e:
        logger.warning(f"Could not load server_config.json: {e}, using defaults")
        _server_config = {
            "primary": {"ip": "localhost"},
            "fallback": {"ip": "localhost"},
            "data_manager": {"port": 5050, "timeout_seconds": 3}
        }
        return _server_config


def is_server_reachable(host: str, port: int, timeout: float = 3.0) -> bool:
    """Check if a server is reachable via TCP connection."""
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(timeout)
        result = sock.connect_ex((host, port))
        sock.close()
        return result == 0
    except Exception:
        return False


def get_active_server() -> str:
    """
    Get the active server IP (primary or fallback).

    Checks if primary server is reachable, falls back to localhost if not.
    Result is cached for the session.
    """
    global _active_server
    if _active_server is not None:
        return _active_server

    config = load_server_config()
    primary_ip = config.get("primary", {}).get("ip", "192.168.20.235")
    fallback_ip = config.get("fallback", {}).get("ip", "localhost")
    dm_port = config.get("data_manager", {}).get("port", 5050)
    timeout = config.get("data_manager", {}).get("timeout_seconds", 3)

    # Check if primary server is reachable
    if is_server_reachable(primary_ip, dm_port, timeout):
        _active_server = primary_ip
        logger.info(f"✓ Using primary server: {primary_ip}")
    else:
        _active_server = fallback_ip
        logger.info(f"⚠ Primary server {primary_ip} not reachable, using fallback: {fallback_ip}")

    return _active_server


def get_data_manager_url() -> str:
    """Get the data manager URL (with fallback to localhost)."""
    config = load_server_config()
    server = get_active_server()
    port = config.get("data_manager", {}).get("port", 5050)
    return f"http://{server}:{port}"


def get_dashboard_url(dashboard_type: str = "live") -> str:
    """Get dashboard URL for the active server."""
    config = load_server_config()
    server = get_active_server()

    if dashboard_type == "training":
        port = config.get("training_dashboard", {}).get("port", 5001)
    else:
        port = config.get("dashboard", {}).get("port", 5000)

    return f"http://{server}:{port}"


def reset_server_cache():
    """Reset the cached server selection (forces re-check on next call)."""
    global _active_server
    _active_server = None

class BotConfig:
    """Trading bot configuration manager"""
    
    def __init__(self, config_path: str = "config.json"):
        self.config_path = Path(config_path)
        self.config = self._load_config()
        self._validate_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from JSON file"""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found: {self.config_path}")
        
        try:
            with open(self.config_path, 'r') as f:
                config = json.load(f)
            logger.info(f"✓ Configuration loaded from {self.config_path}")
            return config
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in config file: {e}")
    
    def _validate_config(self):
        """Validate required configuration fields"""
        required_sections = ['trading', 'data_sources', 'prediction_timeframes']
        
        for section in required_sections:
            if section not in self.config:
                raise ValueError(f"Missing required config section: {section}")
        
        # Validate trading symbol
        symbol = self.config['trading'].get('symbol')
        if not symbol or not isinstance(symbol, str):
            raise ValueError("Invalid or missing trading symbol")
        
        logger.info(f"✓ Configuration validated - Trading: {symbol}")
    
    def get(self, *keys, default=None):
        """Get nested config value"""
        value = self.config
        for key in keys:
            if isinstance(value, dict):
                value = value.get(key)
                if value is None:
                    return default
            else:
                return default
        return value
    
    def get_symbol(self) -> str:
        """Get trading symbol"""
        return self.config['trading']['symbol']
    
    def get_initial_balance(self) -> float:
        """Get initial account balance"""
        return self.config['trading'].get('initial_balance', 1000.0)
    
    def get_paper_initial_balance(self) -> float:
        """Get paper trading initial balance (separate from live account)"""
        return self.config['trading'].get('paper_initial_balance', 10000.0)
    
    def get_enabled_timeframes(self) -> Dict:
        """Get enabled prediction timeframes"""
        timeframes = self.config.get('prediction_timeframes', {})
        return {
            name: config for name, config in timeframes.items()
            if config.get('enabled', True)
        }
    
    def get_data_source(self) -> str:
        """Get primary data source"""
        return self.config['data_sources'].get('primary', 'yahoo')
    
    def is_hmm_enabled(self) -> bool:
        """Check if HMM is enabled"""
        return self.config.get('hmm', {}).get('enabled', True)
    
    def is_shadow_trading_enabled(self) -> bool:
        """Check if shadow trading is enabled"""
        return self.config.get('shadow_trading', {}).get('enabled', True)
    
    def get_shadow_mode(self) -> str:
        """Get shadow trading mode (sandbox or live)"""
        return self.config.get('shadow_trading', {}).get('mode', 'sandbox')
    
    def save(self, filepath: str = None):
        """Save current configuration to file"""
        path = Path(filepath) if filepath else self.config_path
        with open(path, 'w') as f:
            json.dump(self.config, f, indent=2)
        logger.info(f"Configuration saved to {path}")


# Global config instance
_config = None

def load_config(config_path: str = "config.json") -> BotConfig:
    """Load configuration (singleton)"""
    global _config
    if _config is None:
        _config = BotConfig(config_path)
    return _config

def get_config() -> BotConfig:
    """Get loaded configuration"""
    global _config
    if _config is None:
        _config = load_config()
    return _config
