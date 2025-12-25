#!/usr/bin/env python3
"""
Configuration Loader for Standalone Trading Bot

Loads and validates configuration from config.json
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any

logger = logging.getLogger(__name__)

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
