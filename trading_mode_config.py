#!/usr/bin/env python3
"""
Trading Mode Configuration Manager
Controls whether the bot runs in simulation or live trading mode
"""

import json
import os
from pathlib import Path
from typing import Dict
import logging

logger = logging.getLogger(__name__)


class TradingModeConfig:
    """Manages trading mode configuration"""
    
    # Trading modes
    MODE_SIMULATION = 'SIMULATION'
    MODE_PAPER_TRADING = 'PAPER_TRADING'
    MODE_LIVE_TRADIER = 'LIVE_TRADIER'
    
    CONFIG_FILE = '.trading_mode_config.json'  # .gitignored
    
    def __init__(self):
        """Initialize configuration manager"""
        self.config_path = Path(self.CONFIG_FILE)
        self.config = self._load_config()
    
    def _load_config(self) -> Dict:
        """Load configuration from file"""
        if self.config_path.exists():
            try:
                with open(self.config_path, 'r') as f:
                    config = json.load(f)
                    logger.info(f"✓ Trading mode config loaded: {config.get('mode', 'UNKNOWN')}")
                    return config
            except Exception as e:
                logger.error(f"Error loading config: {e}")
        
        # Default configuration
        return {
            'mode': self.MODE_SIMULATION,
            'description': 'Default simulation mode',
            'live_trading': {
                'enabled': False,
                'tradier_sandbox': True
            }
        }
    
    def _save_config(self):
        """Save configuration to file"""
        try:
            with open(self.config_path, 'w') as f:
                json.dump(self.config, f, indent=2)
            
            # Make file readable only by owner (Unix-like systems)
            try:
                os.chmod(self.config_path, 0o600)
            except:
                pass  # Windows doesn't support same permissions
            
            logger.info("✓ Trading mode config saved")
        except Exception as e:
            logger.error(f"Error saving config: {e}")
    
    def get_mode(self) -> str:
        """Get current trading mode"""
        return self.config.get('mode', self.MODE_SIMULATION)
    
    def set_mode(self, mode: str, description: str = ""):
        """Set trading mode"""
        if mode not in [self.MODE_SIMULATION, self.MODE_PAPER_TRADING, self.MODE_LIVE_TRADIER]:
            raise ValueError(f"Invalid mode: {mode}")
        
        self.config['mode'] = mode
        if description:
            self.config['description'] = description
        self._save_config()
        logger.info(f"✓ Trading mode set to: {mode}")
    
    def update_live_trading_config(self, enabled: bool, tradier_sandbox: bool = True):
        """Update live trading configuration"""
        self.config['live_trading'] = {
            'enabled': enabled,
            'tradier_sandbox': tradier_sandbox
        }
        self._save_config()
        logger.info(f"✓ Live trading config updated: enabled={enabled}, sandbox={tradier_sandbox}")
    
    def is_live_trading_enabled(self) -> bool:
        """Check if live trading is enabled"""
        return self.config.get('live_trading', {}).get('enabled', False)
    
    def is_tradier_sandbox(self) -> bool:
        """Check if using Tradier sandbox"""
        return self.config.get('live_trading', {}).get('tradier_sandbox', True)
    
    def print_status(self):
        """Print current configuration status"""
        print("\n" + "=" * 70)
        print("TRADING MODE CONFIGURATION")
        print("=" * 70)
        
        mode = self.get_mode()
        print(f"\nCurrent Mode: {mode}")
        print(f"Description: {self.config.get('description', 'N/A')}")
        
        print("\nLive Trading Settings:")
        live_config = self.config.get('live_trading', {})
        print(f"  Enabled: {live_config.get('enabled', False)}")
        print(f"  Tradier Sandbox: {live_config.get('tradier_sandbox', True)}")
        
        print("\n" + "=" * 70 + "\n")


if __name__ == "__main__":
    config = TradingModeConfig()
    config.print_status()


