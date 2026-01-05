#!/usr/bin/env python3
"""
Tradier API Credentials Manager
Securely handles Tradier API tokens for data and trading
- Production API: For real market data (always)
- Sandbox: For simulated trading (testing)
- Live: For real trading (production)
"""

import json
import os
from pathlib import Path
from typing import Optional, Dict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TradierCredentials:
    """Manages Tradier API credentials securely"""
    
    CREDENTIALS_FILE = '.tradier_credentials.json'  # .gitignored
    CONFIG_FILE = 'config.json'  # New unified config
    
    def __init__(self, config_path: str = None):
        """
        Initialize credentials manager
        
        Args:
            config_path: Path to config.json (optional). If provided, reads from there first.
        """
        self.credentials_path = Path(self.CREDENTIALS_FILE)
        self.config_path = Path(config_path) if config_path else Path(self.CONFIG_FILE)
        self.credentials = self._load_credentials()
    
    def _load_credentials(self) -> Dict:
        """
        Load credentials from file. 
        Priority:
        1. config.json (credentials section)
        2. .tradier_credentials.json (legacy)
        """
        # Try config.json first (new unified approach)
        if self.config_path.exists():
            try:
                with open(self.config_path, 'r') as f:
                    config = json.load(f)
                    if 'credentials' in config and 'tradier' in config['credentials']:
                        tradier_creds = config['credentials']['tradier']
                        
                        # Convert from config.json format to internal format
                        creds = {
                            'data_api': {
                                'access_token': tradier_creds.get('data_api_token'),
                                'description': 'Production API token (for real market data)'
                            },
                            'trading': {
                                'sandbox': {
                                    'account_number': tradier_creds.get('sandbox', {}).get('account_number'),
                                    'access_token': tradier_creds.get('sandbox', {}).get('access_token'),
                                    'description': 'Sandbox account (for testing trades)'
                                },
                                'live': {
                                    'account_number': tradier_creds.get('live', {}).get('account_number'),
                                    'access_token': tradier_creds.get('live', {}).get('access_token'),
                                    'description': 'Live account (for real money trading)'
                                }
                            }
                        }
                        
                        # Only return if at least one credential is set
                        if (creds['data_api']['access_token'] or 
                            creds['trading']['sandbox']['account_number'] or 
                            creds['trading']['live']['account_number']):
                            logger.info("✓ Credentials loaded from config.json")
                            return creds
            except Exception as e:
                logger.debug(f"Could not load from config.json: {e}")
        
        # Fallback to legacy .tradier_credentials.json
        if self.credentials_path.exists():
            try:
                with open(self.credentials_path, 'r') as f:
                    creds = json.load(f)
                    logger.info("✓ Credentials loaded from .tradier_credentials.json (legacy)")
                    return creds
            except Exception as e:
                logger.error(f"Error loading credentials: {e}")
        
        # Return empty structure
        logger.info("No credentials found - using empty structure")
        return {
            'data_api': {
                'access_token': None,
                'description': 'Production API token (for real market data)'
            },
            'trading': {
                'sandbox': {
                    'account_number': None,
                    'access_token': None,
                    'description': 'Sandbox account (for testing trades)'
                },
                'live': {
                    'account_number': None,
                    'access_token': None,
                    'description': 'Live account (for real money trading)'
                }
            }
        }
    
    def _save_credentials(self):
        """Save credentials to file"""
        try:
            with open(self.credentials_path, 'w') as f:
                json.dump(self.credentials, f, indent=2)
            
            # Make file readable only by owner (Unix-like systems)
            try:
                os.chmod(self.credentials_path, 0o600)
            except:
                pass  # Windows doesn't support same permissions
            
            logger.info("✓ Credentials saved securely")
        except Exception as e:
            logger.error(f"Error saving credentials: {e}")
    
    # ============ DATA API (Production - for real market data) ============
    
    def set_data_api_token(self, access_token: str):
        """Set production API token for real market data"""
        self.credentials['data_api'] = {
            'access_token': access_token.strip(),
            'description': 'Production API token (for real market data)'
        }
        self._save_credentials()
        logger.info("✓ Data API token set (production market data)")
    
    def get_data_api_token(self) -> Optional[str]:
        """Get production API token for market data"""
        return self.credentials.get('data_api', {}).get('access_token')
    
    def has_data_api_token(self) -> bool:
        """Check if data API token is configured"""
        return self.get_data_api_token() is not None
    
    # ============ TRADING API - SANDBOX (for testing) ============
    
    def set_sandbox_credentials(self, account_number: str, access_token: str):
        """Set sandbox credentials for simulated trading"""
        self.credentials['trading']['sandbox'] = {
            'account_number': account_number.strip(),
            'access_token': access_token.strip(),
            'description': 'Sandbox account (for testing trades)'
        }
        self._save_credentials()
        logger.info("✓ Sandbox credentials set (simulated trading)")
    
    def get_sandbox_credentials(self) -> Optional[Dict]:
        """Get sandbox trading credentials"""
        creds = self.credentials.get('trading', {}).get('sandbox', {})
        if creds.get('account_number') and creds.get('access_token'):
            return {
                'account_number': creds['account_number'],
                'access_token': creds['access_token']
            }
        return None
    
    def has_sandbox_credentials(self) -> bool:
        """Check if sandbox credentials are configured"""
        return self.get_sandbox_credentials() is not None
    
    # ============ TRADING API - LIVE (for real money) ============
    
    def set_live_credentials(self, account_number: str, access_token: str):
        """Set live credentials for real money trading"""
        logger.warning("⚠️  Setting LIVE trading credentials (REAL MONEY!)")
        self.credentials['trading']['live'] = {
            'account_number': account_number.strip(),
            'access_token': access_token.strip(),
            'description': 'Live account (for real money trading)'
        }
        self._save_credentials()
        logger.warning("⚠️  LIVE credentials saved")
    
    def get_live_credentials(self) -> Optional[Dict]:
        """Get live trading credentials"""
        creds = self.credentials.get('trading', {}).get('live', {})
        if creds.get('account_number') and creds.get('access_token'):
            return {
                'account_number': creds['account_number'],
                'access_token': creds['access_token']
            }
        return None
    
    def has_live_credentials(self) -> bool:
        """Check if live credentials are configured"""
        return self.get_live_credentials() is not None
    
    # ============ UNIFIED API ============
    
    def get_trading_credentials(self, sandbox: bool = True) -> Optional[Dict]:
        """Get trading credentials for the specified environment"""
        if sandbox:
            return self.get_sandbox_credentials()
        else:
            return self.get_live_credentials()
    
    def is_trading_configured(self, sandbox: bool = True) -> bool:
        """Check if trading credentials are configured"""
        if sandbox:
            return self.has_sandbox_credentials()
        else:
            return self.has_live_credentials()
    
    def print_status(self):
        """Print comprehensive credential status"""
        print("\n" + "=" * 70)
        print("TRADIER CREDENTIALS STATUS")
        print("=" * 70)
        
        # Data API
        print("\n📊 MARKET DATA API (Production):")
        data_token = self.get_data_api_token()
        if data_token:
            print(f"  ✓ Token: {data_token[:20]}...")
            print(f"  ✓ Used for: Real market data (always)")
        else:
            print("  ✗ Not configured")
        
        # Trading - Sandbox
        print("\n🎮 TRADING - SANDBOX (Paper/Simulation):")
        sandbox = self.get_sandbox_credentials()
        if sandbox:
            print(f"  ✓ Account: {sandbox['account_number']}")
            print(f"  ✓ Token: {sandbox['access_token'][:20]}...")
            print(f"  ✓ Used for: Simulated trading while learning")
        else:
            print("  ✗ Not configured")
        
        # Trading - Live
        print("\n🔴 TRADING - LIVE (Real Money):")
        live = self.get_live_credentials()
        if live:
            print(f"  🔴 Account: {live['account_number']}")
            print(f"  🔴 Token: {live['access_token'][:20]}...")
            print(f"  🔴 Used for: Real money trading in production")
        else:
            print("  ✗ Not configured")
        
        # Architecture
        print("\n🏗️  ARCHITECTURE:")
        print("  ┌─────────────────────────────────────────┐")
        print("  │ SIMULATION MODE:                        │")
        print("  │ • Market Data: Production API ✓         │")
        print("  │ • Trading: Sandbox Account              │")
        print("  │ → Real prices, simulated trades         │")
        print("  └─────────────────────────────────────────┘")
        print("")
        print("  ┌─────────────────────────────────────────┐")
        print("  │ LIVE MODE:                              │")
        print("  │ • Market Data: Production API ✓         │")
        print("  │ • Trading: Live Account                 │")
        print("  │ → Real prices, REAL trades              │")
        print("  └─────────────────────────────────────────┘")
        
        print("=" * 70 + "\n")


if __name__ == "__main__":
    creds = TradierCredentials()
    creds.print_status()
