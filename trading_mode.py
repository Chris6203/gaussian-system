#!/usr/bin/env python3
"""
Trading Mode Configuration Reader

This module is used by go_live_only.py and train_then_go_live.py
to determine whether to use Sandbox or Live trading.
"""

import json
from pathlib import Path

CONFIG_FILE = Path("data/trading_mode.json")

def get_trading_mode():
    """
    Get current trading mode
    
    Returns:
        dict: {
            'mode': 'sandbox' or 'live',
            'is_sandbox': bool,
            'is_live': bool,
            'description': str
        }
    """
    if CONFIG_FILE.exists():
        try:
            with open(CONFIG_FILE, 'r') as f:
                config = json.load(f)
                mode = config.get('mode', 'sandbox')
        except:
            mode = 'sandbox'  # Default to safe mode
    else:
        mode = 'sandbox'  # Default to safe mode
    
    return {
        'mode': mode,
        'is_sandbox': mode == 'sandbox',
        'is_live': mode == 'live',
        'description': 'Tradier Sandbox (fake money)' if mode == 'sandbox' else 'Tradier Live API (REAL MONEY)'
    }

def require_sandbox():
    """
    Ensure we're in sandbox mode, exit if not
    Useful for safety checks
    """
    mode = get_trading_mode()
    if not mode['is_sandbox']:
        print("ERROR: This script requires SANDBOX mode")
        print("   Current mode: LIVE (real money)")
        print()
        print("To switch to sandbox:")
        print("  python change_trading_mode.py sandbox")
        print()
        import sys
        sys.exit(1)

def require_live():
    """
    Ensure we're in live mode, exit if not
    Useful for production scripts
    """
    mode = get_trading_mode()
    if not mode['is_live']:
        print("ERROR: This script requires LIVE mode")
        print("   Current mode: SANDBOX (fake money)")
        print()
        print("To switch to live:")
        print("  python change_trading_mode.py live")
        print()
        import sys
        sys.exit(1)

def print_trading_mode_banner():
    """Print a clear banner showing current trading mode"""
    mode = get_trading_mode()
    
    if mode['is_sandbox']:
        print("="*70)
        print("TRADING MODE: SANDBOX (Fake Money)")
        print("="*70)
        print("  - Using Tradier Sandbox API")
        print("  - No real money at risk")
        print("  - Safe for testing")
        print("="*70)
    else:
        print("="*70)
        print("TRADING MODE: LIVE (REAL MONEY!)")
        print("="*70)
        print("  - Using Tradier Live API")
        print("  - REAL MONEY AT RISK")
        print("  - Losses are PERMANENT")
        print("="*70)
    print()

