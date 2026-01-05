#!/usr/bin/env python3
"""
Gaussian Options Trading Bot
============================
Live and paper trading with configurable model.

Usage:
    python bot.py models/run_20260105_123456          # Paper mode (default)
    python bot.py models/run_20260105_123456 --live   # Live mode
    python bot.py --list                              # List available models
    python bot.py --status                            # Show current bot status

Configuration:
    Edit config.json to change trading parameters
    Edit server_config.json to change server addresses
"""

import sys
import os
import argparse
from pathlib import Path

# Add project root to path
BASE_DIR = Path(__file__).parent.resolve()
sys.path.insert(0, str(BASE_DIR))


def list_models():
    """List available trained models."""
    models_dir = BASE_DIR / "models"
    if not models_dir.exists():
        print("No models directory found.")
        return

    runs = sorted([d for d in models_dir.iterdir() if d.is_dir() and d.name.startswith("run_")],
                  key=lambda x: x.name, reverse=True)

    if not runs:
        print("No trained models found.")
        return

    print(f"\nAvailable models ({len(runs)} total):\n")
    print(f"{'Model Directory':<40} {'Summary'}")
    print("-" * 70)

    for run in runs[:20]:  # Show last 20
        summary_file = run / "SUMMARY.txt"
        summary = ""
        if summary_file.exists():
            try:
                lines = summary_file.read_text().split('\n')[:3]
                summary = ' | '.join(l.strip() for l in lines if l.strip())[:40]
            except:
                pass
        print(f"{run.name:<40} {summary}")

    if len(runs) > 20:
        print(f"\n... and {len(runs) - 20} more models")

    print(f"\nUsage: python bot.py {runs[0].name}")


def show_status():
    """Show current bot status."""
    status_file = BASE_DIR / "bot_status.json"
    flag_file = BASE_DIR / "go_live.flag"

    print("\n=== Bot Status ===\n")

    # Check trading mode
    if flag_file.exists():
        print("Mode: LIVE TRADING (go_live.flag exists)")
    else:
        print("Mode: PAPER TRADING (no go_live.flag)")

    # Check status file
    if status_file.exists():
        import json
        try:
            status = json.loads(status_file.read_text())
            print(f"Last update: {status.get('timestamp', 'Unknown')}")
            print(f"State: {status.get('state', 'Unknown')}")
            if 'balance' in status:
                print(f"Balance: ${status['balance']:,.2f}")
        except:
            print("Status file exists but could not be parsed")
    else:
        print("No active bot session")

    print()


def main():
    parser = argparse.ArgumentParser(
        description="Gaussian Options Trading Bot",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python bot.py models/run_20260105_123456    # Start paper trading
    python bot.py models/run_20260105_123456 --live  # Start live trading
    python bot.py --list                        # List available models
        """
    )

    parser.add_argument("model_dir", nargs="?", help="Model directory to use")
    parser.add_argument("--live", action="store_true", help="Enable live trading (creates go_live.flag)")
    parser.add_argument("--paper", action="store_true", help="Force paper trading (removes go_live.flag)")
    parser.add_argument("--list", action="store_true", help="List available models")
    parser.add_argument("--status", action="store_true", help="Show bot status")

    args = parser.parse_args()

    if args.list:
        list_models()
        return

    if args.status:
        show_status()
        return

    if not args.model_dir:
        parser.print_help()
        print("\nError: Please specify a model directory or use --list to see available models")
        sys.exit(1)

    model_path = Path(args.model_dir)
    if not model_path.is_absolute():
        model_path = BASE_DIR / model_path

    if not model_path.exists():
        print(f"Error: Model directory not found: {model_path}")
        sys.exit(1)

    # Handle live/paper mode
    flag_file = BASE_DIR / "go_live.flag"

    if args.live:
        flag_file.touch()
        print("[MODE] LIVE TRADING ENABLED - go_live.flag created")
    elif args.paper:
        if flag_file.exists():
            flag_file.unlink()
        print("[MODE] PAPER TRADING - go_live.flag removed")
    else:
        if flag_file.exists():
            print("[MODE] LIVE TRADING (go_live.flag exists)")
        else:
            print("[MODE] PAPER TRADING (default)")

    print(f"[MODEL] Using: {model_path}")
    print()

    # Import and run the actual trading bot
    # This imports the existing go_live_only.py logic
    sys.argv = [sys.argv[0], str(model_path)]  # Set up args for go_live_only

    # Execute go_live_only.py from core/
    exec(open(BASE_DIR / "core" / "go_live_only.py").read())


if __name__ == "__main__":
    main()
