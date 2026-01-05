#!/usr/bin/env python3
"""
Gaussian Options Trading Bot
============================
Live and paper trading with configurable model.

Usage:
    python bot.py                                     # Use best model (from best_model.json)
    python bot.py models/run_20260105_123456          # Use specific model
    python bot.py models/run_20260105_123456 --live   # Live mode with specific model
    python bot.py --list                              # List available models
    python bot.py --status                            # Show current bot status
    python bot.py --set-best models/my_model          # Set the best model

Configuration:
    Edit config.json to change trading parameters
    Edit server_config.json to change server addresses
    Edit best_model.json to change the default model
"""

import sys
import os
import json
import argparse
from pathlib import Path
from datetime import datetime

# Add project root to path
BASE_DIR = Path(__file__).parent.resolve()
sys.path.insert(0, str(BASE_DIR))

BEST_MODEL_FILE = BASE_DIR / "best_model.json"


def get_best_model() -> Path | None:
    """Get the current best model from best_model.json."""
    if not BEST_MODEL_FILE.exists():
        return None

    try:
        config = json.loads(BEST_MODEL_FILE.read_text())
        model_dir = config.get("model_dir")
        if model_dir:
            model_path = BASE_DIR / model_dir
            if model_path.exists():
                return model_path
            # Try without models/ prefix
            if not model_dir.startswith("models/"):
                model_path = BASE_DIR / "models" / model_dir
                if model_path.exists():
                    return model_path
    except Exception as e:
        print(f"Warning: Could not load best_model.json: {e}")

    return None


def set_best_model(model_dir: str, notes: str = ""):
    """Set the best model in best_model.json."""
    model_path = Path(model_dir)
    if not model_path.is_absolute():
        model_path = BASE_DIR / model_path

    if not model_path.exists():
        print(f"Error: Model directory not found: {model_path}")
        sys.exit(1)

    # Get relative path for storage
    try:
        rel_path = model_path.relative_to(BASE_DIR)
    except ValueError:
        rel_path = model_path

    # Read summary if available
    validation = {}
    summary_file = model_path / "SUMMARY.txt"
    if summary_file.exists():
        try:
            content = summary_file.read_text()
            for line in content.split('\n'):
                if 'P&L:' in line and '%' in line:
                    import re
                    match = re.search(r'\(([-+]?\d+\.?\d*)%\)', line)
                    if match:
                        validation['pnl_pct'] = float(match.group(1))
                if 'Win Rate:' in line:
                    import re
                    match = re.search(r'([\d.]+)%', line)
                    if match:
                        validation['win_rate'] = float(match.group(1))
        except:
            pass

    config = {
        "_description": "Current best model for production use. Update with: python bot.py --set-best models/run_xxx",
        "model_dir": str(rel_path),
        "set_date": datetime.now().strftime("%Y-%m-%d"),
        "set_by": "manual",
        "notes": notes or f"Set as best model on {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        "validation": validation
    }

    BEST_MODEL_FILE.write_text(json.dumps(config, indent=2))
    print(f"✓ Best model set to: {rel_path}")
    if validation:
        print(f"  P&L: {validation.get('pnl_pct', 'N/A')}%")
        print(f"  Win Rate: {validation.get('win_rate', 'N/A')}%")


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
    python bot.py                               # Use best model (default)
    python bot.py models/run_20260105_123456    # Use specific model
    python bot.py --live                        # Live trading with best model
    python bot.py --list                        # List available models
    python bot.py --set-best models/my_model    # Set the best model
        """
    )

    parser.add_argument("model_dir", nargs="?", help="Model directory to use (default: best model)")
    parser.add_argument("--live", action="store_true", help="Enable live trading (creates go_live.flag)")
    parser.add_argument("--paper", action="store_true", help="Force paper trading (removes go_live.flag)")
    parser.add_argument("--list", action="store_true", help="List available models")
    parser.add_argument("--status", action="store_true", help="Show bot status")
    parser.add_argument("--set-best", metavar="MODEL", help="Set the best model for default use")

    args = parser.parse_args()

    if args.list:
        list_models()
        return

    if args.status:
        show_status()
        return

    if args.set_best:
        set_best_model(args.set_best)
        return

    # Determine model path
    if args.model_dir:
        model_path = Path(args.model_dir)
        if not model_path.is_absolute():
            model_path = BASE_DIR / model_path
    else:
        # Use best model
        model_path = get_best_model()
        if model_path:
            print(f"[MODEL] Using best model: {model_path.relative_to(BASE_DIR)}")
        else:
            print("Error: No model specified and no best_model.json found.")
            print("       Use --list to see available models, or --set-best to set a default.")
            sys.exit(1)

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

    # Add core/ to path for imports (unified_options_trading_bot is in core/)
    sys.path.insert(0, str(BASE_DIR / "core"))

    # Execute go_live_only.py from core/
    exec(open(BASE_DIR / "core" / "go_live_only.py").read())


if __name__ == "__main__":
    main()
