#!/usr/bin/env python3
"""
Configuration Schema for Bot Competition Platform

Defines which configuration fields are relevant for trading reproducibility
and provides functions to extract, sanitize, and hash configurations.

When a bot registers with the Data Manager, its full trading configuration
is captured (minus sensitive fields like API keys) so that:
1. Winning strategies can be replicated
2. Config changes can be tracked over time
3. Leaderboard rankings can be correlated with specific settings
"""

import copy
import hashlib
import json
import re
from typing import Dict, Any, List, Set


# Configuration sections relevant for trading reproducibility
# These are the sections that affect trading behavior and should be captured
TRADING_CONFIG_SECTIONS = [
    "trading",
    "risk",
    "risk_management",
    "entry_controller",
    "architecture",
    "neural_network",
    "hmm",
    "options",
    "calibration",
    "cold_start",
    "threshold_management",
    "prediction_timeframes",
    "feature_pipeline",
    "time_travel_training",
    "adaptive_learning",
    "emergency_circuit_breakers",
    "liquidity",
]

# Sections to explicitly exclude (not relevant for reproducibility)
EXCLUDED_SECTIONS = [
    "credentials",
    "data_manager",
    "bot_reporter",
    "data_sources",
    "data_fetching",
    "logging",
    "system",
    "dashboard",
    "training_dashboard",
    "health",
    "shadow_trading",
    "live_trading",
]

# Sensitive field patterns to remove (case-insensitive regex)
SENSITIVE_PATTERNS = [
    r".*api[_-]?key.*",
    r".*token.*",
    r".*password.*",
    r".*secret.*",
    r".*credential.*",
    r".*access[_-]?key.*",
    r".*auth.*",
    r".*webhook.*",
]

# Compiled patterns for efficiency
_SENSITIVE_REGEX = [re.compile(p, re.IGNORECASE) for p in SENSITIVE_PATTERNS]


def is_sensitive_key(key: str) -> bool:
    """Check if a key name matches sensitive patterns."""
    for pattern in _SENSITIVE_REGEX:
        if pattern.match(key):
            return True
    return False


def remove_sensitive_fields(obj: Any, path: str = "") -> Any:
    """
    Recursively remove sensitive fields from a config object.

    Args:
        obj: Config dict or value
        path: Current path for debugging

    Returns:
        Sanitized copy of the object
    """
    if isinstance(obj, dict):
        result = {}
        for key, value in obj.items():
            # Skip keys that match sensitive patterns
            if is_sensitive_key(key):
                continue
            # Skip keys that start with underscore (internal notes)
            if key.startswith("_"):
                continue
            # Recursively process nested dicts
            result[key] = remove_sensitive_fields(value, f"{path}.{key}")
        return result
    elif isinstance(obj, list):
        return [remove_sensitive_fields(item, f"{path}[]") for item in obj]
    else:
        return obj


def extract_trading_config(full_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract only trading-relevant configuration sections.

    Args:
        full_config: Complete config.json dict

    Returns:
        Dict containing only trading-relevant sections, sanitized
    """
    result = {}

    for section in TRADING_CONFIG_SECTIONS:
        if section in full_config:
            # Deep copy and sanitize
            section_data = copy.deepcopy(full_config[section])
            result[section] = remove_sensitive_fields(section_data, section)

    return result


def compute_config_hash(config: Dict[str, Any]) -> str:
    """
    Compute a deterministic hash of the trading configuration.

    Used to detect when configuration has changed between runs.

    Args:
        config: Trading configuration dict (should be pre-sanitized)

    Returns:
        16-character hex hash string
    """
    # Sort keys for deterministic serialization
    serialized = json.dumps(config, sort_keys=True, separators=(',', ':'))
    # Use SHA256 and take first 16 chars
    return hashlib.sha256(serialized.encode('utf-8')).hexdigest()[:16]


def prepare_config_for_reporting(full_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Prepare a full config for reporting to the Data Manager.

    This is the main entry point - extracts relevant sections,
    removes sensitive data, and adds metadata.

    Args:
        full_config: Complete config.json dict

    Returns:
        Sanitized config ready for transmission
    """
    from datetime import datetime

    # Extract trading-relevant sections
    trading_config = extract_trading_config(full_config)

    # Add metadata
    trading_config["_meta"] = {
        "captured_at": datetime.utcnow().isoformat() + "Z",
        "schema_version": "1.0",
    }

    return trading_config


def get_config_summary(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Get a brief summary of key config settings for display.

    Args:
        config: Trading configuration dict

    Returns:
        Dict with key settings for quick viewing
    """
    summary = {}

    # Trading basics
    if "trading" in config:
        t = config["trading"]
        summary["symbol"] = t.get("symbol", "?")
        summary["initial_balance"] = t.get("initial_balance", 0)
        summary["max_positions"] = t.get("max_positions", 0)
        summary["confidence_threshold"] = t.get("base_confidence_threshold", 0)

    # Entry controller
    if "entry_controller" in config:
        ec = config["entry_controller"]
        summary["entry_controller"] = ec.get("type", "?")

    # Exit policy
    if "architecture" in config:
        arch = config["architecture"]
        if "exit_policy" in arch:
            ep = arch["exit_policy"]
            summary["stop_loss"] = ep.get("hard_stop_loss_pct", 0)
            summary["take_profit"] = ep.get("hard_take_profit_pct", 0)
            summary["max_hold_minutes"] = ep.get("hard_max_hold_minutes", 0)

    # Risk
    if "risk_management" in config:
        rm = config["risk_management"]
        summary["stop_loss_pct"] = rm.get("stop_loss_pct", 0)
        summary["take_profit_pct"] = rm.get("take_profit_pct", 0)

    # Neural network
    if "neural_network" in config:
        nn = config["neural_network"]
        summary["sequence_length"] = nn.get("sequence_length", 0)
        summary["learning_rate"] = nn.get("learning_rate", 0)

    # HMM
    if "hmm" in config:
        summary["hmm_enabled"] = config["hmm"].get("enabled", False)

    return summary


def configs_are_equivalent(config1: Dict[str, Any], config2: Dict[str, Any]) -> bool:
    """
    Check if two configs are functionally equivalent.

    Args:
        config1: First config
        config2: Second config

    Returns:
        True if configs have the same hash
    """
    hash1 = compute_config_hash(config1)
    hash2 = compute_config_hash(config2)
    return hash1 == hash2


def diff_configs(old_config: Dict[str, Any], new_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Get the differences between two configs.

    Args:
        old_config: Previous config
        new_config: Current config

    Returns:
        Dict with added, removed, and changed keys
    """
    def flatten_dict(d: Dict, prefix: str = "") -> Dict[str, Any]:
        """Flatten nested dict to dot-notation keys."""
        items = {}
        for k, v in d.items():
            key = f"{prefix}.{k}" if prefix else k
            if isinstance(v, dict):
                items.update(flatten_dict(v, key))
            else:
                items[key] = v
        return items

    old_flat = flatten_dict(old_config)
    new_flat = flatten_dict(new_config)

    old_keys = set(old_flat.keys())
    new_keys = set(new_flat.keys())

    added = {k: new_flat[k] for k in (new_keys - old_keys)}
    removed = {k: old_flat[k] for k in (old_keys - new_keys)}
    changed = {
        k: {"old": old_flat[k], "new": new_flat[k]}
        for k in (old_keys & new_keys)
        if old_flat[k] != new_flat[k]
    }

    return {
        "added": added,
        "removed": removed,
        "changed": changed,
    }


__all__ = [
    'TRADING_CONFIG_SECTIONS',
    'EXCLUDED_SECTIONS',
    'extract_trading_config',
    'compute_config_hash',
    'prepare_config_for_reporting',
    'get_config_summary',
    'configs_are_equivalent',
    'diff_configs',
    'is_sensitive_key',
    'remove_sensitive_fields',
]
