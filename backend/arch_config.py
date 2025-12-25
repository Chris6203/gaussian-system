#!/usr/bin/env python3
"""
Architecture Configuration Module
=================================

Centralized configuration for the trading bot architecture.
Provides clean access to all architecture-related settings.

Usage:
    from backend.arch_config import ArchConfig
    
    config = ArchConfig.from_file("config.json")
    
    # Access settings
    print(config.entry_policy_type)  # "unified_ppo"
    print(config.exit_policy_type)   # "xgboost_exit"
    print(config.predictor_frozen)   # True
"""

import json
import logging
from dataclasses import dataclass, field
from typing import Dict, Optional, Any, List
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class EntryPolicyConfig:
    """Configuration for entry policy selection."""
    type: str = "unified_ppo"  # "unified_ppo" | "ppo" | "rules"
    min_confidence: float = 0.55
    require_trend_alignment: bool = True
    max_consecutive_losses_before_pause: int = 3
    
    # PPO-specific settings
    ppo_learning_rate: float = 0.0003
    ppo_gamma: float = 0.99
    ppo_entropy_coef: float = 0.05
    
    # Rules-specific settings
    rules_min_predicted_return: float = 0.02
    rules_max_vix: float = 30.0
    rules_min_vix: float = 12.0


@dataclass
class ExitPolicyConfig:
    """Configuration for exit policy selection."""
    type: str = "xgboost_exit"  # "xgboost_exit" | "nn_exit"
    
    # Hard safety rules (always apply first)
    hard_stop_loss_pct: float = -15.0
    hard_take_profit_pct: float = 40.0
    hard_max_hold_minutes: int = 240  # 4 hours
    hard_min_expiry_minutes: int = 30
    
    # Trailing stop
    trailing_stop_activation_pct: float = 10.0
    trailing_stop_distance_pct: float = 5.0
    
    # XGBoost-specific
    xgb_exit_threshold: float = 0.55
    xgb_min_trades_for_ml: int = 30
    
    # NN-specific
    nn_exit_threshold: float = 0.6


@dataclass
class PredictorConfig:
    """Configuration for predictor architecture."""
    arch: str = "v2_slim_bayesian"  # "v1_original" | "v2_slim_bayesian"
    frozen_during_rl: bool = True
    
    # Architecture variants
    sequence_length: int = 60
    use_gaussian_kernels: bool = True
    temporal_encoder: str = "tcn"  # "tcn" | "lstm"
    
    # Regularization (v2_slim_bayesian uses lower values)
    dropout_backbone: float = 0.15  # v1 uses 0.3
    dropout_heads: float = 0.1
    bayesian_in_backbone: bool = False  # v2 only uses Bayesian in heads
    
    # Model paths
    checkpoint_path: str = "models/predictor_v2.pt"


@dataclass
class SafetyFilterConfig:
    """Configuration for optional safety filter."""
    enabled: bool = True
    can_veto: bool = True
    can_downgrade_size: bool = True
    
    # Veto conditions
    veto_min_confidence: float = 0.40  # Veto if below this
    veto_max_vix: float = 35.0  # Veto if VIX above this
    veto_conflicting_regime: bool = True  # Veto if signal conflicts with HMM trend
    
    # Downgrade conditions
    downgrade_low_volume: bool = True
    downgrade_near_close: bool = True  # Reduce size in last hour


@dataclass
class RegimeConfig:
    """Configuration for HMM regime integration."""
    use_embedding: bool = True
    embedding_dim: int = 8
    
    # HMM settings
    n_trend_states: int = 3
    n_vol_states: int = 3
    n_liq_states: int = 3
    retrain_interval_hours: int = 24
    min_samples_for_training: int = 500


@dataclass
class UncertaintyConfig:
    """Configuration for Bayesian uncertainty."""
    mc_samples: int = 5
    inject_into_state: bool = True
    
    # How to use uncertainty
    use_risk_adjusted_return: bool = True  # return_mean / (return_std + eps)
    uncertainty_exit_boost: bool = True  # More aggressive exit on high uncertainty
    uncertainty_threshold_for_boost: float = 0.3  # std > this triggers boost


@dataclass
class HorizonsConfig:
    """Configuration for time horizon alignment."""
    prediction_minutes: int = 15
    rl_reward_minutes: int = 15
    exit_reference_minutes: int = 15
    
    # Assertions to ensure alignment
    assert_aligned: bool = True


@dataclass
class TradabilityGateConfig:
    """
    Configuration for the optional tradability gate.
    
    The tradability gate is a small model that decides whether a proposed trade
    is likely to beat friction (spread/fees) over the configured horizon.
    """
    enabled: bool = False
    checkpoint_path: str = "models/tradability_gate.pt"
    # If predicted P(tradable) < veto_threshold and can_veto is true → veto trade
    veto_threshold: float = 0.45
    # If predicted P(tradable) < downgrade_threshold and can_downgrade is true → downgrade size
    downgrade_threshold: float = 0.55
    # Training label threshold (used by training script)
    min_realized_move_to_count: float = 0.0010  # 0.10% underlying move


@dataclass
class ArchConfig:
    """
    Main architecture configuration container.
    
    This is the single source of truth for architecture decisions.
    """
    entry_policy: EntryPolicyConfig = field(default_factory=EntryPolicyConfig)
    exit_policy: ExitPolicyConfig = field(default_factory=ExitPolicyConfig)
    predictor: PredictorConfig = field(default_factory=PredictorConfig)
    safety_filter: SafetyFilterConfig = field(default_factory=SafetyFilterConfig)
    tradability_gate: TradabilityGateConfig = field(default_factory=TradabilityGateConfig)
    regime: RegimeConfig = field(default_factory=RegimeConfig)
    uncertainty: UncertaintyConfig = field(default_factory=UncertaintyConfig)
    horizons: HorizonsConfig = field(default_factory=HorizonsConfig)
    
    # Convenience properties
    @property
    def entry_policy_type(self) -> str:
        return self.entry_policy.type
    
    @property
    def exit_policy_type(self) -> str:
        return self.exit_policy.type
    
    @property
    def predictor_frozen(self) -> bool:
        return self.predictor.frozen_during_rl
    
    @property
    def predictor_arch(self) -> str:
        return self.predictor.arch
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "ArchConfig":
        """Create ArchConfig from a dictionary."""
        arch_dict = config_dict.get("architecture", {})
        
        def filter_doc_keys(d: Dict) -> Dict:
            """Remove documentation keys (starting with _ or named 'description')."""
            return {k: v for k, v in d.items() if not k.startswith("_") and k != "description"}
        
        # Parse sub-configs (filter out documentation keys like _type_options, description, etc.)
        entry_policy = EntryPolicyConfig(**filter_doc_keys(arch_dict.get("entry_policy", {}))) if "entry_policy" in arch_dict else EntryPolicyConfig()
        exit_policy = ExitPolicyConfig(**filter_doc_keys(arch_dict.get("exit_policy", {}))) if "exit_policy" in arch_dict else ExitPolicyConfig()
        predictor = PredictorConfig(**filter_doc_keys(arch_dict.get("predictor", {}))) if "predictor" in arch_dict else PredictorConfig()
        safety_filter = SafetyFilterConfig(**filter_doc_keys(arch_dict.get("safety_filter", {}))) if "safety_filter" in arch_dict else SafetyFilterConfig()
        tradability_gate = TradabilityGateConfig(**filter_doc_keys(arch_dict.get("tradability_gate", {}))) if "tradability_gate" in arch_dict else TradabilityGateConfig()
        regime = RegimeConfig(**filter_doc_keys(arch_dict.get("regime", {}))) if "regime" in arch_dict else RegimeConfig()
        uncertainty = UncertaintyConfig(**filter_doc_keys(arch_dict.get("uncertainty", {}))) if "uncertainty" in arch_dict else UncertaintyConfig()
        horizons = HorizonsConfig(**filter_doc_keys(arch_dict.get("horizons", {}))) if "horizons" in arch_dict else HorizonsConfig()
        
        return cls(
            entry_policy=entry_policy,
            exit_policy=exit_policy,
            predictor=predictor,
            safety_filter=safety_filter,
            tradability_gate=tradability_gate,
            regime=regime,
            uncertainty=uncertainty,
            horizons=horizons,
        )
    
    @classmethod
    def from_file(cls, config_path: str) -> "ArchConfig":
        """Load ArchConfig from a JSON config file."""
        path = Path(config_path)
        if not path.exists():
            logger.warning(f"Config file not found: {config_path}, using defaults")
            return cls()
        
        with open(path, "r") as f:
            config_dict = json.load(f)
        
        return cls.from_dict(config_dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        from dataclasses import asdict
        return {"architecture": asdict(self)}
    
    def validate(self) -> List[str]:
        """Validate configuration and return list of issues."""
        issues = []
        
        # Validate entry policy type
        valid_entry_types = ["unified_ppo", "ppo", "rules"]
        if self.entry_policy.type not in valid_entry_types:
            issues.append(f"Invalid entry_policy.type: {self.entry_policy.type}. Must be one of {valid_entry_types}")
        
        # Validate exit policy type
        valid_exit_types = ["xgboost_exit", "nn_exit"]
        if self.exit_policy.type not in valid_exit_types:
            issues.append(f"Invalid exit_policy.type: {self.exit_policy.type}. Must be one of {valid_exit_types}")
        
        # Validate predictor arch
        valid_predictor_archs = ["v1_original", "v2_slim_bayesian"]
        if self.predictor.arch not in valid_predictor_archs:
            issues.append(f"Invalid predictor.arch: {self.predictor.arch}. Must be one of {valid_predictor_archs}")
        
        # Validate horizon alignment
        if self.horizons.assert_aligned:
            horizons = [
                self.horizons.prediction_minutes,
                self.horizons.rl_reward_minutes,
                self.horizons.exit_reference_minutes,
            ]
            if len(set(horizons)) > 1:
                issues.append(f"Horizons not aligned: prediction={horizons[0]}, rl_reward={horizons[1]}, exit_ref={horizons[2]}")
        
        # Validate safety rules
        if self.exit_policy.hard_stop_loss_pct > 0:
            issues.append(f"hard_stop_loss_pct should be negative: {self.exit_policy.hard_stop_loss_pct}")

        # Validate tradability gate thresholds
        if self.tradability_gate.enabled:
            if not (0.0 <= self.tradability_gate.veto_threshold <= 1.0):
                issues.append(f"tradability_gate.veto_threshold must be in [0,1]: {self.tradability_gate.veto_threshold}")
            if not (0.0 <= self.tradability_gate.downgrade_threshold <= 1.0):
                issues.append(f"tradability_gate.downgrade_threshold must be in [0,1]: {self.tradability_gate.downgrade_threshold}")
            if self.tradability_gate.veto_threshold > self.tradability_gate.downgrade_threshold:
                issues.append("tradability_gate.veto_threshold should be <= tradability_gate.downgrade_threshold")
        
        return issues
    
    def print_summary(self):
        """Print configuration summary."""
        logger.info("=" * 60)
        logger.info("ARCHITECTURE CONFIGURATION SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Entry Policy: {self.entry_policy.type}")
        logger.info(f"Exit Policy:  {self.exit_policy.type}")
        logger.info(f"Predictor:    {self.predictor.arch} (frozen={self.predictor.frozen_during_rl})")
        logger.info(f"Safety Filter: {'ENABLED' if self.safety_filter.enabled else 'DISABLED'}")
        logger.info(f"Regime Embedding: {'YES (dim={})'.format(self.regime.embedding_dim) if self.regime.use_embedding else 'NO'}")
        logger.info(f"Uncertainty MC: {self.uncertainty.mc_samples} samples")
        logger.info(f"Prediction Horizon: {self.horizons.prediction_minutes} min")
        logger.info("=" * 60)


# Singleton instance for global access
_global_arch_config: Optional[ArchConfig] = None


def get_arch_config() -> ArchConfig:
    """Get the global architecture configuration."""
    global _global_arch_config
    if _global_arch_config is None:
        _global_arch_config = ArchConfig()
    return _global_arch_config


def set_arch_config(config: ArchConfig):
    """Set the global architecture configuration."""
    global _global_arch_config
    _global_arch_config = config


def init_arch_config(config_path: str = "config.json") -> ArchConfig:
    """Initialize global architecture configuration from file."""
    config = ArchConfig.from_file(config_path)
    
    # Validate
    issues = config.validate()
    if issues:
        for issue in issues:
            logger.warning(f"⚠️ Config issue: {issue}")
    
    set_arch_config(config)
    config.print_summary()
    
    return config


# Export
__all__ = [
    "ArchConfig",
    "EntryPolicyConfig",
    "ExitPolicyConfig",
    "PredictorConfig",
    "SafetyFilterConfig",
    "RegimeConfig",
    "UncertaintyConfig",
    "HorizonsConfig",
    "get_arch_config",
    "set_arch_config",
    "init_arch_config",
]
