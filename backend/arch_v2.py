#!/usr/bin/env python3
"""
Architecture V2 Integration Module
===================================

Central integration point for the V2 architecture.
Provides clean initialization and access to all V2 components.

This module should be imported at bot startup to configure
the architecture according to config.json.

Usage:
    from backend.arch_v2 import ArchV2, init_arch_v2
    
    # Initialize from config
    arch = init_arch_v2("config.json")
    
    # Access components
    predictor = arch.predictor_manager
    entry_policy = arch.entry_policy
    exit_manager = arch.exit_manager
    safety_filter = arch.safety_filter
    
    # Or use the global instance
    from backend.arch_v2 import get_arch_v2
    arch = get_arch_v2()
"""

import logging
import os
from typing import Dict, Optional, Any
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class ArchV2:
    """
    V2 Architecture container.
    
    Holds all architecture components and provides clean access.
    """
    # Configuration
    config: Any  # ArchConfig
    
    # Components (lazy loaded)
    predictor_manager: Optional[Any] = None
    entry_policy: Optional[Any] = None
    exit_manager: Optional[Any] = None
    safety_filter: Optional[Any] = None
    regime_embedding: Optional[Any] = None
    
    # State
    initialized: bool = False
    
    def initialize_components(self, device: str = "cpu"):
        """Initialize all architecture components."""
        from backend.arch_config import ArchConfig
        
        if self.initialized:
            logger.info("Architecture already initialized")
            return
        
        logger.info("=" * 60)
        logger.info("INITIALIZING V2 ARCHITECTURE")
        logger.info("=" * 60)
        
        # 1. Initialize Predictor Manager
        try:
            from backend.predictor_manager import PredictorManager
            self.predictor_manager = PredictorManager(
                feature_dim=50,
                sequence_length=60,
                mc_samples=self.config.uncertainty.mc_samples,
                predictor_arch=self.config.predictor_arch,
                device=device,
            )
            logger.info(f"✅ Predictor Manager: {self.config.predictor_arch}")
        except Exception as e:
            logger.error(f"❌ Predictor Manager failed: {e}")
        
        # 2. Initialize Entry Policy
        try:
            entry_type = self.config.entry_policy_type
            
            if entry_type == "unified_ppo":
                from backend.unified_rl_policy import UnifiedRLPolicy
                self.entry_policy = UnifiedRLPolicy(
                    learning_rate=0.0003,
                    gamma=0.99,
                    device=device,
                    min_confidence_to_trade=self.config.entry_policy.min_confidence,
                )
            elif entry_type == "ppo":
                from backend.rl_trading_policy import RLTradingPolicy
                self.entry_policy = RLTradingPolicy(
                    state_dim=32,
                    action_dim=5,
                    learning_rate=0.0003,
                    device=device,
                )
            elif entry_type == "rules":
                from backend.unified_rl_policy import AdaptiveRulesPolicy
                self.entry_policy = AdaptiveRulesPolicy()
            
            logger.info(f"✅ Entry Policy: {entry_type}")
        except Exception as e:
            logger.error(f"❌ Entry Policy failed: {e}")
        
        # 3. Initialize Exit Manager
        try:
            from backend.unified_exit_manager import UnifiedExitManager
            self.exit_manager = UnifiedExitManager.from_config(self.config)
            logger.info(f"✅ Exit Manager: {self.config.exit_policy_type}")
        except Exception as e:
            logger.error(f"❌ Exit Manager failed: {e}")
        
        # 4. Initialize Safety Filter (if enabled)
        if self.config.safety_filter.enabled:
            try:
                from backend.safety_filter import EntrySafetyFilter, SafetyFilterConfig
                filter_config = SafetyFilterConfig(
                    enabled=True,
                    can_veto=self.config.safety_filter.can_veto,
                    can_downgrade_size=self.config.safety_filter.can_downgrade_size,
                    veto_min_confidence=self.config.safety_filter.veto_min_confidence,
                    veto_max_vix=self.config.safety_filter.veto_max_vix,
                    tradability_gate_enabled=self.config.tradability_gate.enabled,
                    tradability_gate_checkpoint_path=self.config.tradability_gate.checkpoint_path,
                    tradability_veto_threshold=self.config.tradability_gate.veto_threshold,
                    tradability_downgrade_threshold=self.config.tradability_gate.downgrade_threshold,
                )
                self.safety_filter = EntrySafetyFilter(filter_config)
                logger.info("✅ Safety Filter: ENABLED")
            except Exception as e:
                logger.error(f"❌ Safety Filter failed: {e}")
        else:
            logger.info("ℹ️ Safety Filter: DISABLED")
        
        # 5. Initialize Regime Embedding (if enabled)
        if self.config.regime.use_embedding:
            try:
                from backend.regime_embedding import RegimeEmbedding
                self.regime_embedding = RegimeEmbedding(
                    embedding_dim=self.config.regime.embedding_dim,
                    n_trend_states=self.config.regime.n_trend_states,
                    n_vol_states=self.config.regime.n_vol_states,
                    n_liq_states=self.config.regime.n_liq_states,
                )
                logger.info(f"✅ Regime Embedding: dim={self.config.regime.embedding_dim}")
            except Exception as e:
                logger.error(f"❌ Regime Embedding failed: {e}")
        
        # 6. Set up horizon alignment
        try:
            from backend.horizon_alignment import HorizonConfig, set_horizon_config
            horizon_config = HorizonConfig(
                prediction_minutes=self.config.horizons.prediction_minutes,
                rl_reward_minutes=self.config.horizons.rl_reward_minutes,
                exit_reference_minutes=self.config.horizons.exit_reference_minutes,
                strict_alignment=self.config.horizons.assert_aligned,
            )
            set_horizon_config(horizon_config)
            logger.info(f"✅ Horizons aligned: {self.config.horizons.prediction_minutes}m")
        except Exception as e:
            logger.error(f"❌ Horizon alignment failed: {e}")
        
        # 7. If frozen_during_rl is enabled, freeze predictor now
        #    (RL training should call prepare_for_rl_training() explicitly)
        if self.config.predictor.frozen_during_rl:
            logger.info("ℹ️ Predictor will be FROZEN during RL training")
            logger.info("   Call arch.prepare_for_rl_training() before RL training loop")
        
        self.initialized = True
        logger.info("=" * 60)
        logger.info("V2 ARCHITECTURE INITIALIZED")
        logger.info("=" * 60)
    
    def prepare_for_rl_training(self):
        """
        Prepare architecture for RL training.
        
        This should be called BEFORE starting RL training to ensure:
        1. Predictor is frozen (if frozen_during_rl is true)
        2. RL policy optimizers don't include predictor params
        3. Predictor runs in inference mode
        
        The predictor STILL RUNS every step to generate predictions
        for the RL state, but its weights are NOT updated.
        """
        logger.info("")
        logger.info("=" * 60)
        logger.info("🎯 PREPARING FOR RL TRAINING")
        logger.info("=" * 60)
        
        if self.config.predictor.frozen_during_rl:
            self.freeze_predictor()
            logger.info("🔒 PREDICTOR: FROZEN")
            logger.info("   ✓ Predictor runs every step for predictions")
            logger.info("   ✓ Predictor weights will NOT be updated")
            logger.info("   ✓ Forward passes use torch.no_grad()")
        else:
            logger.warning("⚠️ PREDICTOR: TRAINABLE (not frozen)")
            logger.warning("   This may cause conflicting gradients!")
        
        # Log the RL policy that will be trained
        entry_type = self.config.entry_policy_type
        logger.info(f"🎮 ENTRY POLICY: {entry_type}")
        if self.entry_policy:
            if hasattr(self.entry_policy, 'optimizer'):
                logger.info("   ✓ Has its own optimizer (will be trained)")
            else:
                logger.info("   ⚠️ No optimizer found")
        
        # Log exit policy
        exit_type = self.config.exit_policy_type
        logger.info(f"🚪 EXIT POLICY: {exit_type}")
        
        # Log horizons
        h = self.config.horizons
        logger.info(f"⏱️ HORIZONS: pred={h.prediction_minutes}m, rl_reward={h.rl_reward_minutes}m")
        
        logger.info("-" * 60)
        logger.info("Ready for RL training loop!")
        logger.info("=" * 60)
        logger.info("")
    
    def prepare_for_predictor_training(self):
        """
        Prepare architecture for supervised predictor training.
        
        This should be called BEFORE starting predictor training to ensure:
        1. Predictor is unfrozen (gradients enabled)
        2. Predictor can be trained with its own optimizer
        """
        self.unfreeze_predictor()
        logger.info("=" * 50)
        logger.info("🔓 PREDICTOR UNFROZEN FOR SUPERVISED TRAINING")
        logger.info("   ✓ Predictor gradients enabled")
        logger.info("   ✓ Use manager.get_parameters() for optimizer")
        logger.info("=" * 50)
    
    def verify_predictor_frozen(self) -> bool:
        """
        Verify that predictor is properly frozen.
        
        Returns True if all predictor parameters have requires_grad=False.
        """
        if not self.predictor_manager or not self.predictor_manager.model:
            return False
        
        for param in self.predictor_manager.model.parameters():
            if param.requires_grad:
                return False
        return True
    
    def load_checkpoints(self, models_dir: str = "models"):
        """Load saved model checkpoints."""
        if self.predictor_manager:
            pred_path = self.config.predictor.checkpoint_path
            if os.path.exists(pred_path):
                self.predictor_manager.load_checkpoint(pred_path)
            else:
                # This is a *very* common cause of sudden performance collapse:
                # trading starts with a randomly initialized predictor.
                logger.warning(f"[ARCH] Predictor checkpoint missing: {pred_path}")
                logger.warning("[ARCH] If you switched predictor.arch, you must (a) train it or (b) point checkpoint_path to a real file.")
        
        if self.entry_policy:
            entry_type = self.config.entry_policy_type
            if entry_type == "unified_ppo":
                entry_path = f"{models_dir}/unified_rl_policy.pth"
            else:
                entry_path = f"{models_dir}/rl_trading_policy.pth"
            
            if os.path.exists(entry_path):
                try:
                    self.entry_policy.load(entry_path)
                except Exception as e:
                    logger.warning(f"Could not load entry policy: {e}")
            else:
                logger.info(f"[ARCH] No saved entry policy found at: {entry_path} (starting fresh)")
    
    def save_checkpoints(self, models_dir: str = "models"):
        """Save model checkpoints."""
        os.makedirs(models_dir, exist_ok=True)
        
        if self.predictor_manager:
            self.predictor_manager.save_checkpoint(self.config.predictor.checkpoint_path)
        
        if self.entry_policy and hasattr(self.entry_policy, 'save'):
            entry_type = self.config.entry_policy_type
            if entry_type == "unified_ppo":
                entry_path = f"{models_dir}/unified_rl_policy.pth"
            else:
                entry_path = f"{models_dir}/rl_trading_policy.pth"
            self.entry_policy.save(entry_path)
        
        if self.exit_manager:
            self.exit_manager.save(models_dir)
    
    def freeze_predictor(self):
        """Freeze predictor for RL training."""
        if self.predictor_manager:
            self.predictor_manager.freeze()
    
    def unfreeze_predictor(self):
        """Unfreeze predictor for predictor training."""
        if self.predictor_manager:
            self.predictor_manager.unfreeze()
    
    def get_status(self) -> Dict[str, Any]:
        """Get architecture status."""
        return {
            'initialized': self.initialized,
            'entry_policy_type': self.config.entry_policy_type if self.config else None,
            'exit_policy_type': self.config.exit_policy_type if self.config else None,
            'predictor_arch': self.config.predictor_arch if self.config else None,
            'predictor_frozen': self.predictor_manager.is_frozen if self.predictor_manager else None,
            'safety_filter_enabled': self.config.safety_filter.enabled if self.config else False,
            'regime_embedding_enabled': self.config.regime.use_embedding if self.config else False,
            'prediction_horizon': self.config.horizons.prediction_minutes if self.config else 15,
        }
    
    def log_startup_banner(self):
        """
        Log a loud, clear startup banner showing the active configuration.
        
        This makes it immediately obvious what "brain" is running when
        reviewing logs from any training or trading session.
        """
        if not self.config:
            logger.warning("[ARCH] No configuration loaded!")
            return
        
        c = self.config
        
        # Build frozen status string
        frozen_str = "frozen" if c.predictor.frozen_during_rl else "TRAINABLE"
        
        # Build safety filter status
        safety_str = "enabled" if c.safety_filter.enabled else "disabled"
        if c.safety_filter.enabled:
            veto = "can_veto" if c.safety_filter.can_veto else "no_veto"
            safety_str = f"enabled ({veto})"
        
        # Build regime string
        if c.regime.use_embedding:
            regime_str = f"embedding_dim={c.regime.embedding_dim}"
        else:
            regime_str = "scalars_only"
        
        # Print the banner
        logger.info("")
        logger.info("=" * 70)
        logger.info("🧠 V2 ARCHITECTURE CONFIGURATION")
        logger.info("=" * 70)
        logger.info(f"[ARCH] entry_policy    = {c.entry_policy_type}")
        logger.info(f"[ARCH] exit_policy     = {c.exit_policy_type}")
        # Predictor checkpoint status (prevents confusion about random weights)
        ckpt_status = "not_loaded"
        ckpt_path = None
        if self.predictor_manager is not None:
            ckpt_path = getattr(self.predictor_manager, "last_loaded_checkpoint", None)
            ckpt_loaded = getattr(self.predictor_manager, "checkpoint_loaded", False)
            ckpt_status = "loaded" if ckpt_loaded else "not_loaded"
        ckpt_suffix = f", ckpt={ckpt_status}" + (f" ({ckpt_path})" if ckpt_path else "")
        
        logger.info(f"[ARCH] predictor       = {c.predictor_arch} ({frozen_str}, mc_samples={c.uncertainty.mc_samples}{ckpt_suffix})")
        logger.info(f"[ARCH] safety_filter   = {safety_str}")
        logger.info(f"[ARCH] tradability_gate= {'enabled' if c.tradability_gate.enabled else 'disabled'}"
                    + (f" (ckpt={c.tradability_gate.checkpoint_path})" if c.tradability_gate.enabled else ""))
        logger.info(f"[ARCH] regime          = {regime_str}")
        logger.info(f"[ARCH] horizons        = pred={c.horizons.prediction_minutes}m, rl={c.horizons.rl_reward_minutes}m, exit={c.horizons.exit_reference_minutes}m")
        logger.info("-" * 70)
        
        # Add component status
        components = []
        if self.predictor_manager:
            status = "initialized" if self.predictor_manager._initialized else "manager_ready"
            components.append(f"predictor({status})")
        if self.entry_policy:
            components.append("entry_policy")
        if self.exit_manager:
            components.append("exit_manager")
        if self.safety_filter:
            components.append("safety_filter")
        if self.regime_embedding:
            components.append("regime_embedding")
        
        logger.info(f"[ARCH] Active components: {', '.join(components) if components else 'NONE'}")
        logger.info("=" * 70)
        logger.info("")


# Global instance
_arch_v2: Optional[ArchV2] = None


def init_arch_v2(config_path: str = "config.json", device: str = "cpu") -> ArchV2:
    """
    Initialize the V2 architecture from config.
    
    Args:
        config_path: Path to config.json
        device: Torch device ("cpu" or "cuda")
        
    Returns:
        Initialized ArchV2 instance
    """
    global _arch_v2
    
    from backend.arch_config import ArchConfig
    
    # Load config
    config = ArchConfig.from_file(config_path)
    
    # Validate
    issues = config.validate()
    if issues:
        for issue in issues:
            logger.warning(f"⚠️ Config issue: {issue}")
    
    # Create instance
    _arch_v2 = ArchV2(config=config)
    _arch_v2.initialize_components(device=device)
    
    # Attempt to load checkpoints by default (predictor/entry policy).
    # This prevents silently trading with randomly initialized models.
    try:
        _arch_v2.load_checkpoints(models_dir="models")
    except Exception as e:
        logger.warning(f"[ARCH] Could not auto-load checkpoints: {e}")
    
    # Log the startup banner (loud and clear!)
    _arch_v2.log_startup_banner()
    
    return _arch_v2


def get_arch_v2() -> Optional[ArchV2]:
    """Get the global architecture instance."""
    return _arch_v2


def set_arch_v2(arch: ArchV2):
    """Set the global architecture instance."""
    global _arch_v2
    _arch_v2 = arch


# Convenience functions for common operations
def get_entry_policy():
    """Get the configured entry policy."""
    arch = get_arch_v2()
    return arch.entry_policy if arch else None


def get_exit_manager():
    """Get the exit manager."""
    arch = get_arch_v2()
    return arch.exit_manager if arch else None


def get_predictor():
    """Get the predictor manager."""
    arch = get_arch_v2()
    return arch.predictor_manager if arch else None


def get_safety_filter():
    """Get the safety filter (may be None if disabled)."""
    arch = get_arch_v2()
    return arch.safety_filter if arch else None


# Export
__all__ = [
    'ArchV2',
    'init_arch_v2',
    'get_arch_v2',
    'set_arch_v2',
    'get_entry_policy',
    'get_exit_manager',
    'get_predictor',
    'get_safety_filter',
]
