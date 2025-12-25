#!/usr/bin/env python3
"""
Component Integration Module
============================

Integrates all the new ML/performance components into the trading system.

This module provides factory functions to wire up:
- RobustFetcher (retries, caching)
- FeatureCache (LRU caching)
- CalibrationTracker (Platt + Isotonic)
- ModelHealthMonitor (drift detection)
- HealthCheckSystem (pre-cycle validation)
- AsyncOperations (parallel execution)
- EnhancedPPOTrainer (prioritized replay + Sharpe rewards)

Usage:
    from backend.integration import integrate_all_components
    
    # In your bot initialization:
    components = integrate_all_components(bot, config)
    
    # Use enhanced data fetching
    data = components['fetcher'].get_data('SPY')
    
    # Use cached features
    features = components['feature_cache'].cached('rsi')(compute_rsi)(data)
"""

import logging
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


def integrate_robust_fetcher(data_source: Any, config: Dict = None) -> Any:
    """
    Wrap data source with RobustFetcher for retries and caching.
    
    Args:
        data_source: Original data source (e.g., EnhancedDataSource)
        config: Optional config dict
        
    Returns:
        RobustFetcher wrapping the data source
    """
    try:
        from backend.robust_fetcher import RobustFetcher
        
        fetcher = RobustFetcher(
            data_source=data_source,
            cache_enabled=config.get('cache_enabled', True) if config else True,
            max_retries=config.get('max_retries', 3) if config else 3,
            rate_limit_calls=config.get('rate_limit_calls', 60) if config else 60
        )
        
        logger.info("✅ RobustFetcher integrated")
        return fetcher
        
    except ImportError as e:
        logger.warning(f"RobustFetcher not available: {e}")
        return data_source


def integrate_feature_cache(config: Dict = None) -> Any:
    """
    Create FeatureCache for expensive computations.
    
    Args:
        config: Optional config dict
        
    Returns:
        FeatureCache instance
    """
    try:
        from backend.feature_cache import FeatureCache
        
        cache = FeatureCache(
            max_size=config.get('max_size', 1000) if config else 1000,
            ttl_seconds=config.get('ttl_seconds', 60) if config else 60,
            max_memory_mb=config.get('max_memory_mb', 100) if config else 100
        )
        
        logger.info("✅ FeatureCache integrated")
        return cache
        
    except ImportError as e:
        logger.warning(f"FeatureCache not available: {e}")
        return None


def integrate_calibration(bot: Any, config: Dict = None) -> Any:
    """
    Ensure CalibrationTracker is properly initialized in bot.
    
    Issue 1 fix: Now includes PnL-based calibration alongside direction.
    
    Args:
        bot: UnifiedOptionsBot instance
        config: Optional config dict
        
    Returns:
        CalibrationTracker instance (also attached to bot)
    """
    try:
        from backend.calibration_tracker import CalibrationTracker
        
        if hasattr(bot, 'calibration_tracker') and bot.calibration_tracker is not None:
            logger.info("✅ CalibrationTracker already initialized")
            return bot.calibration_tracker
        
        tracker = CalibrationTracker(
            direction_horizon=config.get('direction_horizon', config.get('horizon_minutes', 15)) if config else 15,
            pnl_horizon=config.get('pnl_horizon', 60) if config else 60,
            buffer_size=config.get('buffer_size', 1000) if config else 1000,
            use_hybrid=config.get('use_hybrid', True) if config else True
        )
        
        bot.calibration_tracker = tracker
        logger.info("✅ CalibrationTracker integrated (direction + PnL calibration)")
        return tracker
        
    except ImportError as e:
        logger.warning(f"CalibrationTracker not available: {e}")
        return None


def integrate_risk_manager(config: Dict = None) -> Any:
    """
    Create centralized RiskManager for all risk calculations.
    
    Issue 6 fix: Single source of truth for risk limits.
    
    Args:
        config: Optional config dict
        
    Returns:
        RiskManager instance
    """
    try:
        from backend.risk_manager import RiskManager
        
        manager = RiskManager(config=config)
        logger.info("✅ RiskManager integrated (centralized risk)")
        return manager
        
    except ImportError as e:
        logger.warning(f"RiskManager not available: {e}")
        return None


def integrate_regime_mapper(config: Dict = None) -> Any:
    """
    Create RegimeMapper for canonical regime classification.
    
    Issue 3 fix: Bridge between HMM states and VIX buckets.
    
    Args:
        config: Optional config dict
        
    Returns:
        RegimeMapper instance
    """
    try:
        from backend.regime_mapper import RegimeMapper
        
        threshold_config = config.get('threshold_management', {}) if config else {}
        
        mapper = RegimeMapper(
            vix_thresholds=tuple(threshold_config.get('vix_thresholds', [12, 15, 20, 25, 35])),
            hmm_adjustment_range=threshold_config.get('hmm_adjustment_range', 0.1),
            freeze_hmm_structure=threshold_config.get('freeze_hmm_structure', True)
        )
        
        logger.info("✅ RegimeMapper integrated (canonical regime mapping)")
        return mapper
        
    except ImportError as e:
        logger.warning(f"RegimeMapper not available: {e}")
        return None


def integrate_trading_environment(broker: Any = None, config: Dict = None, mode: str = 'simulation') -> Any:
    """
    Create unified TradingEnvironment for sim/live parity.
    
    Issue 7 fix: Same interface for simulation and live trading.
    
    Args:
        broker: Broker adapter (optional, will create SimulatedBroker if None)
        config: Optional config dict
        mode: 'simulation', 'paper', or 'live'
        
    Returns:
        TradingEnvironment instance
    """
    try:
        from backend.trading_environment import create_trading_environment
        
        initial_balance = 5000.0
        if config:
            initial_balance = config.get('trading', {}).get('initial_balance', 5000.0)
        
        env = create_trading_environment(
            mode=mode,
            initial_balance=initial_balance,
            config=config
        )
        
        logger.info(f"✅ TradingEnvironment integrated (mode={mode})")
        return env
        
    except ImportError as e:
        logger.warning(f"TradingEnvironment not available: {e}")
        return None


def integrate_health_monitoring(config: Dict = None) -> Dict[str, Any]:
    """
    Create health monitoring components.
    
    Args:
        config: Optional config dict
        
    Returns:
        Dict with 'health_check', 'model_health', 'alert_manager'
    """
    components = {}
    
    try:
        from backend.health_checks import HealthCheckSystem, AlertManager
        from backend.model_health import ModelHealthMonitor
        
        # Alert manager
        alert_config = config.get('alerts', {}) if config else {}
        components['alert_manager'] = AlertManager(
            slack_webhook=alert_config.get('slack_webhook'),
            alert_cooldown_minutes=alert_config.get('cooldown_minutes', 30)
        )
        
        # Health check system
        components['health_check'] = HealthCheckSystem(
            alert_manager=components['alert_manager'],
            max_data_age_seconds=config.get('max_data_age_seconds', 300) if config else 300,
            min_health_score=config.get('min_health_score', 0.5) if config else 0.5
        )
        
        # Model health monitor
        components['model_health'] = ModelHealthMonitor(
            error_threshold=config.get('error_threshold', 0.02) if config else 0.02,
            drift_threshold=config.get('drift_threshold', 0.05) if config else 0.05
        )
        
        logger.info("✅ Health monitoring integrated")
        
    except ImportError as e:
        logger.warning(f"Health monitoring not available: {e}")
    
    return components


def integrate_async_operations(data_source: Any, bot: Any, symbols: list = None, config: Dict = None) -> Any:
    """
    Create AsyncCycleRunner for parallel operations.
    
    Args:
        data_source: Data source for fetching
        bot: Trading bot instance
        symbols: List of symbols to fetch
        config: Optional config dict
        
    Returns:
        AsyncCycleRunner instance
    """
    try:
        from backend.async_operations import AsyncCycleRunner
        
        runner = AsyncCycleRunner(
            data_source=data_source,
            bot=bot,
            symbols=symbols or ['SPY'],
            max_workers=config.get('max_workers', 4) if config else 4
        )
        
        logger.info("✅ AsyncOperations integrated")
        return runner
        
    except ImportError as e:
        logger.warning(f"AsyncOperations not available: {e}")
        return None


def integrate_rl_enhancements(policy: Any, config: Dict = None) -> Dict[str, Any]:
    """
    Integrate RL enhancements with existing policy.
    
    Args:
        policy: RLTradingPolicy instance
        config: Optional config dict
        
    Returns:
        Dict with 'per_buffer', 'reward_calc', 'trainer'
    """
    components = {}
    
    try:
        from backend.rl_enhancements import (
            PrioritizedReplayBuffer,
            SharpeRewardCalculator,
            EnhancedPPOTrainer
        )
        
        # Prioritized Experience Replay
        components['per_buffer'] = PrioritizedReplayBuffer(
            capacity=config.get('buffer_capacity', 10000) if config else 10000,
            alpha=config.get('per_alpha', 0.6) if config else 0.6,
            beta=config.get('per_beta', 0.4) if config else 0.4
        )
        
        # Sharpe-based rewards
        components['reward_calc'] = SharpeRewardCalculator(
            pnl_scale=config.get('pnl_scale', 100.0) if config else 100.0,
            sharpe_weight=config.get('sharpe_weight', 0.5) if config else 0.5
        )
        
        # Enhanced trainer
        components['trainer'] = EnhancedPPOTrainer(
            policy=policy,
            per_buffer=components['per_buffer'],
            reward_calculator=components['reward_calc']
        )
        
        logger.info("✅ RL enhancements integrated")
        
    except ImportError as e:
        logger.warning(f"RL enhancements not available: {e}")
    
    return components


def integrate_regime_strategies(config: Dict = None) -> Any:
    """
    Create RegimeStrategyManager for regime-specific trading.
    
    Args:
        config: Optional config dict
        
    Returns:
        RegimeStrategyManager instance
    """
    try:
        from backend.regime_strategies import RegimeStrategyManager, RegimeDetector
        
        detector = RegimeDetector(
            vix_thresholds=tuple(config.get('vix_thresholds', [12, 15, 20, 25, 35])) if config else (12, 15, 20, 25, 35),
            trend_lookback=config.get('trend_lookback', 20) if config else 20,
            use_hmm=config.get('use_hmm', True) if config else True
        )
        
        manager = RegimeStrategyManager(detector=detector)
        logger.info("✅ RegimeStrategyManager integrated")
        return manager
        
    except ImportError as e:
        logger.warning(f"RegimeStrategies not available: {e}")
        return None


def integrate_monitoring(db_path: str = None, config: Dict = None) -> Any:
    """
    Create TradingMonitor for position and rejection tracking.
    
    Args:
        db_path: Path to trades database
        config: Optional config dict
        
    Returns:
        TradingMonitor instance
    """
    try:
        from backend.monitoring_dashboard import TradingMonitor
        
        monitor = TradingMonitor(
            db_path=db_path or "data/paper_trades.db",
            log_buffer_size=config.get('log_buffer_size', 1000) if config else 1000
        )
        
        # Load existing positions
        monitor.load_positions_from_db()
        
        logger.info("✅ TradingMonitor integrated")
        return monitor
        
    except ImportError as e:
        logger.warning(f"TradingMonitor not available: {e}")
        return None


def integrate_regime_models(bot: Any, config: Dict = None) -> Any:
    """
    Create RegimeModelManager for per-regime model training.
    
    Args:
        bot: UnifiedOptionsBot instance
        config: Optional config dict
        
    Returns:
        RegimeModelManager instance
    """
    try:
        from backend.regime_models import RegimeModelManager
        
        manager = RegimeModelManager(
            bot=bot,
            models_dir=config.get('models_dir', 'models/regime_models') if config else 'models/regime_models',
            enable_transfer_learning=config.get('enable_transfer_learning', True) if config else True
        )
        
        logger.info("✅ RegimeModelManager integrated")
        return manager
        
    except ImportError as e:
        logger.warning(f"RegimeModelManager not available: {e}")
        return None


def integrate_all_components(
    bot: Any,
    config: Dict = None,
    symbols: list = None
) -> Dict[str, Any]:
    """
    Integrate all enhanced components into the bot.
    
    This is the main entry point for full integration.
    
    Includes Issue 1-7 fixes:
    - Calibration with PnL tracking (Issue 1)
    - Risk Manager (Issue 6)
    - Regime Mapper (Issue 3)
    
    Args:
        bot: UnifiedOptionsBot instance
        config: Configuration dict (or loads from config.json)
        symbols: List of symbols (default: from config)
        
    Returns:
        Dict with all integrated components
    """
    import json
    from pathlib import Path
    
    # Load config if not provided
    if config is None:
        config_path = Path('config.json')
        if config_path.exists():
            with open(config_path) as f:
                config = json.load(f)
        else:
            config = {}
    
    # Get symbols from config if not provided
    if symbols is None:
        symbols = config.get('data_fetching', {}).get('symbols', ['SPY'])
    
    components = {
        'config': config,
        'symbols': symbols
    }
    
    # 1. Robust data fetching
    if hasattr(bot, 'data_source'):
        components['fetcher'] = integrate_robust_fetcher(
            bot.data_source,
            config.get('fetcher', {})
        )
        # Optionally replace bot's data source
        # bot.data_source = components['fetcher']
    
    # 2. Feature caching
    components['feature_cache'] = integrate_feature_cache(
        config.get('feature_cache', {})
    )
    
    # 3. Calibration (Issue 1 fix: now includes PnL calibration)
    components['calibration'] = integrate_calibration(
        bot,
        config.get('calibration', {})
    )
    
    # 4. Health monitoring
    health_components = integrate_health_monitoring(
        config.get('health', {})
    )
    components.update(health_components)
    
    # 5. Async operations
    if hasattr(bot, 'data_source'):
        components['async_runner'] = integrate_async_operations(
            bot.data_source,
            bot,
            symbols,
            config.get('async', {})
        )
    
    # 6. RL enhancements
    if hasattr(bot, 'rl_policy') and bot.rl_policy is not None:
        rl_components = integrate_rl_enhancements(
            bot.rl_policy,
            config.get('rl', {})
        )
        components.update(rl_components)
    
    # 7. Regime-specific strategies
    components['regime_strategies'] = integrate_regime_strategies(
        config.get('regime', {})
    )
    
    # 8. Trading monitor (stuck positions, rejections)
    components['monitor'] = integrate_monitoring(
        db_path=config.get('data', {}).get('trades_db', 'data/paper_trades.db'),
        config=config.get('monitoring', {})
    )
    
    # 9. Regime-specific models
    components['regime_models'] = integrate_regime_models(
        bot,
        config.get('regime_models', {})
    )
    
    # ==========================================================================
    # NEW COMPONENTS (Issues 1-7 fixes)
    # ==========================================================================
    
    # 10. Risk Manager (Issue 6 fix: centralized risk)
    components['risk_manager'] = integrate_risk_manager(config)
    
    # 11. Regime Mapper (Issue 3 fix: canonical regime mapping)
    components['regime_mapper'] = integrate_regime_mapper(config)
    
    logger.info(f"✅ All components integrated: {list(components.keys())}")
    
    return components


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def patch_bot_with_components(bot: Any, components: Dict[str, Any]) -> None:
    """
    Patch bot methods to use integrated components.
    
    This modifies the bot in-place to use the enhanced components.
    
    Args:
        bot: UnifiedOptionsBot instance
        components: Dict from integrate_all_components()
    """
    # Patch data fetching to use RobustFetcher
    if 'fetcher' in components and components['fetcher'] is not None:
        original_data_source = bot.data_source
        bot.data_source = components['fetcher']
        bot._original_data_source = original_data_source
        logger.info("📦 Patched bot.data_source with RobustFetcher")
    
    # Patch feature computation to use cache
    if 'feature_cache' in components and components['feature_cache'] is not None:
        cache = components['feature_cache']
        
        # Cache expensive feature computations
        if hasattr(bot, 'compute_features'):
            original_compute = bot.compute_features
            bot.compute_features = cache.cached('compute_features')(original_compute)
            logger.info("📦 Patched bot.compute_features with caching")
    
    # Attach health monitoring
    if 'health_check' in components:
        bot.health_check = components['health_check']
    if 'model_health' in components:
        bot.model_health = components['model_health']
    
    # Attach RL enhancements
    if 'trainer' in components:
        bot.enhanced_trainer = components['trainer']
    if 'per_buffer' in components:
        bot.per_buffer = components['per_buffer']
    if 'reward_calc' in components:
        bot.reward_calculator = components['reward_calc']
    
    # Attach regime strategies
    if 'regime_strategies' in components and components['regime_strategies'] is not None:
        bot.regime_manager = components['regime_strategies']
        logger.info("📦 Attached regime strategy manager to bot")
    
    # Attach trading monitor
    if 'monitor' in components and components['monitor'] is not None:
        bot.trading_monitor = components['monitor']
        logger.info("📦 Attached trading monitor to bot")
    
    # Attach regime models
    if 'regime_models' in components and components['regime_models'] is not None:
        bot.regime_model_manager = components['regime_models']
        logger.info("📦 Attached regime model manager to bot")
    
    # ==========================================================================
    # NEW COMPONENTS (Issues 1-7 fixes)
    # ==========================================================================
    
    # Attach Risk Manager (Issue 6 fix)
    if 'risk_manager' in components and components['risk_manager'] is not None:
        bot.risk_manager = components['risk_manager']
        logger.info("📦 Attached centralized risk manager to bot")
    
    # Attach Regime Mapper (Issue 3 fix)
    if 'regime_mapper' in components and components['regime_mapper'] is not None:
        bot.regime_mapper = components['regime_mapper']
        logger.info("📦 Attached regime mapper to bot")
    
    logger.info("✅ Bot patched with all components")


def create_enhanced_bot(
    bot_class: type,
    config: Dict = None,
    **kwargs
) -> Any:
    """
    Factory function to create a bot with all enhancements pre-integrated.
    
    Args:
        bot_class: Bot class to instantiate (e.g., UnifiedOptionsBot)
        config: Configuration dict
        **kwargs: Additional arguments for bot constructor
        
    Returns:
        Enhanced bot instance
    """
    # Create base bot
    bot = bot_class(**kwargs)
    
    # Integrate all components
    components = integrate_all_components(bot, config)
    
    # Patch bot with components
    patch_bot_with_components(bot, components)
    
    # Store components reference
    bot._enhanced_components = components
    
    return bot

