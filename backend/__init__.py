# BITX Trading Bot Package
# 
# Contains trading system components:
#   - live_trading_engine.py  -> Live trading cycle orchestration
#   - shadow_trading_bridge.py -> Tradier shadow trading sync
#   - position_recovery.py    -> Position management on restart
#   - tradier_trading_system.py -> Tradier API integration
#   - paper_trading_system.py -> Paper trading simulation
#   - data_sources.py         -> Market data fetching
#   - liquidity_validator.py  -> Trade liquidity checks
#
# V2 Architecture Components:
#   - arch_config.py          -> Central architecture configuration
#   - arch_v2.py              -> V2 architecture integration
#   - safety_filter.py        -> Entry safety filter
#   - unified_exit_manager.py -> Unified exit management
#   - predictor_manager.py    -> Predictor with uncertainty
#   - regime_embedding.py     -> HMM regime embeddings
#   - horizon_alignment.py    -> Time horizon alignment

# Lazy imports to avoid circular dependencies
# Use: from backend.live_trading_engine import LiveTradingEngine
# Or:  from backend import LiveTradingEngine (triggers lazy load)

def __getattr__(name):
    """Lazy loading of module components."""
    # Live trading components
    if name == 'LiveTradingEngine':
        from backend.live_trading_engine import LiveTradingEngine
        return LiveTradingEngine
    elif name == 'CycleStats':
        from backend.live_trading_engine import CycleStats
        return CycleStats
    elif name == 'log_subsystem_health':
        from backend.live_trading_engine import log_subsystem_health
        return log_subsystem_health
    elif name == 'ShadowTradingBridge':
        from backend.shadow_trading_bridge import ShadowTradingBridge
        return ShadowTradingBridge
    elif name == 'ShadowTradeResult':
        from backend.shadow_trading_bridge import ShadowTradeResult
        return ShadowTradeResult
    elif name == 'PositionRecovery':
        from backend.position_recovery import PositionRecovery
        return PositionRecovery
    elif name == 'RecoveryConfig':
        from backend.position_recovery import RecoveryConfig
        return RecoveryConfig
    elif name == 'evaluate_existing_positions':
        from backend.position_recovery import evaluate_existing_positions
        return evaluate_existing_positions
    
    # V2 Architecture components
    elif name == 'ArchConfig':
        from backend.arch_config import ArchConfig
        return ArchConfig
    elif name == 'init_arch_config':
        from backend.arch_config import init_arch_config
        return init_arch_config
    elif name == 'ArchV2':
        from backend.arch_v2 import ArchV2
        return ArchV2
    elif name == 'init_arch_v2':
        from backend.arch_v2 import init_arch_v2
        return init_arch_v2
    elif name == 'get_arch_v2':
        from backend.arch_v2 import get_arch_v2
        return get_arch_v2
    elif name == 'EntrySafetyFilter':
        from backend.safety_filter import EntrySafetyFilter
        return EntrySafetyFilter
    elif name == 'UnifiedExitManager':
        from backend.unified_exit_manager import UnifiedExitManager
        return UnifiedExitManager
    elif name == 'PredictorManager':
        from backend.predictor_manager import PredictorManager
        return PredictorManager
    elif name == 'RegimeEmbedding':
        from backend.regime_embedding import RegimeEmbedding
        return RegimeEmbedding
    elif name == 'HorizonConfig':
        from backend.horizon_alignment import HorizonConfig
        return HorizonConfig
    
    raise AttributeError(f"module 'backend' has no attribute '{name}'")
