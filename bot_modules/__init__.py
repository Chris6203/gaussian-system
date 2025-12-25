"""
Bot Modules
===========

Modular components extracted from the monolithic UnifiedOptionsBot.

This package provides a clean separation of concerns:
- neural_networks: NN architectures (TCN, LSTM, Bayesian layers, predictor)
- gaussian_processor: Gaussian kernel feature processing
- features: Feature computation and engineering
- signals: Signal generation and combination
- market_utils: Market hours, device management, time utilities
- calibration_utils: Confidence calibration (Platt, bucket, metrics)

Usage:
    from bot_modules.neural_networks import UnifiedOptionsPredictor, BayesianLinear
    from bot_modules.gaussian_processor import GaussianKernelProcessor
    from bot_modules.market_utils import MarketHours, DeviceManager
    from bot_modules.calibration_utils import HybridCalibrator
    from bot_modules.features import FeatureComputer
    from bot_modules.signals import SignalCombiner
"""

# Version info
__version__ = '3.0.0'

# Direct imports for convenience (lazy loading for heavy modules)
from .market_utils import (
    MarketHours,
    DeviceManager,
    get_market_time,
    is_market_open,
    annualization_scale,
)

from .calibration_utils import (
    CalibrationConfig,
    PlattCalibrator,
    BucketCalibrator,
    CalibrationMetrics,
    CalibrationBuffer,
    HybridCalibrator,
)

from .gaussian_processor import (
    GaussianKernelProcessor,
    compute_gaussian_features,
)

from .features import (
    FeatureComputer,
    FeatureConfig,
    compute_technical_features,
    compute_volatility_features,
    compute_all_features,
)

from .signals import (
    SignalAction,
    SignalState,
    Signal,
    SignalConfig,
    DirectionAnalyzer,
    ConfidenceCalculator,
    SignalStateMachine,
    SignalCombiner,
    combine_signals,
)

# Trade execution
from .trade_execution import (
    TradeAction,
    OrderTypeMapper,
    PositionSizeConfig,
    PositionSizer,
    ExecutionOutcome,
    ExecutionTracker,
    TradeExecutor,
)

# Online learning
from .online_learning import (
    StoredPrediction,
    PredictionBuffer,
    Experience,
    ExperienceReplayBuffer,
    OnlineLearningConfig,
    OnlineLearner,
    ContinuousLearningManager,
)


# Lazy imports for heavy neural network modules
def get_neural_networks():
    """Get neural network classes (lazy load to avoid heavy imports)."""
    from .neural_networks import (
        BayesianLinear,
        RBFKernelLayer,
        TCNBlock,
        OptionsTCN,
        OptionsLSTM,
        ResidualBlock,
        UnifiedOptionsPredictor,
    )
    return {
        'BayesianLinear': BayesianLinear,
        'RBFKernelLayer': RBFKernelLayer,
        'TCNBlock': TCNBlock,
        'OptionsTCN': OptionsTCN,
        'OptionsLSTM': OptionsLSTM,
        'ResidualBlock': ResidualBlock,
        'UnifiedOptionsPredictor': UnifiedOptionsPredictor,
    }


# All exports
__all__ = [
    # Version
    '__version__',
    
    # Market utilities
    'MarketHours',
    'DeviceManager',
    'get_market_time',
    'is_market_open',
    'annualization_scale',
    
    # Calibration
    'CalibrationConfig',
    'PlattCalibrator',
    'BucketCalibrator',
    'CalibrationMetrics',
    'CalibrationBuffer',
    'HybridCalibrator',
    
    # Gaussian processor
    'GaussianKernelProcessor',
    'compute_gaussian_features',
    
    # Features
    'FeatureComputer',
    'FeatureConfig',
    'compute_technical_features',
    'compute_volatility_features',
    'compute_all_features',
    
    # Signals
    'SignalAction',
    'SignalState',
    'Signal',
    'SignalConfig',
    'DirectionAnalyzer',
    'ConfidenceCalculator',
    'SignalStateMachine',
    'SignalCombiner',
    'combine_signals',
    
    # Trade execution
    'TradeAction',
    'OrderTypeMapper',
    'PositionSizeConfig',
    'PositionSizer',
    'ExecutionOutcome',
    'ExecutionTracker',
    'TradeExecutor',
    
    # Online learning
    'StoredPrediction',
    'PredictionBuffer',
    'Experience',
    'ExperienceReplayBuffer',
    'OnlineLearningConfig',
    'OnlineLearner',
    'ContinuousLearningManager',
    
    # Lazy loaders
    'get_neural_networks',
]
