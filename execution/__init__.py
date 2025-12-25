"""
Liquidity-Aware Execution Layer

Professional-grade execution module that treats liquidity as a first-class signal.
Routes orders through an execution layer that:
  1. Selects tradeable contracts
  2. Sizes by displayed/expected liquidity  
  3. Works midpoint-pegged child orders that adapt if they don't fill
  4. Falls back to tighter verticals if single-leg slippage is poor
"""

from execution.liquidity_exec import LiquidityExecutor, LiquidityRules, OrderIntent
from execution.tradier_adapter import TradierAdapter

__all__ = ['LiquidityExecutor', 'LiquidityRules', 'OrderIntent', 'TradierAdapter']


