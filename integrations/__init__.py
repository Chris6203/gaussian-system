"""
Integrations Package for Gaussian System
========================================

External integrations and adapters:
- quantor: Quantor-MTFuzz features (fuzzy sizing, regime filter, volatility)
"""

from . import quantor

__all__ = ['quantor']
