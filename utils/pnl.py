"""
Centralized P&L calculation utilities.

IMPORTANT: This module exists to prevent the "100x contract multiplier" bug.

Options contracts represent 100 shares each. When calculating P&L percentage:
- trade['pnl'] is in DOLLARS (already multiplied by 100 × quantity)
- trade['premium_paid'] is per-contract premium (NOT multiplied)

Correct formula:
    cost_basis = premium_paid × quantity × 100
    pnl_pct = pnl / cost_basis

NEVER divide pnl directly by premium_paid without the multiplier!
"""

from typing import Dict, Optional


def compute_pnl_pct(trade: Dict) -> float:
    """
    Returns fractional P&L (e.g. +0.10 = +10%, -0.05 = -5%)
    
    Args:
        trade: Dict with 'pnl', 'premium_paid', and optionally 'quantity'
            - pnl: Dollar P&L (already × 100 × quantity)
            - premium_paid: Per-contract premium price
            - quantity: Number of contracts (default 1)
    
    Returns:
        Fractional P&L (0.10 = 10%, -0.05 = -5%)
    
    Example:
        premium_paid = 1.50
        quantity = 1
        entry = 1.50, exit = 1.35
        pnl = (1.35 - 1.50) × 100 = -15
        
        cost_basis = 1.50 × 1 × 100 = 150
        pnl_pct = -15 / 150 = -0.10 = -10%
    """
    quantity = trade.get("quantity", 1)
    premium = trade.get("premium_paid", 0)
    pnl = trade.get("pnl", 0)
    
    cost_basis = premium * quantity * 100
    
    if cost_basis <= 0:
        return 0.0
    
    return pnl / cost_basis


def compute_pnl_pct_from_values(
    pnl_dollars: float,
    premium_paid: float,
    quantity: int = 1
) -> float:
    """
    Returns fractional P&L from explicit values.
    
    Args:
        pnl_dollars: Dollar P&L (already × 100 × quantity)
        premium_paid: Per-contract premium price
        quantity: Number of contracts
    
    Returns:
        Fractional P&L (0.10 = 10%)
    """
    cost_basis = premium_paid * quantity * 100
    
    if cost_basis <= 0:
        return 0.0
    
    return pnl_dollars / cost_basis


def compute_pnl_pct_from_prices(
    entry_premium: float,
    exit_premium: float
) -> float:
    """
    Returns fractional P&L from entry/exit premiums.
    
    This is the simplest form - just the percentage change.
    
    Args:
        entry_premium: Premium paid at entry
        exit_premium: Premium received at exit
    
    Returns:
        Fractional P&L (0.10 = 10%)
    
    Example:
        entry = 1.50, exit = 1.65
        pnl_pct = (1.65 - 1.50) / 1.50 = 0.10 = +10%
    """
    if entry_premium <= 0:
        return 0.0
    
    return (exit_premium - entry_premium) / entry_premium


def is_winner(trade: Dict) -> bool:
    """
    Returns True if trade has positive P&L.
    
    Uses dollar P&L directly (no percentage calculation needed).
    """
    return trade.get("pnl", 0) > 0 or trade.get("profit_loss", 0) > 0

