"""
Proper Confidence Estimation for Options Trading
=================================================

Fix 2 for broken confidence head: Compute P(win) from the model's predictive
distribution rather than relying on the untrained confidence head.

The original confidence head has NO loss function training it (TRAIN_CONFIDENCE_BCE=0
by default), causing it to output inverted values through gradient leakage:
  - 40%+ confidence -> 0% actual win rate
  - 15-20% confidence -> 7.2% actual win rate (highest!)

This module provides mathematically grounded confidence estimates:
  1. P(win) = Phi(mu/sigma) using the normal CDF
  2. Direction entropy as a secondary confidence signal
  3. Combined trade_confidence() that merges both signals

Usage:
    from core.confidence import trade_confidence

    # In training loop or inference:
    output = model(features, sequence)
    conf = trade_confidence(
        return_mean=output['return'],
        return_sigma=output.get('return_std', torch.ones_like(output['return']) * 0.02),
        direction_probs=torch.softmax(output['direction'], dim=-1),
        uncertainty=output.get('uncertainty', None),
    )

Environment Variables:
    CONFIDENCE_USE_PWIN (default "1"): Use P(win) from return distribution
    CONFIDENCE_USE_ENTROPY (default "1"): Use direction entropy
    CONFIDENCE_UNCERTAINTY_ALPHA (default "2.0"): Uncertainty penalty weight
"""

import math
import os
import logging
from typing import Optional

import torch

logger = logging.getLogger(__name__)


def _normal_cdf(x: torch.Tensor) -> torch.Tensor:
    """
    Standard normal CDF: Phi(x) = 0.5 * (1 + erf(x / sqrt(2)))

    This gives P(Z <= x) where Z ~ N(0,1).
    """
    return 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


@torch.no_grad()
def prob_win_from_return(
    mu: torch.Tensor,
    sigma: torch.Tensor,
    eps: float = 1e-6
) -> torch.Tensor:
    """
    Compute P(return > 0) given predicted return distribution N(mu, sigma^2).

    P(return > 0) = P(Z > -mu/sigma) = Phi(mu/sigma)

    where Z is standard normal.

    Args:
        mu: Predicted return mean [B, 1]
        sigma: Predicted return std [B, 1]
        eps: Minimum sigma to avoid division by zero

    Returns:
        P(win) in [0, 1] for each sample [B, 1]
    """
    z = mu / sigma.clamp_min(eps)
    return _normal_cdf(z).clamp(0.0, 1.0)


@torch.no_grad()
def direction_entropy_conf(
    direction_probs: torch.Tensor,
    eps: float = 1e-8
) -> torch.Tensor:
    """
    Compute confidence from entropy of direction probabilities.

    Low entropy = model is certain about direction = high confidence
    High entropy = model is uncertain = low confidence

    Confidence = 1 - (entropy / max_entropy)

    Args:
        direction_probs: Softmax probabilities [B, num_classes]
        eps: Small value to avoid log(0)

    Returns:
        Confidence in [0, 1] for each sample [B, 1]
    """
    p = direction_probs.clamp_min(eps)
    # Entropy: -sum(p * log(p))
    ent = -(p * p.log()).sum(dim=-1, keepdim=True)
    # Max entropy for uniform distribution
    max_ent = math.log(direction_probs.shape[-1])
    # Convert to confidence (1 = certain, 0 = uncertain)
    return (1.0 - (ent / max_ent)).clamp(0.0, 1.0)


@torch.no_grad()
def trade_confidence(
    return_mean: torch.Tensor,
    return_sigma: torch.Tensor,
    direction_probs: torch.Tensor,
    uncertainty: Optional[torch.Tensor] = None,
    uncertainty_alpha: float = 2.0,
) -> torch.Tensor:
    """
    Compute unified trade confidence combining P(win) and direction entropy.

    This replaces the broken neural network confidence head output.

    Confidence = P(win) * dir_conf * exp(-alpha * uncertainty)

    Where:
    - P(win) = Phi(mu/sigma) from the return distribution
    - dir_conf = 1 - normalized_entropy from direction probs
    - uncertainty penalty (optional) from Bayesian model uncertainty

    Args:
        return_mean: Predicted return mean [B, 1]
        return_sigma: Predicted return std [B, 1]
        direction_probs: Softmax direction probabilities [B, num_classes]
        uncertainty: Optional epistemic uncertainty estimate [B, 1]
        uncertainty_alpha: Weight for uncertainty penalty (default 2.0)

    Returns:
        Calibrated confidence in [0, 1] for each sample [B, 1]
    """
    # Check environment variables for configuration
    use_pwin = os.environ.get('CONFIDENCE_USE_PWIN', '1') == '1'
    use_entropy = os.environ.get('CONFIDENCE_USE_ENTROPY', '1') == '1'
    alpha = float(os.environ.get('CONFIDENCE_UNCERTAINTY_ALPHA', str(uncertainty_alpha)))

    # P(win) from return distribution
    if use_pwin:
        p_win = prob_win_from_return(return_mean, return_sigma)
    else:
        p_win = torch.ones_like(return_mean) * 0.5  # Neutral

    # Direction entropy confidence
    if use_entropy:
        dir_conf = direction_entropy_conf(direction_probs)
    else:
        dir_conf = torch.ones_like(return_mean)  # No penalty

    # Combine: multiplicative so both need to agree
    conf = (p_win * dir_conf).clamp(0.0, 1.0)

    # Uncertainty penalty (optional)
    if uncertainty is not None:
        conf = conf * torch.exp(-alpha * uncertainty.clamp_min(0.0))

    return conf.clamp(0.0, 1.0)


# Alias for backward compatibility
compute_confidence = trade_confidence


def get_confidence_from_model_output(output: dict) -> torch.Tensor:
    """
    Convenience function to extract proper confidence from model output dict.

    Falls back to original confidence head if return_std not available.

    Args:
        output: Dict from model.forward() with keys like 'return', 'direction', etc.

    Returns:
        Confidence tensor [B, 1]
    """
    # Check if we should use the new method
    use_new_conf = os.environ.get('USE_PROPER_CONFIDENCE', '1') == '1'

    if not use_new_conf:
        # Fall back to original (broken) confidence head
        return output.get('confidence', torch.tensor([[0.5]]))

    return_mean = output.get('return', output.get('return_mean'))
    return_sigma = output.get('return_std', output.get('volatility'))
    direction = output.get('direction')

    if return_mean is None or direction is None:
        logger.warning("Missing return_mean or direction in output, falling back to original confidence")
        return output.get('confidence', torch.tensor([[0.5]]))

    # Get direction probs (softmax if needed)
    if direction.dim() > 1 and direction.shape[-1] > 1:
        direction_probs = torch.softmax(direction, dim=-1)
    else:
        # Binary case
        direction_probs = torch.cat([1 - direction, direction], dim=-1)

    # Default sigma if not available
    if return_sigma is None:
        return_sigma = torch.ones_like(return_mean) * 0.02  # 2% default

    # Get uncertainty if available
    uncertainty = output.get('uncertainty', output.get('epistemic_uncertainty'))

    return trade_confidence(
        return_mean=return_mean,
        return_sigma=return_sigma,
        direction_probs=direction_probs,
        uncertainty=uncertainty,
    )


# Environment variable documentation
ENV_VARS = {
    'USE_PROPER_CONFIDENCE': 'Use new P(win) confidence instead of broken head (default "1")',
    'CONFIDENCE_USE_PWIN': 'Include P(win) from return distribution (default "1")',
    'CONFIDENCE_USE_ENTROPY': 'Include direction entropy (default "1")',
    'CONFIDENCE_UNCERTAINTY_ALPHA': 'Uncertainty penalty weight (default "2.0")',
}
