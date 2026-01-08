#!/usr/bin/env python3
"""
RL Policy Factory

Unified factory for creating RL policies based on configuration.

Supported architectures:
- standard: UnifiedRLPolicy (MLP, default)
- game_ai: GameAIRLPolicy (LSTM + Attention, like OpenAI Five)
- alphastar: AlphaStarRLPolicy (Transformer + Entity encoding)

Usage:
    # Via env var
    RL_ARCHITECTURE=alphastar python scripts/train_time_travel.py

    # Via code
    from backend.rl_factory import create_rl_policy
    policy = create_rl_policy(arch='alphastar', device='cuda')
"""

import os
import logging
from typing import Optional

logger = logging.getLogger(__name__)


def create_rl_policy(
    arch: Optional[str] = None,
    state_dim: int = 22,
    device: str = 'cpu',
    **kwargs
):
    """
    Create RL policy based on architecture choice.

    Args:
        arch: Architecture type. Options:
            - 'standard' or None: UnifiedRLPolicy (MLP)
            - 'game_ai': GameAIRLPolicy (LSTM + Attention)
            - 'alphastar': AlphaStarRLPolicy (Transformer)
        state_dim: State dimension (default 22)
        device: torch device
        **kwargs: Passed to policy constructor

    Returns:
        RL policy instance
    """
    # Get architecture from env if not specified
    if arch is None:
        arch = os.environ.get('RL_ARCHITECTURE', 'standard').lower()

    arch = arch.lower().strip()

    if arch in ('game_ai', 'gameai', 'openai_five', 'dota'):
        from backend.game_ai_policy import GameAIRLPolicy
        logger.info("🎮 Creating Game-AI (OpenAI Five) style RL policy")
        return GameAIRLPolicy(state_dim=state_dim, device=device, **kwargs)

    elif arch in ('alphastar', 'alpha_star', 'starcraft', 'transformer'):
        from backend.alphastar_policy import AlphaStarRLPolicy
        logger.info("⭐ Creating AlphaStar style RL policy")
        return AlphaStarRLPolicy(state_dim=state_dim, device=device, **kwargs)

    else:
        # Default: standard MLP policy
        from backend.unified_rl_policy import UnifiedRLPolicy
        logger.info("🧠 Creating standard RL policy (MLP)")
        return UnifiedRLPolicy(device=device, **kwargs)


def get_available_architectures():
    """Return list of available RL architectures"""
    return ['standard', 'game_ai', 'alphastar']


def describe_architecture(arch: str) -> str:
    """Get description of architecture"""
    descriptions = {
        'standard': 'Simple MLP with optional GRU/Attention (64-128 hidden)',
        'game_ai': 'OpenAI Five style: LSTM memory + Attention + Separate Actor-Critic (256 hidden)',
        'alphastar': 'DeepMind AlphaStar style: Transformer + Entity encoding + RoPE (256 hidden)'
    }
    return descriptions.get(arch.lower(), 'Unknown architecture')


if __name__ == '__main__':
    print("Available RL Architectures:")
    print("-" * 60)
    for arch in get_available_architectures():
        print(f"  {arch}: {describe_architecture(arch)}")
    print()

    # Quick test each
    import numpy as np

    for arch in get_available_architectures():
        print(f"Testing {arch}...")
        policy = create_rl_policy(arch=arch, state_dim=22, device='cpu')
        policy.reset()

        state = np.random.randn(22).astype(np.float32)
        action, info = policy.get_action(state)
        print(f"  ✅ action={policy.ACTION_NAMES[action]}, value={info['value']:.3f}")

    print("\n✅ All architectures work!")
