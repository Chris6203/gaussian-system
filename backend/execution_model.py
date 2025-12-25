"""
Learned Execution Model for Options Trading

Predicts:
- Fill probability p_fill(T): Probability of full fill within T seconds
- Expected slippage: Expected $ per contract
- Expected time-to-fill: Expected seconds

Learns from real execution outcomes logged during trading.
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, Optional, List
from pathlib import Path
import json
from datetime import datetime

logger = logging.getLogger(__name__)


class ExecutionModel:
    """
    Learned model for execution prediction
    
    Phase 1: Heuristic-based (no training data yet)
    Phase 2: Gradient boosted trees (after collecting 500+ fills)
    Phase 3: Continuous online learning
    """
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        log_path: str = "data/execution_logs.jsonl"
    ):
        """
        Args:
            model_path: Path to trained model (None = use heuristics)
            log_path: Path to execution log file
        """
        self.model_path = model_path
        self.log_path = log_path
        self.model = None
        self.use_learned_model = False
        
        # Try to load trained model if available
        if model_path and Path(model_path).exists():
            try:
                self._load_model(model_path)
                self.use_learned_model = True
                logger.info(f"[EXEC] Loaded trained execution model from {model_path}")
            except Exception as e:
                logger.warning(f"[EXEC] Could not load model: {e}, using heuristics")
        else:
            logger.info("[EXEC] Using heuristic execution model (no training data yet)")
        
        # Ensure log directory exists
        Path(log_path).parent.mkdir(parents=True, exist_ok=True)
    
    def predict(self, features: Dict) -> Dict[str, float]:
        """
        Predict execution outcomes
        
        Args:
            features: Dictionary with:
                - spread_pct: Bid-ask spread %
                - spread_dollars: Bid-ask spread in $
                - bid_size: Contracts at bid
                - ask_size: Contracts at ask
                - open_interest: Total OI
                - volume_today: Volume traded today
                - minutes_since_open: Minutes since market open
                - moneyness: (strike - price) / price
                - delta: Option delta
                - gamma: Option gamma
                - vega: Option vega
                - theta: Option theta
                - regime: HMM regime (optional)
                - days_to_expiry: Days until expiration
                
        Returns:
            Dictionary with:
                - p_fill: Fill probability (0-1)
                - exp_slip: Expected slippage $ per contract
                - exp_ttf: Expected time-to-fill in seconds
        """
        if self.use_learned_model:
            return self._predict_learned(features)
        else:
            return self._predict_heuristic(features)
    
    def _predict_heuristic(self, features: Dict) -> Dict[str, float]:
        """
        Heuristic-based prediction (used until we have training data)
        
        Based on market microstructure principles:
        - Wider spreads → lower fill probability, higher slippage
        - Better depth → higher fill probability
        - Higher OI/volume → better fills
        - Time-of-day matters (open/close worse)
        - Regime matters (stress → worse fills)
        """
        # Extract features
        spread_pct = features.get('spread_pct', 10.0)
        spread_dollars = features.get('spread_dollars', 0.10)
        bid_size = features.get('bid_size', 10)
        ask_size = features.get('ask_size', 10)
        depth_min = min(bid_size, ask_size)
        oi = features.get('open_interest', 100)
        volume = features.get('volume_today', 50)
        minutes_since_open = features.get('minutes_since_open', 60)
        regime = features.get('regime', None)
        
        # === Fill Probability ===
        # Base probability
        p_fill = 0.85
        
        # Regime adjustment (apply early, before other factors)
        if regime:
            if 'stress' in str(regime).lower() or 'panic' in str(regime).lower():
                p_fill -= 0.10  # Worse fills in stress
            elif 'calm' in str(regime).lower():
                p_fill += 0.05  # Better fills in calm
        
        # Adjust for spread (penalize wide spreads)
        if spread_pct > 5.0:
            p_fill -= 0.01 * (spread_pct - 5.0)
        
        # Adjust for depth
        p_fill += 0.002 * min(depth_min, 50)
        
        # Adjust for OI (more OI = more market makers)
        p_fill += 0.0008 * min(oi, 1000)
        
        # Adjust for volume
        p_fill += 0.0006 * min(volume, 2000)
        
        # Penalize open/close periods
        if minutes_since_open <= 10 or minutes_since_open >= 380:
            p_fill -= 0.07
        
        # Clamp to reasonable range
        p_fill = max(0.05, min(0.98, p_fill))
        
        # === Expected Slippage ===
        # Assume we pay ~30% of spread on average with smart limit orders
        # Cap at $0.08 for sanity
        exp_slip = min(spread_dollars * 0.30, 0.08)
        
        # Adjust for market conditions
        if minutes_since_open <= 10 or minutes_since_open >= 380:
            exp_slip *= 1.3  # 30% worse at open/close
        
        if regime and ('stress' in str(regime).lower() or 'panic' in str(regime).lower()):
            exp_slip *= 1.5  # 50% worse in stress
        
        # === Expected Time-to-Fill ===
        # Base time depends on spread
        if spread_pct < 2.0:
            exp_ttf = 5.0  # Very tight, fills fast
        elif spread_pct < 5.0:
            exp_ttf = 15.0  # Normal
        elif spread_pct < 10.0:
            exp_ttf = 45.0  # Wide, takes longer
        else:
            exp_ttf = 120.0  # Very wide, may take minutes
        
        # Adjust for depth and volume
        if depth_min > 20 and volume > 100:
            exp_ttf *= 0.7  # Faster with good depth/volume
        elif depth_min < 5 or volume < 20:
            exp_ttf *= 1.5  # Slower with poor depth/volume
        
        logger.debug(f"[EXEC] Heuristic prediction: p_fill={p_fill:.2%}, "
                    f"slip=${exp_slip:.2f}, ttf={exp_ttf:.0f}s")
        
        return {
            'p_fill': p_fill,
            'exp_slip': exp_slip,
            'exp_ttf': exp_ttf
        }
    
    def _predict_learned(self, features: Dict) -> Dict[str, float]:
        """
        Learned model prediction (used after training on real data)
        
        Uses gradient boosted trees trained on execution logs.
        """
        # TODO: Implement after collecting training data
        # For now, fall back to heuristic
        logger.debug("[EXEC] Learned model not yet implemented, using heuristic")
        return self._predict_heuristic(features)
    
    def log_execution(
        self,
        features: Dict,
        outcome: Dict,
        timestamp: Optional[datetime] = None
    ):
        """
        Log an execution attempt for future model training
        
        Args:
            features: Same dict passed to predict()
            outcome: Dictionary with:
                - filled: bool (did order fill?)
                - time_to_fill: float (seconds, or None if not filled)
                - actual_slippage: float ($ per contract)
                - entry_price: float (price paid)
                - target_price: float (mid price we tried for)
            timestamp: When this execution happened
        """
        if timestamp is None:
            timestamp = datetime.now()
        
        log_entry = {
            'timestamp': timestamp.isoformat(),
            'features': features,
            'outcome': outcome
        }
        
        # Append to log file
        try:
            with open(self.log_path, 'a') as f:
                f.write(json.dumps(log_entry) + '\n')
            logger.debug(f"[EXEC] Logged execution: filled={outcome.get('filled')}")
        except Exception as e:
            logger.error(f"[EXEC] Failed to log execution: {e}")
    
    def train_from_logs(self, min_samples: int = 500) -> bool:
        """
        Train execution model from logged data
        
        Args:
            min_samples: Minimum number of samples needed to train
            
        Returns:
            True if training successful, False otherwise
        """
        # Load logs
        if not Path(self.log_path).exists():
            logger.warning(f"[EXEC] No log file found at {self.log_path}")
            return False
        
        logs = []
        with open(self.log_path, 'r') as f:
            for line in f:
                try:
                    logs.append(json.loads(line))
                except:
                    continue
        
        if len(logs) < min_samples:
            logger.warning(f"[EXEC] Only {len(logs)} samples, need {min_samples} to train")
            return False
        
        logger.info(f"[EXEC] Training execution model on {len(logs)} samples...")
        
        # TODO: Implement training with XGBoost or LightGBM
        # For now, just log that we would train here
        logger.info("[EXEC] Training not yet implemented - collect more data first!")
        
        return False
    
    def _load_model(self, path: str):
        """Load trained model from disk"""
        # TODO: Implement model loading
        pass
    
    def _save_model(self, path: str):
        """Save trained model to disk"""
        # TODO: Implement model saving
        pass

