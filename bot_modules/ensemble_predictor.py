"""
Phase 44: Ensemble Predictor Module

Combines multiple models for improved prediction accuracy:
1. Ensemble Stacking (TCN + LSTM + XGBoost + meta-learner)
2. LSTM-XGBoost Hybrid
3. Multi-Indicator Stacking
4. Attention-weighted predictions
"""

import os
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Optional, Tuple
import xgboost as xgb
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
import pickle
from pathlib import Path


class AttentionLayer(nn.Module):
    """Self-attention layer for weighting predictions."""

    def __init__(self, hidden_dim: int = 64):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
            nn.Softmax(dim=1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: [batch, seq, hidden]
        weights = self.attention(x)  # [batch, seq, 1]
        return (x * weights).sum(dim=1)  # [batch, hidden]


class EnsemblePredictor:
    """
    Ensemble predictor combining multiple models.

    Models:
    - TCN (Temporal Convolutional Network)
    - LSTM
    - XGBoost
    - Meta-learner (Ridge regression)
    """

    def __init__(self, feature_dim: int = 50, hidden_dim: int = 64, device: str = 'cpu'):
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.device = device

        # TCN component (simplified)
        self.tcn = nn.Sequential(
            nn.Conv1d(feature_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1, dilation=2),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        ).to(device)

        # LSTM component
        self.lstm = nn.LSTM(
            input_size=feature_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=0.2
        ).to(device)

        # Attention layer
        self.attention = AttentionLayer(hidden_dim).to(device)

        # XGBoost model
        self.xgb_model = None
        self.xgb_scaler = StandardScaler()

        # Meta-learner
        self.meta_learner = Ridge(alpha=1.0)
        self.meta_scaler = StandardScaler()

        # Training data buffers
        self.train_features = []
        self.train_labels = []
        self.is_fitted = False

    def _tcn_forward(self, x: torch.Tensor) -> torch.Tensor:
        """TCN forward pass. x shape: [batch, seq, features]"""
        x = x.permute(0, 2, 1)  # [batch, features, seq]
        out = self.tcn(x)  # [batch, hidden, 1]
        return out.squeeze(-1)  # [batch, hidden]

    def _lstm_forward(self, x: torch.Tensor) -> torch.Tensor:
        """LSTM forward pass with attention."""
        out, _ = self.lstm(x)  # [batch, seq, hidden]
        return self.attention(out)  # [batch, hidden]

    def _prepare_xgb_features(self, sequence: np.ndarray) -> np.ndarray:
        """Prepare features for XGBoost from sequence."""
        # Use last values, mean, std, min, max across sequence
        if len(sequence.shape) == 3:
            # [batch, seq, features]
            last = sequence[:, -1, :]  # Last timestep
            mean = sequence.mean(axis=1)
            std = sequence.std(axis=1)
            min_val = sequence.min(axis=1)
            max_val = sequence.max(axis=1)
            return np.concatenate([last, mean, std, min_val, max_val], axis=1)
        else:
            # [seq, features]
            last = sequence[-1, :]
            mean = sequence.mean(axis=0)
            std = sequence.std(axis=0)
            min_val = sequence.min(axis=0)
            max_val = sequence.max(axis=0)
            return np.concatenate([last, mean, std, min_val, max_val])

    def train_xgboost(self, sequences: np.ndarray, labels: np.ndarray):
        """Train XGBoost on accumulated data."""
        if len(sequences) < 100:
            return False

        # Prepare features
        X = np.array([self._prepare_xgb_features(seq) for seq in sequences])
        y = labels

        # Scale features
        X_scaled = self.xgb_scaler.fit_transform(X)

        # Train XGBoost
        self.xgb_model = xgb.XGBRegressor(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            objective='reg:squarederror',
            n_jobs=-1,
            random_state=42
        )
        self.xgb_model.fit(X_scaled, y)
        return True

    def predict_ensemble(self, sequence: torch.Tensor) -> Dict[str, float]:
        """
        Generate ensemble predictions.

        Returns:
            Dict with predictions from each model and ensemble
        """
        results = {}

        with torch.no_grad():
            # Ensure batch dimension
            if len(sequence.shape) == 2:
                sequence = sequence.unsqueeze(0)

            # TCN prediction
            tcn_hidden = self._tcn_forward(sequence)
            tcn_pred = tcn_hidden.mean(dim=1).item()
            results['tcn'] = tcn_pred

            # LSTM prediction
            lstm_hidden = self._lstm_forward(sequence)
            lstm_pred = lstm_hidden.mean(dim=1).item()
            results['lstm'] = lstm_pred

            # XGBoost prediction
            if self.xgb_model is not None:
                seq_np = sequence.cpu().numpy()
                xgb_features = self._prepare_xgb_features(seq_np[0])
                xgb_features_scaled = self.xgb_scaler.transform([xgb_features])
                xgb_pred = self.xgb_model.predict(xgb_features_scaled)[0]
                results['xgboost'] = float(xgb_pred)
            else:
                results['xgboost'] = 0.0

            # Simple ensemble (average)
            preds = [results['tcn'], results['lstm'], results['xgboost']]
            results['ensemble_avg'] = np.mean(preds)

            # Weighted ensemble (if meta-learner trained)
            if self.is_fitted:
                meta_input = np.array([[results['tcn'], results['lstm'], results['xgboost']]])
                meta_input_scaled = self.meta_scaler.transform(meta_input)
                results['ensemble_meta'] = float(self.meta_learner.predict(meta_input_scaled)[0])
            else:
                results['ensemble_meta'] = results['ensemble_avg']

        return results

    def add_training_sample(self, sequence: np.ndarray, label: float):
        """Add a training sample for meta-learner."""
        self.train_features.append(sequence)
        self.train_labels.append(label)

    def fit_meta_learner(self):
        """Fit the meta-learner on accumulated predictions."""
        if len(self.train_features) < 50:
            return False

        # Get predictions from base models
        meta_features = []
        for seq in self.train_features:
            seq_tensor = torch.tensor(seq, dtype=torch.float32, device=self.device)
            preds = self.predict_ensemble(seq_tensor)
            meta_features.append([preds['tcn'], preds['lstm'], preds['xgboost']])

        meta_features = np.array(meta_features)
        labels = np.array(self.train_labels)

        # Scale and fit
        meta_features_scaled = self.meta_scaler.fit_transform(meta_features)
        self.meta_learner.fit(meta_features_scaled, labels)
        self.is_fitted = True
        return True

    def save(self, path: str):
        """Save ensemble models."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        # Save PyTorch components
        torch.save({
            'tcn_state': self.tcn.state_dict(),
            'lstm_state': self.lstm.state_dict(),
            'attention_state': self.attention.state_dict(),
        }, path / 'ensemble_nn.pt')

        # Save XGBoost
        if self.xgb_model is not None:
            self.xgb_model.save_model(str(path / 'xgb_model.json'))
            with open(path / 'xgb_scaler.pkl', 'wb') as f:
                pickle.dump(self.xgb_scaler, f)

        # Save meta-learner
        if self.is_fitted:
            with open(path / 'meta_learner.pkl', 'wb') as f:
                pickle.dump({
                    'model': self.meta_learner,
                    'scaler': self.meta_scaler
                }, f)

    def load(self, path: str):
        """Load ensemble models."""
        path = Path(path)

        # Load PyTorch components
        if (path / 'ensemble_nn.pt').exists():
            state = torch.load(path / 'ensemble_nn.pt', map_location=self.device)
            self.tcn.load_state_dict(state['tcn_state'])
            self.lstm.load_state_dict(state['lstm_state'])
            self.attention.load_state_dict(state['attention_state'])

        # Load XGBoost
        if (path / 'xgb_model.json').exists():
            self.xgb_model = xgb.XGBRegressor()
            self.xgb_model.load_model(str(path / 'xgb_model.json'))
            with open(path / 'xgb_scaler.pkl', 'rb') as f:
                self.xgb_scaler = pickle.load(f)

        # Load meta-learner
        if (path / 'meta_learner.pkl').exists():
            with open(path / 'meta_learner.pkl', 'rb') as f:
                data = pickle.load(f)
                self.meta_learner = data['model']
                self.meta_scaler = data['scaler']
                self.is_fitted = True


class GammaExposureSignals:
    """
    Gamma Exposure (GEX) signal generator.

    Uses options volume and open interest to estimate dealer positioning.
    """

    def __init__(self):
        self.gex_history = []
        self.put_call_ratio_history = []

    def calculate_gex_proxy(self,
                           call_volume: float,
                           put_volume: float,
                           call_oi: float,
                           put_oi: float,
                           spot_price: float,
                           vix: float) -> Dict[str, float]:
        """
        Calculate gamma exposure proxy signals.

        In absence of full options chain data, we use volume/OI ratios
        as proxies for dealer positioning.
        """
        # Put/Call volume ratio
        pc_volume_ratio = put_volume / max(call_volume, 1)

        # Put/Call OI ratio
        pc_oi_ratio = put_oi / max(call_oi, 1)

        # Net flow direction (positive = bullish flow)
        net_flow = (call_volume - put_volume) / max(call_volume + put_volume, 1)

        # GEX proxy: high call OI near spot = dealer long gamma (expect pinning)
        # Simplified: use ratio of volumes adjusted by VIX
        gex_proxy = net_flow * (1 - vix/100)

        # Dealer position signal
        # When PC ratio > 1.2, dealers likely short puts = long gamma
        # When PC ratio < 0.8, dealers likely short calls = short gamma
        if pc_oi_ratio > 1.2:
            dealer_gamma = 0.7  # Long gamma (expect mean reversion)
        elif pc_oi_ratio < 0.8:
            dealer_gamma = 0.3  # Short gamma (expect trend)
        else:
            dealer_gamma = 0.5  # Neutral

        # Store history
        self.gex_history.append(gex_proxy)
        self.put_call_ratio_history.append(pc_volume_ratio)

        # Keep last 100
        if len(self.gex_history) > 100:
            self.gex_history = self.gex_history[-100:]
            self.put_call_ratio_history = self.put_call_ratio_history[-100:]

        return {
            'gex_proxy': gex_proxy,
            'pc_volume_ratio': pc_volume_ratio,
            'pc_oi_ratio': pc_oi_ratio,
            'net_flow': net_flow,
            'dealer_gamma': dealer_gamma,
            'gex_momentum': np.mean(self.gex_history[-5:]) if len(self.gex_history) >= 5 else 0
        }


class OrderFlowSignals:
    """
    Order flow imbalance signals.

    Tracks buy/sell pressure and volume dynamics.
    """

    def __init__(self, lookback: int = 20):
        self.lookback = lookback
        self.volume_history = []
        self.price_history = []
        self.bid_ask_history = []

    def calculate_imbalance(self,
                           price: float,
                           volume: float,
                           bid: float = None,
                           ask: float = None,
                           vwap: float = None) -> Dict[str, float]:
        """Calculate order flow imbalance signals."""

        self.price_history.append(price)
        self.volume_history.append(volume)

        if len(self.price_history) > self.lookback:
            self.price_history = self.price_history[-self.lookback:]
            self.volume_history = self.volume_history[-self.lookback:]

        signals = {}

        # Volume-weighted price change
        if len(self.price_history) >= 2:
            price_changes = np.diff(self.price_history)
            volumes = self.volume_history[1:]

            # Buy volume (positive price change)
            buy_volume = sum(v for p, v in zip(price_changes, volumes) if p > 0)
            sell_volume = sum(v for p, v in zip(price_changes, volumes) if p < 0)
            total_volume = buy_volume + sell_volume

            # Order flow imbalance (-1 to 1)
            if total_volume > 0:
                signals['flow_imbalance'] = (buy_volume - sell_volume) / total_volume
            else:
                signals['flow_imbalance'] = 0

            # Volume momentum
            recent_vol = np.mean(self.volume_history[-5:]) if len(self.volume_history) >= 5 else volume
            avg_vol = np.mean(self.volume_history)
            signals['volume_momentum'] = recent_vol / max(avg_vol, 1) - 1
        else:
            signals['flow_imbalance'] = 0
            signals['volume_momentum'] = 0

        # Bid-ask spread signal
        if bid is not None and ask is not None:
            spread = (ask - bid) / price
            self.bid_ask_history.append(spread)
            if len(self.bid_ask_history) > self.lookback:
                self.bid_ask_history = self.bid_ask_history[-self.lookback:]

            avg_spread = np.mean(self.bid_ask_history)
            signals['spread_signal'] = spread / max(avg_spread, 0.0001) - 1
        else:
            signals['spread_signal'] = 0

        # VWAP deviation
        if vwap is not None:
            signals['vwap_deviation'] = (price - vwap) / price
        else:
            signals['vwap_deviation'] = 0

        return signals


class MultiIndicatorStack:
    """
    Multi-indicator stacking for enhanced signals.

    Combines 6+ technical indicators for robust signal generation.
    """

    def __init__(self):
        self.rsi_history = []
        self.macd_history = []

    def calculate_indicators(self, prices: np.ndarray, volumes: np.ndarray = None) -> Dict[str, float]:
        """
        Calculate multiple technical indicators.

        Args:
            prices: Array of recent prices (at least 30 values)
            volumes: Optional array of volumes
        """
        if len(prices) < 30:
            return self._default_indicators()

        signals = {}

        # 1. RSI (14 period)
        signals['rsi'] = self._calculate_rsi(prices, 14)

        # 2. MACD
        macd, signal_line, histogram = self._calculate_macd(prices)
        signals['macd'] = macd
        signals['macd_signal'] = signal_line
        signals['macd_histogram'] = histogram

        # 3. Bollinger Bands position
        bb_upper, bb_middle, bb_lower = self._calculate_bollinger(prices, 20, 2)
        current_price = prices[-1]
        bb_range = bb_upper - bb_lower
        if bb_range > 0:
            signals['bb_position'] = (current_price - bb_lower) / bb_range
        else:
            signals['bb_position'] = 0.5

        # 4. Moving average crossover signals
        sma_5 = np.mean(prices[-5:])
        sma_20 = np.mean(prices[-20:])
        signals['ma_crossover'] = (sma_5 - sma_20) / sma_20

        # 5. Price momentum (rate of change)
        signals['momentum_5'] = (prices[-1] - prices[-5]) / prices[-5] if len(prices) >= 5 else 0
        signals['momentum_10'] = (prices[-1] - prices[-10]) / prices[-10] if len(prices) >= 10 else 0

        # 6. ATR (Average True Range) normalized
        signals['atr_norm'] = self._calculate_atr_normalized(prices, 14)

        # 7. Stochastic oscillator
        signals['stoch_k'], signals['stoch_d'] = self._calculate_stochastic(prices, 14, 3)

        # 8. Volume indicators (if available)
        if volumes is not None and len(volumes) >= 20:
            signals['volume_sma_ratio'] = volumes[-1] / np.mean(volumes[-20:])
            signals['obv_trend'] = self._calculate_obv_trend(prices, volumes)
        else:
            signals['volume_sma_ratio'] = 1.0
            signals['obv_trend'] = 0

        # Composite signal (weighted average of normalized indicators)
        signals['composite'] = self._calculate_composite(signals)

        return signals

    def _default_indicators(self) -> Dict[str, float]:
        """Return default indicator values."""
        return {
            'rsi': 50, 'macd': 0, 'macd_signal': 0, 'macd_histogram': 0,
            'bb_position': 0.5, 'ma_crossover': 0, 'momentum_5': 0,
            'momentum_10': 0, 'atr_norm': 0, 'stoch_k': 50, 'stoch_d': 50,
            'volume_sma_ratio': 1.0, 'obv_trend': 0, 'composite': 0
        }

    def _calculate_rsi(self, prices: np.ndarray, period: int = 14) -> float:
        """Calculate RSI."""
        deltas = np.diff(prices)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)

        avg_gain = np.mean(gains[-period:])
        avg_loss = np.mean(losses[-period:])

        if avg_loss == 0:
            return 100
        rs = avg_gain / avg_loss
        return 100 - (100 / (1 + rs))

    def _calculate_macd(self, prices: np.ndarray) -> Tuple[float, float, float]:
        """Calculate MACD."""
        ema_12 = self._ema(prices, 12)
        ema_26 = self._ema(prices, 26)
        macd = ema_12 - ema_26

        # Signal line (9-period EMA of MACD)
        if len(prices) >= 35:
            macd_series = []
            for i in range(9, len(prices)):
                e12 = self._ema(prices[:i+1], 12)
                e26 = self._ema(prices[:i+1], 26)
                macd_series.append(e12 - e26)
            signal = self._ema(np.array(macd_series), 9)
        else:
            signal = macd

        histogram = macd - signal
        return macd, signal, histogram

    def _ema(self, prices: np.ndarray, period: int) -> float:
        """Calculate Exponential Moving Average."""
        if len(prices) < period:
            return np.mean(prices)

        multiplier = 2 / (period + 1)
        ema = np.mean(prices[:period])

        for price in prices[period:]:
            ema = (price - ema) * multiplier + ema
        return ema

    def _calculate_bollinger(self, prices: np.ndarray, period: int = 20, std_dev: float = 2) -> Tuple[float, float, float]:
        """Calculate Bollinger Bands."""
        sma = np.mean(prices[-period:])
        std = np.std(prices[-period:])
        upper = sma + std_dev * std
        lower = sma - std_dev * std
        return upper, sma, lower

    def _calculate_atr_normalized(self, prices: np.ndarray, period: int = 14) -> float:
        """Calculate normalized ATR."""
        if len(prices) < period + 1:
            return 0

        tr_values = []
        for i in range(1, len(prices)):
            high_low = abs(prices[i] - prices[i-1])  # Simplified (using close-to-close)
            tr_values.append(high_low)

        atr = np.mean(tr_values[-period:])
        return atr / prices[-1]  # Normalized by current price

    def _calculate_stochastic(self, prices: np.ndarray, k_period: int = 14, d_period: int = 3) -> Tuple[float, float]:
        """Calculate Stochastic oscillator."""
        if len(prices) < k_period:
            return 50, 50

        recent = prices[-k_period:]
        low = np.min(recent)
        high = np.max(recent)

        if high == low:
            k = 50
        else:
            k = ((prices[-1] - low) / (high - low)) * 100

        # D is SMA of K (simplified - just use current K)
        d = k  # Simplified
        return k, d

    def _calculate_obv_trend(self, prices: np.ndarray, volumes: np.ndarray) -> float:
        """Calculate OBV trend direction."""
        if len(prices) < 2 or len(volumes) < 2:
            return 0

        obv = 0
        for i in range(1, min(len(prices), len(volumes))):
            if prices[i] > prices[i-1]:
                obv += volumes[i]
            elif prices[i] < prices[i-1]:
                obv -= volumes[i]

        # Return normalized trend
        return np.sign(obv)

    def _calculate_composite(self, signals: Dict[str, float]) -> float:
        """Calculate composite signal from all indicators."""
        # Normalize each to -1 to 1 range and combine
        components = [
            (signals['rsi'] - 50) / 50,  # RSI
            np.clip(signals['macd_histogram'] * 10, -1, 1),  # MACD histogram
            (signals['bb_position'] - 0.5) * 2,  # BB position
            np.clip(signals['ma_crossover'] * 100, -1, 1),  # MA crossover
            np.clip(signals['momentum_5'] * 100, -1, 1),  # Momentum
            (signals['stoch_k'] - 50) / 50,  # Stochastic
        ]
        return np.mean(components)


# Factory function for easy creation
def create_ensemble_predictor(feature_dim: int = 50, device: str = 'cpu') -> EnsemblePredictor:
    """Create and return an ensemble predictor."""
    return EnsemblePredictor(feature_dim=feature_dim, device=device)


def create_signal_generators() -> Tuple[GammaExposureSignals, OrderFlowSignals, MultiIndicatorStack]:
    """Create all signal generators."""
    return GammaExposureSignals(), OrderFlowSignals(), MultiIndicatorStack()
