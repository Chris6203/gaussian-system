import backtrader as bt
import math
import random
from collections import deque


class IndicesMultiTFBot(bt.Strategy):
    params = {
        "symbol": "NAS100.R",
        "lots_per_trade": 0.1,
        "bar_minutes": 1,
        "min_time_between_trades_seconds": 300,
        "max_hold_time_hours": 24,
        "stop_loss_pct": 0.15,
        "take_profit_pct": 0.20,
        "trailing_profit_pct": 0.10,
        "risk_per_trade": 0.02,
        "min_confidence_threshold": 0.30,
        "max_drawdown_pct": 0.20,
        "max_daily_loss_pct": 0.05,
        "sequence_length": 30,
        "mc_samples": 50,
        "kernel_sigma": 2.0,
        "weight_update_interval": 100,
    }

    params_metadata = {
        "symbol": {
            "label": "Symbol",
            "helper_text": "Instrument symbol for logging only (data feed decides actual symbol)",
            "value_type": "str",
        },
        "lots_per_trade": {
            "label": "Lots per trade",
            "helper_text": "Base lot size for each trade",
            "value_type": "float",
        },
        "bar_minutes": {
            "label": "Bar minutes",
            "helper_text": "Bar duration in minutes (used for timeframe mapping)",
            "value_type": "int",
        },
        "min_time_between_trades_seconds": {
            "label": "Min time between trades (sec)",
            "helper_text": "Throttle between entries",
            "value_type": "int",
        },
        "max_hold_time_hours": {
            "label": "Max hold time (hours)",
            "helper_text": "Maximum time to hold a position",
            "value_type": "int",
        },
        "stop_loss_pct": {
            "label": "Stop loss %",
            "helper_text": "Stop loss percentage",
            "value_type": "float",
        },
        "take_profit_pct": {
            "label": "Take profit %",
            "helper_text": "Take profit percentage",
            "value_type": "float",
        },
        "min_confidence_threshold": {
            "label": "Min confidence",
            "helper_text": "Minimum confidence required to trade",
            "value_type": "float",
        },
    }

    def __init__(self) -> None:
        self.order = None
        self.last_trade_dt = None
        self.entry_price = None
        self.entry_bar = None
        self.stop_price = None
        self.target_price = None
        self.peak_value = None
        self.daily_start_value = None
        self.daily_date = None
        self.recent_rewards = deque(maxlen=100)
        self.prediction_history = deque(maxlen=50)
        self.vol_history = deque(maxlen=300)
        self.pending_preds = []
        self.prediction_count = 0

        # Indicators
        self.rsi = bt.ind.RSI(self.data, period=14)
        self.macd = bt.ind.MACD(self.data)
        self.bbands = bt.ind.BollingerBands(self.data, period=20, devfactor=2.0)
        self.sma10 = bt.ind.SMA(self.data, period=10)
        self.sma20 = bt.ind.SMA(self.data, period=20)
        self.sma50 = bt.ind.SMA(self.data, period=50)
        self.sma200 = bt.ind.SMA(self.data, period=200)
        self.vol_sma20 = bt.ind.SMA(self.data.volume, period=20)

        self.prediction_timeframes = {
            "15m": {"minutes": 15, "base_weight": 0.25},
            "30m": {"minutes": 30, "base_weight": 0.20},
            "1h": {"minutes": 60, "base_weight": 0.15},
            "4h": {"minutes": 240, "base_weight": 0.15},
            "12h": {"minutes": 720, "base_weight": 0.10},
            "24h": {"minutes": 1440, "base_weight": 0.10},
            "48h": {"minutes": 2880, "base_weight": 0.05},
        }

        self.tf_state = {}
        for tf_name, cfg in self.prediction_timeframes.items():
            self.tf_state[tf_name] = {
                "weight": cfg["base_weight"],
                "base_weight": cfg["base_weight"],
                "accuracy": deque(maxlen=50),
            }

    def log(self, msg: str) -> None:
        dt = self.data.datetime.datetime(0)
        print(f"{dt.isoformat()} | {msg}")

    def _calc_return(self, period: int) -> float:
        if len(self.data) <= period:
            return 0.0
        prev = self.data.close[-period]
        if prev == 0:
            return 0.0
        return (self.data.close[0] / prev) - 1.0

    def _calc_std_return(self, period: int) -> float:
        if len(self.data) <= period:
            return 0.0
        rets = []
        for i in range(period, 0, -1):
            prev = self.data.close[-i]
            if prev == 0:
                continue
            rets.append((self.data.close[-i + 1] / prev) - 1.0)
        if len(rets) < 2:
            return 0.0
        mean = sum(rets) / len(rets)
        var = sum((r - mean) ** 2 for r in rets) / (len(rets) - 1)
        return math.sqrt(var)

    def _gaussian_smooth(self, values, sigma: float) -> float:
        if not values:
            return 0.0
        n = len(values)
        weights = []
        for i in range(n):
            dist = (n - 1) - i
            weight = math.exp(-0.5 * (dist / sigma) ** 2)
            weights.append(weight)
        total = sum(weights)
        if total == 0:
            return values[-1]
        return sum(v * w for v, w in zip(values, weights)) / total

    def _prediction_consistency(self, lookback: int = 5) -> dict:
        history = list(self.prediction_history)[-lookback:]
        if len(history) < 2:
            return {"samples": len(history), "direction_consistency": 0.0, "consensus": "NEUTRAL"}
        signs = [1 if h > 0 else -1 if h < 0 else 0 for h in history]
        bullish = sum(1 for s in signs if s > 0)
        bearish = sum(1 for s in signs if s < 0)
        total = bullish + bearish
        if total == 0:
            return {"samples": len(history), "direction_consistency": 0.0, "consensus": "NEUTRAL"}
        consensus = "BULLISH" if bullish >= bearish else "BEARISH"
        consistency = max(bullish, bearish) / total
        return {"samples": len(history), "direction_consistency": consistency, "consensus": consensus}

    def _predict_for_horizon(self, horizon_bars: int) -> dict:
        mom = self._calc_return(horizon_bars)
        trend = 0.0
        if self.sma50[0] != 0:
            trend = (self.sma10[0] - self.sma50[0]) / self.sma50[0]
        mean_rev = 0.0
        if self.sma20[0] != 0:
            mean_rev = (self.data.close[0] - self.sma20[0]) / self.sma20[0]

        raw_pred = 0.5 * mom + 0.3 * trend - 0.2 * mean_rev
        vol = self._calc_std_return(20)
        noise = max(vol, 0.001)
        samples = []
        for _ in range(self.p.mc_samples):
            samples.append(raw_pred + random.gauss(0.0, noise))
        mean = sum(samples) / len(samples)
        var = sum((x - mean) ** 2 for x in samples) / max(len(samples) - 1, 1)
        std = math.sqrt(var)

        self.prediction_history.append(mean)
        smoothed = self._gaussian_smooth(list(self.prediction_history), self.p.kernel_sigma)

        conf = 0.6 - (vol * 2.0) + min(abs(smoothed) * 5.0, 0.3)
        conf = max(0.05, min(conf, 0.95))

        return {
            "predicted_return": smoothed,
            "neural_confidence": conf,
            "prediction_uncertainty": std,
            "direction": "UP" if smoothed > 0 else "DOWN" if smoothed < 0 else "FLAT",
        }

    def _update_prediction_accuracy(self) -> None:
        if not self.pending_preds:
            return
        current_bar = len(self.data) - 1
        still_pending = []
        for item in self.pending_preds:
            if current_bar < item["target_bar"]:
                still_pending.append(item)
                continue
            entry = item["entry_price"]
            if entry == 0:
                continue
            actual = (self.data.close[0] / entry) - 1.0
            predicted = item["predicted_return"]
            direction_ok = (actual >= 0 and predicted >= 0) or (actual < 0 and predicted < 0)
            self.tf_state[item["tf"]]["accuracy"].append(1.0 if direction_ok else 0.0)
        self.pending_preds = still_pending

        self.prediction_count += 1
        if self.prediction_count >= self.p.weight_update_interval:
            self.prediction_count = 0
            total = 0.0
            for tf_name, state in self.tf_state.items():
                acc_list = list(state["accuracy"])
                acc = sum(acc_list) / len(acc_list) if acc_list else 0.5
                state["weight"] = state["base_weight"] * (0.5 + 0.5 * acc)
                total += state["weight"]
            if total > 0:
                for tf_name, state in self.tf_state.items():
                    state["weight"] = state["weight"] / total

    def _combine_multi_timeframe(self, predictions: dict) -> dict:
        if not predictions:
            return {"signal_strength": 0.0, "direction": "NEUTRAL", "multi_timeframe_confidence": 0.0}
        weighted_return = 0.0
        weighted_conf = 0.0
        bullish = 0
        bearish = 0
        total_weight = 0.0
        details = {}

        for tf_name, pred in predictions.items():
            weight = self.tf_state[tf_name]["weight"]
            total_weight += weight
            weighted_return += pred["predicted_return"] * weight
            weighted_conf += pred["neural_confidence"] * weight
            if pred["predicted_return"] > 0:
                bullish += 1
            elif pred["predicted_return"] < 0:
                bearish += 1
            details[tf_name] = {
                "direction": "UP" if pred["predicted_return"] > 0 else "DOWN" if pred["predicted_return"] < 0 else "FLAT",
                "return": pred["predicted_return"],
                "confidence": pred["neural_confidence"],
            }

        if total_weight == 0:
            return {"signal_strength": 0.0, "direction": "NEUTRAL", "multi_timeframe_confidence": 0.0}
        weighted_return /= total_weight
        weighted_conf /= total_weight
        agreement = 0.0
        if bullish + bearish > 0:
            agreement = max(bullish, bearish) / (bullish + bearish)
        direction = "BULLISH" if bullish >= bearish else "BEARISH"
        signal_strength = abs(weighted_return) * weighted_conf * agreement

        return {
            "multi_timeframe_return": weighted_return,
            "multi_timeframe_confidence": weighted_conf,
            "agreement_score": agreement,
            "direction": direction,
            "signal_strength": signal_strength,
            "timeframe_details": details,
        }

    def _generate_signal(self, predictions: dict) -> dict:
        vol = self._calc_std_return(30)
        self.vol_history.append(vol)
        vol_pct = 50.0
        if len(self.vol_history) > 10:
            vol_pct = (sum(1 for v in self.vol_history if v < vol) / len(self.vol_history)) * 100.0

        mom15 = self._calc_return(int(15 / self.p.bar_minutes))
        vol_spike = 1.0
        if self.vol_sma20[0] != 0:
            vol_spike = self.data.volume[0] / self.vol_sma20[0]

        # Support/resistance proxy from last 30 bars
        price_position = 0.5
        if len(self.data) >= 30:
            highs = list(self.data.high.get(size=30))
            lows = list(self.data.low.get(size=30))
            recent_high = max(highs) if highs else self.data.high[0]
            recent_low = min(lows) if lows else self.data.low[0]
            if recent_high != recent_low:
                price_position = (self.data.close[0] - recent_low) / (recent_high - recent_low)

        near_resistance = price_position > 0.8
        near_support = price_position < 0.2

        signal = {
            "action": "HOLD",
            "confidence": 0.5,
            "reasoning": [],
        }
        components = []

        # Volatility regime signals
        if vol_pct < 30.0 and abs(mom15) < 0.005:
            signal["action"] = "BUY" if mom15 >= 0 else "SELL"
            signal["reasoning"].append("Low vol regime")
            components.append(0.6)
        elif vol_pct > 70.0:
            signal["action"] = "SELL"
            signal["reasoning"].append("High vol regime")
            components.append(0.5)

        # Momentum
        if abs(mom15) > 0.005:
            if mom15 > 0 and not near_resistance:
                signal["action"] = "BUY"
                signal["reasoning"].append("Bullish momentum")
                components.append(0.7)
            elif mom15 < 0 and not near_support:
                signal["action"] = "SELL"
                signal["reasoning"].append("Bearish momentum")
                components.append(0.7)

        # Neural style signal (from multi-timeframe consensus)
        mtf = self._combine_multi_timeframe(predictions)
        if mtf["signal_strength"] > 0.15 and mtf["multi_timeframe_confidence"] > self.p.min_confidence_threshold:
            if signal["action"] == "HOLD":
                signal["action"] = "BUY" if mtf["direction"] == "BULLISH" else "SELL"
            components.append(mtf["signal_strength"])
            signal["reasoning"].append("Multi-timeframe consensus")

        # Volume confirmation
        if vol_spike > 1.5:
            components.append(0.1)
            signal["reasoning"].append("Volume confirmation")

        if components:
            raw_conf = min(max(sum(components) / len(components), 0.0), 0.95)
            consistency = self._prediction_consistency()
            consistency_factor = 1.0
            if consistency["samples"] >= 3:
                current_dir = "BULLISH" if signal["action"] == "BUY" else "BEARISH" if signal["action"] == "SELL" else "NEUTRAL"
                if current_dir != "NEUTRAL" and current_dir == consistency["consensus"]:
                    consistency_factor = 1.1
                elif current_dir != "NEUTRAL" and consistency["consensus"] != "NEUTRAL":
                    consistency_factor = 0.85
            signal["confidence"] = max(0.0, min(raw_conf * consistency_factor, 0.95))
        return signal

    def _rl_policy_allows(self, signal: dict) -> bool:
        if signal["confidence"] < self.p.min_confidence_threshold:
            return False
        if self.last_trade_dt:
            delta = self.data.datetime.datetime(0) - self.last_trade_dt
            if delta.total_seconds() < self.p.min_time_between_trades_seconds:
                return False
        if self.recent_rewards:
            avg_reward = sum(self.recent_rewards) / len(self.recent_rewards)
            if avg_reward < -0.002:
                return False
        return True

    def _risk_limits_ok(self) -> bool:
        value = self.broker.getvalue()
        if self.peak_value is None or value > self.peak_value:
            self.peak_value = value
        drawdown = (self.peak_value - value) / self.peak_value if self.peak_value else 0.0
        if drawdown >= self.p.max_drawdown_pct:
            return False

        dt = self.data.datetime.date(0)
        if self.daily_date != dt:
            self.daily_date = dt
            self.daily_start_value = value
        if self.daily_start_value and (self.daily_start_value - value) / self.daily_start_value >= self.p.max_daily_loss_pct:
            return False
        return True

    def next(self) -> None:
        if len(self.data) < 60:
            return

        self._update_prediction_accuracy()

        predictions = {}
        for tf_name, cfg in self.prediction_timeframes.items():
            horizon_bars = max(int(cfg["minutes"] / self.p.bar_minutes), 1)
            pred = self._predict_for_horizon(horizon_bars)
            predictions[tf_name] = pred
            self.pending_preds.append({
                "tf": tf_name,
                "target_bar": (len(self.data) - 1) + horizon_bars,
                "entry_price": self.data.close[0],
                "predicted_return": pred["predicted_return"],
            })

        signal = self._generate_signal(predictions)

        if self.position:
            self._manage_open_position()
            return

        if not self._risk_limits_ok():
            return

        if signal["action"] in ("BUY", "SELL") and self._rl_policy_allows(signal):
            size = self.p.lots_per_trade
            if self._calc_std_return(30) > 0.02:
                size *= 0.5
            if signal["action"] == "BUY":
                self.order = self.buy(size=size)
            else:
                self.order = self.sell(size=size)
            self.last_trade_dt = self.data.datetime.datetime(0)
            self.log(f"Signal {signal['action']} | conf={signal['confidence']:.2f} | size={size}")

    def _manage_open_position(self) -> None:
        if self.entry_price is None:
            return
        price = self.data.close[0]
        hold_bars = (len(self.data) - 1) - self.entry_bar if self.entry_bar is not None else 0
        max_hold_bars = int((self.p.max_hold_time_hours * 60) / max(self.p.bar_minutes, 1))

        if self.position.size > 0:
            if price <= self.stop_price or price >= self.target_price:
                self.close()
                return
            if price >= self.entry_price * (1 + self.p.trailing_profit_pct):
                trail_stop = price * (1 - (self.p.stop_loss_pct / 2.0))
                if self.stop_price is None or trail_stop > self.stop_price:
                    self.stop_price = trail_stop
        else:
            if price >= self.stop_price or price <= self.target_price:
                self.close()
                return
            if price <= self.entry_price * (1 - self.p.trailing_profit_pct):
                trail_stop = price * (1 + (self.p.stop_loss_pct / 2.0))
                if self.stop_price is None or trail_stop < self.stop_price:
                    self.stop_price = trail_stop

        if hold_bars >= max_hold_bars:
            self.close()

    def notify_order(self, order) -> None:
        if order.status in [order.Submitted, order.Accepted]:
            return
        if order.status == order.Completed:
            if self.position.size != 0:
                self.entry_price = order.executed.price
                self.entry_bar = len(self.data) - 1
                if self.position.size > 0:
                    self.stop_price = self.entry_price * (1 - self.p.stop_loss_pct)
                    self.target_price = self.entry_price * (1 + self.p.take_profit_pct)
                else:
                    self.stop_price = self.entry_price * (1 + self.p.stop_loss_pct)
                    self.target_price = self.entry_price * (1 - self.p.take_profit_pct)
        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.order = None

    def notify_trade(self, trade) -> None:
        if trade.isclosed:
            pnl = trade.pnlcomm if hasattr(trade, "pnlcomm") else trade.pnl
            self.recent_rewards.append(pnl)
            self.entry_price = None
            self.entry_bar = None
            self.stop_price = None
            self.target_price = None
            self.log(f"Trade closed | pnl={pnl:.2f}")