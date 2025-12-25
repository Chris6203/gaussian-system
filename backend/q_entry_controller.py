#!/usr/bin/env python3
"""
Q-style Entry Controller (Offline Q Regression)
==============================================

This is NOT bootstrapped RL.
It loads a supervised regression model that predicts:
    Q_hold, Q_call, Q_put
as expected net reward over a fixed horizon (configured in metadata).

Safety:
- If model/metadata missing or invalid, caller should fall back to edge controller.
- This controller does not place orders; it only chooses an action + provides q_values.

Env overrides:
- Q_SCORER_MODEL_PATH: default "<output3>/models/q_scorer.pt"
- Q_SCORER_METADATA_PATH: default "<output3>/models/q_scorer_metadata.json"
- ENTRY_Q_THRESHOLD: default 0.0 (dollars)
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


def _sf(x: Any, default: float = 0.0) -> float:
    try:
        if x is None:
            return float(default)
        return float(x)
    except Exception:
        return float(default)


def _as_float_list(x: Any) -> Optional[List[float]]:
    if not isinstance(x, list):
        return None
    out: List[float] = []
    try:
        for v in x:
            out.append(float(v))
        return out
    except Exception:
        return None


@dataclass
class QDecision:
    action: str  # HOLD | BUY_CALLS | BUY_PUTS
    q_values: Dict[str, float]  # {"HOLD":..., "BUY_CALLS":..., "BUY_PUTS":...}
    threshold: float
    reason: str
    position_size: int = 0


class QScorerNet(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: Sequence[int] = (128, 64), dropout: float = 0.2):
        super().__init__()
        layers: List[nn.Module] = []
        d = int(input_dim)
        for h in hidden_dims:
            layers.append(nn.Linear(d, int(h)))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(float(dropout)))
            d = int(h)
        layers.append(nn.Linear(d, 3))  # Q_hold, Q_call, Q_put
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class QEntryController:
    def __init__(
        self,
        *,
        model_path: Optional[str] = None,
        metadata_path: Optional[str] = None,
        device: Optional[str] = None,
    ):
        base_dir = Path(__file__).resolve().parents[1]  # <output3>/
        default_model = str(base_dir / "models" / "q_scorer.pt")
        default_meta = str(base_dir / "models" / "q_scorer_metadata.json")

        self.model_path = Path(model_path or os.environ.get("Q_SCORER_MODEL_PATH", default_model))
        self.metadata_path = Path(metadata_path or os.environ.get("Q_SCORER_METADATA_PATH", default_meta))
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.metadata: Dict[str, Any] = {}
        self.model: Optional[QScorerNet] = None
        self._last_mtime_model: Optional[float] = None
        self._last_mtime_meta: Optional[float] = None
        self._last_check_ts: float = 0.0
        try:
            self.reload_interval_sec = float(os.environ.get("Q_SCORER_RELOAD_SEC", "30"))
        except Exception:
            self.reload_interval_sec = 30.0
        self.reload_interval_sec = float(max(1.0, self.reload_interval_sec))

    def is_available(self) -> bool:
        return self.model_path.exists() and self.metadata_path.exists()

    def load(self) -> bool:
        if not self.is_available():
            logger.warning(f"[Q_SCORER] Files not found: model={self.model_path}, meta={self.metadata_path}")
            return False
        try:
            meta = json.loads(self.metadata_path.read_text(encoding="utf-8"))
            input_dim = int(meta.get("input_dim", 0))
            hidden_dims = meta.get("hidden_dims", [128, 64])
            dropout = float(meta.get("dropout", 0.2))
            model = QScorerNet(input_dim=input_dim, hidden_dims=hidden_dims, dropout=dropout).to(self.device)

            state = torch.load(self.model_path, map_location=self.device)
            # allow either full model state dict or wrapped dict
            if isinstance(state, dict) and "state_dict" in state:
                model.load_state_dict(state["state_dict"])
            elif isinstance(state, dict):
                model.load_state_dict(state)
            else:
                logger.error(f"[Q_SCORER] Invalid model state format in {self.model_path}")
                return False

            model.eval()

            # Only swap in after fully loaded (so live bot never loses a working model)
            self.metadata = meta
            self.model = model
            try:
                self._last_mtime_model = float(self.model_path.stat().st_mtime)
                self._last_mtime_meta = float(self.metadata_path.stat().st_mtime)
            except Exception:
                self._last_mtime_model = None
                self._last_mtime_meta = None
            return True
        except Exception as e:
            logger.error(f"[Q_SCORER] Load failed: {e}")
            import traceback
            traceback.print_exc()
            return False

    def maybe_reload(self) -> bool:
        """
        Hot-reload support for live bots.

        Checks mtimes every `Q_SCORER_RELOAD_SEC` and reloads if model/metadata changed.
        This is safe to call every cycle.
        """
        now = time.time()
        if (now - float(self._last_check_ts)) < float(self.reload_interval_sec):
            return False
        self._last_check_ts = now

        try:
            m_model = float(self.model_path.stat().st_mtime)
            m_meta = float(self.metadata_path.stat().st_mtime)
        except Exception:
            return False

        # If not loaded yet, try to load.
        if self.model is None:
            ok = self.load()
            if ok:
                logger.info(f"[Q_SCORER] Loaded model from {self.model_path}")
            return bool(ok)

        # Only reload on change.
        if self._last_mtime_model is not None and self._last_mtime_meta is not None:
            if m_model == self._last_mtime_model and m_meta == self._last_mtime_meta:
                return False

        ok = self.load()
        if ok:
            logger.info(f"[Q_SCORER] Reloaded model from {self.model_path}")
        return bool(ok)

    def build_x(self, trade_state: Mapping[str, Any]) -> Optional[np.ndarray]:
        """
        Build input vector X = [embedding (64) + scalars...], consistent with metadata.
        """
        if not self.metadata:
            return None
        emb_key = str(self.metadata.get("embedding_key", "predictor_embedding"))
        emb = _as_float_list(trade_state.get(emb_key))
        if emb is None:
            # fallback: common alternative key
            emb = _as_float_list(trade_state.get("predictor_embedding"))

        expected_emb_dim = int(self.metadata.get("embedding_dim", 64))

        if emb is None:
            # Use zero embedding as fallback when missing (time-travel mode may not have embeddings)
            logger.debug(f"[Q_SCORER] Missing predictor_embedding, using zero vector of dim {expected_emb_dim}")
            emb = [0.0] * expected_emb_dim
        elif len(emb) != expected_emb_dim:
            logger.warning(f"[Q_SCORER] Embedding dim mismatch: got {len(emb)}, expected {expected_emb_dim}")
            return None

        scalars: List[str] = list(self.metadata.get("scalar_features", []))
        x: List[float] = list(emb)

        # Better defaults for missing fields (match training-time typical ranges).
        default_scalar: Dict[str, float] = {
            "vix_bb_pos": 0.5,
            "hmm_trend": 0.5,
            "hmm_volatility": 0.5,
            "hmm_liquidity": 0.5,
            "hmm_confidence": 0.5,
            "fillability": 0.5,
            "volume_spike": 1.0,
            "liquidity_proxy": 0.5,
            "spread_proxy": 0.5,
        }

        def _get_scalar(k: str) -> float:
            v = trade_state.get(k)
            if v is None:
                # Common fallbacks from live signal dictionaries
                if k == "prediction_confidence":
                    v = trade_state.get("confidence")
                elif k == "predicted_volatility":
                    v = trade_state.get("prediction_uncertainty")
            if isinstance(v, bool):
                return 1.0 if v else 0.0
            if v is None:
                return float(default_scalar.get(k, 0.0))
            return _sf(v, float(default_scalar.get(k, 0.0)))

        for k in scalars:
            x.append(_get_scalar(k))

        return np.asarray(x, dtype=np.float32)

    def decide(
        self,
        trade_state: Mapping[str, Any],
        *,
        q_threshold: Optional[float] = None,
        size_scale: float = 0.05,
    ) -> Optional[QDecision]:
        """
        Returns QDecision or None if model unavailable.

        Rule:
        - choose argmax(Q_call, Q_put) vs Q_hold
        - only trade if max(Q_call, Q_put) > threshold; else HOLD
        """
        if self.model is None:
            return None

        # Optional conservative pre-gates (reduce overtrading in live).
        try:
            min_conf = float(os.environ.get("Q_ENTRY_MIN_CONF", "0") or "0")
        except Exception:
            min_conf = 0.0
        try:
            min_abs_ret = float(os.environ.get("Q_ENTRY_MIN_ABS_RET", "0") or "0")
        except Exception:
            min_abs_ret = 0.0
        if min_conf > 0.0:
            conf = trade_state.get("prediction_confidence")
            if conf is None:
                conf = trade_state.get("confidence")
            if _sf(conf, 0.0) < float(min_conf):
                return QDecision(
                    action="HOLD",
                    q_values={"HOLD": 0.0, "BUY_CALLS": 0.0, "BUY_PUTS": 0.0},
                    threshold=float(q_threshold if q_threshold is not None else _sf(os.environ.get("ENTRY_Q_THRESHOLD", 0.0), 0.0)),
                    reason=f"q_gate: conf<{min_conf:.2f}",
                    position_size=0,
                )
        if min_abs_ret > 0.0:
            pr = trade_state.get("predicted_return")
            if pr is None:
                pr = trade_state.get("multi_timeframe_return")
            if abs(_sf(pr, 0.0)) < float(min_abs_ret):
                return QDecision(
                    action="HOLD",
                    q_values={"HOLD": 0.0, "BUY_CALLS": 0.0, "BUY_PUTS": 0.0},
                    threshold=float(q_threshold if q_threshold is not None else _sf(os.environ.get("ENTRY_Q_THRESHOLD", 0.0), 0.0)),
                    reason=f"q_gate: |ret|<{min_abs_ret:.4f}",
                    position_size=0,
                )

        x = self.build_x(trade_state)
        if x is None:
            return None

        thr = float(q_threshold if q_threshold is not None else _sf(os.environ.get("ENTRY_Q_THRESHOLD", 0.0), 0.0))

        with torch.no_grad():
            xt = torch.from_numpy(x).to(self.device).unsqueeze(0)
            q = self.model(xt).squeeze(0).detach().cpu().numpy().astype(np.float32)

        q_hold, q_call, q_put = float(q[0]), float(q[1]), float(q[2])

        # CRITICAL FIX: Invert Q-values to fix anti-selection bug
        # The model was trained with inverted labels, so we flip them at inference
        # NOTE: bool("0") is True in Python, so we must check for "1"/"true" explicitly
        invert_q = os.environ.get("Q_INVERT_FIX", "1") in ("1", "true", "True")  # Default ON
        if invert_q:
            q_call = -q_call
            q_put = -q_put

        # Optional output calibration (stored in metadata).
        # We currently apply a best-trade offset to CALL/PUT so that threshold 0 is closer to break-even.
        # IMPORTANT: When using Q_INVERT_FIX, calibration may be wrong (computed on non-inverted model)
        disable_cal = os.environ.get("Q_DISABLE_CALIBRATION", "0") in ("1", "true", "True")
        if not disable_cal:
            try:
                cal = self.metadata.get("calibration") or {}
                best_off = _sf(cal.get("best_offset", 0.0), 0.0)
                if abs(float(best_off)) > 1e-9:
                    q_call = float(q_call + float(best_off))
                    q_put = float(q_put + float(best_off))
            except Exception:
                pass
        q_values = {"HOLD": q_hold, "BUY_CALLS": q_call, "BUY_PUTS": q_put}

        best_trade_q = max(q_call, q_put)
        best_trade_action = "BUY_CALLS" if q_call >= q_put else "BUY_PUTS"

        if best_trade_q <= thr:
            return QDecision(
                action="HOLD",
                q_values=q_values,
                threshold=thr,
                reason=f"q_hold: best_trade_q={best_trade_q:+.2f} <= thr={thr:+.2f}",
                position_size=0,
            )

        # suggested size: proportional to positive Q (clamped)
        size = int(max(1, min(5, round(max(1.0, best_trade_q * size_scale)))))
        return QDecision(
            action=best_trade_action,
            q_values=q_values,
            threshold=thr,
            reason=f"q_trade: {best_trade_action} q={best_trade_q:+.2f} thr={thr:+.2f}",
            position_size=size,
        )





