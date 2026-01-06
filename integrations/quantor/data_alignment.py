"""
Data Alignment System for Gaussian System
==========================================

Adapted from Quantor-MTFuzz option_chain.py and data_engine.py
Provides:
- ChainAlignment interface for options data with lag/confidence metadata
- Alignment diagnostics tracker for run quality assessment
- Fail-fast mechanism for backtests with misaligned data

Key Concepts:
- iv_conf: Confidence score that decays with data staleness (half-life decay)
- Alignment modes: exact, prior, stale, none
- Fail-fast: Stop backtests when data is too misaligned to be useful
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum
import numpy as np
import pandas as pd
import logging

logger = logging.getLogger(__name__)


class AlignmentMode(Enum):
    """Data alignment mode classification."""
    EXACT = "exact"      # Timestamp matches exactly
    PRIOR = "prior"      # Using most recent prior snapshot (acceptable lag)
    STALE = "stale"      # Data is old but usable with heavy penalty
    NONE = "none"        # No data available


@dataclass(frozen=True)
class ChainAlignment:
    """
    Alignment metadata for options chain data.

    This is the HARD INTERFACE that all options data fetchers must return.
    Provides transparency about data freshness and confidence.
    """
    chain: Optional[pd.DataFrame]  # The actual option chain (or None if unavailable)
    used_ts: Optional[datetime]    # Which timestamp was actually used
    mode: AlignmentMode            # How the data was aligned
    lag_sec: float                 # Time lag in seconds from requested timestamp
    iv_conf: float                 # [0,1] confidence score (decays with lag)

    def __post_init__(self):
        # Validate iv_conf is in range
        if not 0 <= self.iv_conf <= 1:
            object.__setattr__(self, 'iv_conf', max(0, min(1, self.iv_conf)))

    @property
    def is_usable(self) -> bool:
        """Check if this chain is usable for trading decisions."""
        return self.mode != AlignmentMode.NONE and self.chain is not None

    @property
    def lag_minutes(self) -> float:
        """Lag in minutes for easier reading."""
        return self.lag_sec / 60.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging/serialization."""
        return {
            'mode': self.mode.value,
            'lag_sec': self.lag_sec,
            'iv_conf': self.iv_conf,
            'used_ts': self.used_ts.isoformat() if self.used_ts else None,
            'chain_size': len(self.chain) if self.chain is not None else 0
        }


class AlignmentPolicy(Enum):
    """Policy for handling data lag."""
    HARD_CUTOFF = "hard_cutoff"           # Empty chain if lag > max_lag
    DECAY_ONLY = "decay_only"             # Use prior, apply decay to iv_conf
    DECAY_THEN_CUTOFF = "decay_then_cutoff"  # Decay AND empty if exceeds cutoff


@dataclass
class AlignmentConfig:
    """Configuration for data alignment behavior."""
    max_lag_sec: float = 600.0           # Maximum acceptable lag (10 minutes)
    iv_decay_half_life_sec: float = 300.0  # Seconds for iv_conf to halve (5 minutes)
    policy: AlignmentPolicy = AlignmentPolicy.DECAY_THEN_CUTOFF

    # Fail-fast settings
    fail_fast_enabled: bool = True
    fail_fast_stale_threshold: float = 0.3   # Fail if >30% of bars are stale
    fail_fast_none_threshold: float = 0.1    # Fail if >10% of bars have no data
    fail_fast_min_bars: int = 50             # Minimum bars before checking

    # iv_conf thresholds for trading
    min_iv_conf_for_entry: float = 0.3       # Don't enter trades below this
    min_iv_conf_for_sizing: float = 0.5      # Full position sizing above this


class DataAligner:
    """
    Handles data alignment with lag tracking and confidence decay.

    Use this to wrap your existing options data fetcher to add
    alignment metadata to every response.
    """

    def __init__(self, config: AlignmentConfig = None):
        self.config = config or AlignmentConfig()
        self._options_cache: Dict[datetime, pd.DataFrame] = {}
        self._cache_timestamps: List[datetime] = []

    def compute_iv_confidence(self, lag_sec: float) -> float:
        """
        Compute IV confidence based on data lag.

        Uses exponential decay: iv_conf = 0.5^(lag / half_life)

        Args:
            lag_sec: Lag in seconds

        Returns:
            Confidence score in [0, 1]
        """
        if lag_sec <= 0:
            return 1.0

        half_life = self.config.iv_decay_half_life_sec
        iv_conf = 0.5 ** (lag_sec / half_life)
        return max(0.0, min(1.0, iv_conf))

    def align_chain(
        self,
        requested_ts: datetime,
        chain_data: Optional[pd.DataFrame],
        chain_ts: Optional[datetime]
    ) -> ChainAlignment:
        """
        Create ChainAlignment from raw chain data.

        Args:
            requested_ts: The timestamp we want data for
            chain_data: The options chain data (may be None)
            chain_ts: The actual timestamp of the chain data

        Returns:
            ChainAlignment with appropriate mode and confidence
        """
        # No data at all
        if chain_data is None or chain_ts is None:
            return ChainAlignment(
                chain=None,
                used_ts=None,
                mode=AlignmentMode.NONE,
                lag_sec=float('inf'),
                iv_conf=0.0
            )

        # Calculate lag
        if isinstance(requested_ts, str):
            requested_ts = pd.to_datetime(requested_ts)
        if isinstance(chain_ts, str):
            chain_ts = pd.to_datetime(chain_ts)

        lag_sec = (requested_ts - chain_ts).total_seconds()

        # Exact match
        if abs(lag_sec) < 1.0:
            return ChainAlignment(
                chain=chain_data,
                used_ts=chain_ts,
                mode=AlignmentMode.EXACT,
                lag_sec=0.0,
                iv_conf=1.0
            )

        # Compute confidence decay
        iv_conf = self.compute_iv_confidence(abs(lag_sec))

        # Apply policy
        policy = self.config.policy
        max_lag = self.config.max_lag_sec

        if policy == AlignmentPolicy.HARD_CUTOFF:
            if abs(lag_sec) > max_lag:
                return ChainAlignment(
                    chain=None,
                    used_ts=chain_ts,
                    mode=AlignmentMode.NONE,
                    lag_sec=lag_sec,
                    iv_conf=0.0
                )
            mode = AlignmentMode.PRIOR

        elif policy == AlignmentPolicy.DECAY_ONLY:
            mode = AlignmentMode.STALE if abs(lag_sec) > max_lag else AlignmentMode.PRIOR

        else:  # DECAY_THEN_CUTOFF
            if abs(lag_sec) > max_lag:
                return ChainAlignment(
                    chain=None,
                    used_ts=chain_ts,
                    mode=AlignmentMode.NONE,
                    lag_sec=lag_sec,
                    iv_conf=0.0
                )
            mode = AlignmentMode.PRIOR

        return ChainAlignment(
            chain=chain_data,
            used_ts=chain_ts,
            mode=mode,
            lag_sec=lag_sec,
            iv_conf=iv_conf
        )

    def cache_chain(self, timestamp: datetime, chain: pd.DataFrame):
        """Cache an options chain for later lookup."""
        self._options_cache[timestamp] = chain
        self._cache_timestamps.append(timestamp)
        self._cache_timestamps.sort()

    def get_aligned_chain(self, requested_ts: datetime) -> ChainAlignment:
        """
        Get the best aligned chain from cache.

        Args:
            requested_ts: The timestamp we want data for

        Returns:
            ChainAlignment with the best available data
        """
        if not self._cache_timestamps:
            return ChainAlignment(
                chain=None,
                used_ts=None,
                mode=AlignmentMode.NONE,
                lag_sec=float('inf'),
                iv_conf=0.0
            )

        # Find the most recent timestamp <= requested_ts
        best_ts = None
        for ts in reversed(self._cache_timestamps):
            if ts <= requested_ts:
                best_ts = ts
                break

        if best_ts is None:
            return ChainAlignment(
                chain=None,
                used_ts=None,
                mode=AlignmentMode.NONE,
                lag_sec=float('inf'),
                iv_conf=0.0
            )

        chain_data = self._options_cache.get(best_ts)
        return self.align_chain(requested_ts, chain_data, best_ts)


@dataclass
class AlignmentStats:
    """Statistics for alignment diagnostics."""
    total_bars: int = 0
    exact_count: int = 0
    prior_count: int = 0
    stale_count: int = 0
    none_count: int = 0

    lag_values: List[float] = field(default_factory=list)
    iv_conf_values: List[float] = field(default_factory=list)

    @property
    def exact_pct(self) -> float:
        return 100 * self.exact_count / self.total_bars if self.total_bars > 0 else 0

    @property
    def prior_pct(self) -> float:
        return 100 * self.prior_count / self.total_bars if self.total_bars > 0 else 0

    @property
    def stale_pct(self) -> float:
        return 100 * self.stale_count / self.total_bars if self.total_bars > 0 else 0

    @property
    def none_pct(self) -> float:
        return 100 * self.none_count / self.total_bars if self.total_bars > 0 else 0

    @property
    def lag_median(self) -> float:
        return float(np.median(self.lag_values)) if self.lag_values else 0

    @property
    def lag_p90(self) -> float:
        return float(np.percentile(self.lag_values, 90)) if self.lag_values else 0

    @property
    def lag_max(self) -> float:
        return max(self.lag_values) if self.lag_values else 0

    @property
    def iv_conf_min(self) -> float:
        return min(self.iv_conf_values) if self.iv_conf_values else 0

    @property
    def iv_conf_p10(self) -> float:
        return float(np.percentile(self.iv_conf_values, 10)) if self.iv_conf_values else 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging."""
        return {
            'total_bars': self.total_bars,
            'exact_pct': f"{self.exact_pct:.1f}%",
            'prior_pct': f"{self.prior_pct:.1f}%",
            'stale_pct': f"{self.stale_pct:.1f}%",
            'none_pct': f"{self.none_pct:.1f}%",
            'lag_median_sec': f"{self.lag_median:.1f}",
            'lag_p90_sec': f"{self.lag_p90:.1f}",
            'lag_max_sec': f"{self.lag_max:.1f}",
            'iv_conf_min': f"{self.iv_conf_min:.3f}",
            'iv_conf_p10': f"{self.iv_conf_p10:.3f}"
        }

    def summary(self) -> str:
        """Human-readable summary."""
        return (
            f"Alignment Stats ({self.total_bars} bars):\n"
            f"  Modes: exact={self.exact_pct:.1f}%, prior={self.prior_pct:.1f}%, "
            f"stale={self.stale_pct:.1f}%, none={self.none_pct:.1f}%\n"
            f"  Lag: median={self.lag_median:.1f}s, p90={self.lag_p90:.1f}s, max={self.lag_max:.1f}s\n"
            f"  IV Conf: min={self.iv_conf_min:.3f}, p10={self.iv_conf_p10:.3f}"
        )


class AlignmentDiagnosticsTracker:
    """
    Tracks alignment diagnostics over a simulation run.

    Use this to monitor data quality and detect misalignment issues.
    """

    def __init__(self, config: AlignmentConfig = None):
        self.config = config or AlignmentConfig()
        self.stats = AlignmentStats()
        self._fail_fast_triggered = False
        self._fail_fast_reason = ""

    def record(self, alignment: ChainAlignment):
        """Record an alignment result."""
        self.stats.total_bars += 1

        if alignment.mode == AlignmentMode.EXACT:
            self.stats.exact_count += 1
        elif alignment.mode == AlignmentMode.PRIOR:
            self.stats.prior_count += 1
        elif alignment.mode == AlignmentMode.STALE:
            self.stats.stale_count += 1
        else:  # NONE
            self.stats.none_count += 1

        if alignment.lag_sec < float('inf'):
            self.stats.lag_values.append(alignment.lag_sec)

        self.stats.iv_conf_values.append(alignment.iv_conf)

        # Check fail-fast conditions
        if self.config.fail_fast_enabled:
            self._check_fail_fast()

    def _check_fail_fast(self):
        """Check if we should trigger fail-fast."""
        if self._fail_fast_triggered:
            return

        if self.stats.total_bars < self.config.fail_fast_min_bars:
            return

        # Check stale threshold
        stale_rate = (self.stats.stale_count + self.stats.none_count) / self.stats.total_bars
        if stale_rate > self.config.fail_fast_stale_threshold:
            self._fail_fast_triggered = True
            self._fail_fast_reason = (
                f"Stale data rate {stale_rate:.1%} exceeds threshold "
                f"{self.config.fail_fast_stale_threshold:.1%}"
            )
            return

        # Check none threshold
        none_rate = self.stats.none_count / self.stats.total_bars
        if none_rate > self.config.fail_fast_none_threshold:
            self._fail_fast_triggered = True
            self._fail_fast_reason = (
                f"Missing data rate {none_rate:.1%} exceeds threshold "
                f"{self.config.fail_fast_none_threshold:.1%}"
            )

    @property
    def should_fail_fast(self) -> bool:
        """Check if fail-fast has been triggered."""
        return self._fail_fast_triggered

    @property
    def fail_fast_reason(self) -> str:
        """Get the reason for fail-fast (if triggered)."""
        return self._fail_fast_reason

    def check_and_raise(self):
        """Check fail-fast and raise RuntimeError if triggered."""
        if self.should_fail_fast:
            raise RuntimeError(
                f"FAIL-FAST: Data alignment failure - {self.fail_fast_reason}\n"
                f"{self.stats.summary()}"
            )

    def get_stats(self) -> AlignmentStats:
        """Get current statistics."""
        return self.stats

    def reset(self):
        """Reset all statistics."""
        self.stats = AlignmentStats()
        self._fail_fast_triggered = False
        self._fail_fast_reason = ""

    def log_summary(self, level: int = logging.INFO):
        """Log a summary of alignment statistics."""
        logger.log(level, self.stats.summary())
        if self._fail_fast_triggered:
            logger.warning(f"FAIL-FAST TRIGGERED: {self._fail_fast_reason}")


@dataclass
class AlignedStepState:
    """
    Per-step state with alignment metadata.

    Use this to enrich your existing step state with alignment info
    for logging, model features, and confidence adjustment.
    """
    # Core alignment fields
    chain_alignment: ChainAlignment

    # Convenience fields (extracted for easy access)
    iv_conf: float = 0.0
    data_lag_sec: float = 0.0
    alignment_mode: str = "none"

    # Feature adjustments
    confidence_multiplier: float = 1.0  # Apply to trade confidence

    def __post_init__(self):
        """Extract convenience fields from alignment."""
        self.iv_conf = self.chain_alignment.iv_conf
        self.data_lag_sec = self.chain_alignment.lag_sec
        self.alignment_mode = self.chain_alignment.mode.value

        # iv_conf directly becomes confidence multiplier
        self.confidence_multiplier = self.iv_conf

    def apply_to_confidence(self, base_confidence: float) -> float:
        """
        Apply alignment penalty to trading confidence.

        Args:
            base_confidence: Original confidence score

        Returns:
            Adjusted confidence (penalized for stale data)
        """
        return base_confidence * self.confidence_multiplier

    def get_features(self) -> Dict[str, float]:
        """
        Get alignment features for model input.

        Returns dict with:
        - iv_conf: The confidence score [0, 1]
        - data_lag_normalized: Lag normalized to [0, 1] (0 = fresh, 1 = 10min+)
        - alignment_exact: 1.0 if exact match, else 0.0
        - alignment_usable: 1.0 if data is usable, else 0.0
        """
        # Normalize lag: 0-600 sec → 0-1
        lag_normalized = min(1.0, self.data_lag_sec / 600.0)

        return {
            'iv_conf': self.iv_conf,
            'data_lag_normalized': lag_normalized,
            'alignment_exact': 1.0 if self.alignment_mode == 'exact' else 0.0,
            'alignment_usable': 1.0 if self.chain_alignment.is_usable else 0.0
        }

    def to_log_dict(self) -> Dict[str, Any]:
        """Get dict for logging."""
        return {
            'iv_conf': self.iv_conf,
            'data_lag_sec': self.data_lag_sec,
            'alignment_mode': self.alignment_mode,
            'confidence_multiplier': self.confidence_multiplier
        }


def create_alignment_wrapper(
    fetch_options_func,
    config: AlignmentConfig = None
):
    """
    Decorator/wrapper to add alignment metadata to any options fetch function.

    Usage:
        @create_alignment_wrapper
        def get_options_chain(symbol, expiration):
            # ... existing fetch logic ...
            return chain_df, chain_timestamp

        # Or wrap existing function:
        aligned_fetch = create_alignment_wrapper(existing_fetch_func)
        alignment = aligned_fetch(requested_ts, symbol, expiration)
    """
    aligner = DataAligner(config or AlignmentConfig())

    def wrapper(requested_ts: datetime, *args, **kwargs) -> ChainAlignment:
        try:
            result = fetch_options_func(*args, **kwargs)

            if result is None:
                return aligner.align_chain(requested_ts, None, None)

            # Handle different return types
            if isinstance(result, tuple) and len(result) == 2:
                chain_data, chain_ts = result
            elif isinstance(result, pd.DataFrame):
                chain_data = result
                chain_ts = datetime.now()  # Assume current if not provided
            else:
                chain_data = result
                chain_ts = datetime.now()

            return aligner.align_chain(requested_ts, chain_data, chain_ts)

        except Exception as e:
            logger.error(f"Error in aligned fetch: {e}")
            return ChainAlignment(
                chain=None,
                used_ts=None,
                mode=AlignmentMode.NONE,
                lag_sec=float('inf'),
                iv_conf=0.0
            )

    return wrapper
