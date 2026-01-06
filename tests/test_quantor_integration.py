"""
Tests for Quantor-MTFuzz Integration
====================================

Validates fuzzy position sizing, regime detection, and volatility analytics.
"""

import pytest
import numpy as np
import sys
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from integrations.quantor import (
    QuantorFacade,
    FuzzyPositionSizer,
    RegimeFilter,
    MarketRegime,
    VolatilityAnalyzer
)


class TestFuzzyPositionSizer:
    """Tests for FuzzyPositionSizer."""

    def setup_method(self):
        self.sizer = FuzzyPositionSizer(
            risk_fraction=0.02,
            min_contracts=1,
            max_contracts=10,
            mode='directional'
        )

    def test_rsi_membership_call_ideal(self):
        """RSI 50-70 should be ideal for CALL."""
        membership = self.sizer.calculate_rsi_membership(60.0, 'CALL')
        assert membership == 1.0

    def test_rsi_membership_call_overbought(self):
        """RSI > 70 should penalize CALL entries."""
        membership = self.sizer.calculate_rsi_membership(85.0, 'CALL')
        assert membership <= 0.5  # Overbought penalty

    def test_rsi_membership_put_ideal(self):
        """RSI 30-50 should be ideal for PUT."""
        membership = self.sizer.calculate_rsi_membership(40.0, 'PUT')
        assert membership == 1.0

    def test_adx_membership_trending(self):
        """High ADX should be favorable for directional trading."""
        membership = self.sizer.calculate_adx_membership(45.0)
        assert membership == 1.0

    def test_adx_membership_weak_trend(self):
        """Low ADX should be less favorable for directional trading."""
        membership = self.sizer.calculate_adx_membership(15.0)
        assert membership < 0.8  # Weaker trend = lower but not zero

    def test_bbands_membership_call_near_lower(self):
        """Price near lower band should be good for CALL (bounce)."""
        membership = self.sizer.calculate_bbands_membership(0.2, direction='CALL')
        assert membership >= 0.8

    def test_bbands_membership_put_near_upper(self):
        """Price near upper band should be good for PUT (rejection)."""
        membership = self.sizer.calculate_bbands_membership(0.8, direction='PUT')
        assert membership >= 0.8

    def test_position_size_basic(self):
        """Test basic position sizing calculation."""
        memberships = self.sizer.compute_memberships(
            rsi=55.0,
            adx=35.0,
            bb_position=0.5,
            stoch_k=50.0,
            vix=18.0,
            direction='CALL'
        )

        # With equity=50000 and max_loss=200:
        # max_risk = 0.02 * 50000 = 1000
        # q0 = 1000 // 200 = 5 contracts
        size, debug = self.sizer.compute_position_size(
            equity=50000,
            max_loss_per_contract=200,
            memberships=memberships,
            realized_vol=0.15,
            debug=True
        )

        assert size >= self.sizer.min_contracts
        assert size <= self.sizer.max_contracts
        assert 'confidence' in debug
        assert 'scaling' in debug

    def test_position_size_zero_equity(self):
        """Zero equity should return zero size."""
        memberships = self.sizer.compute_memberships()
        size = self.sizer.compute_position_size(
            equity=0,
            max_loss_per_contract=500,
            memberships=memberships
        )
        assert size == 0


class TestRegimeFilter:
    """Tests for RegimeFilter."""

    def setup_method(self):
        self.filter = RegimeFilter(
            vix_crash_threshold=35.0,
            vix_high_vol_threshold=20.0,
            adx_trend_threshold=25.0
        )

    def test_crash_mode(self):
        """VIX > 35 should trigger CRASH_MODE."""
        regime = self.filter.classify_regime(vix=40.0)
        assert regime == MarketRegime.CRASH_MODE

    def test_bull_trend(self):
        """High ADX + price > SMA should be BULL_TREND."""
        regime = self.filter.classify_regime(
            vix=18.0,
            adx=30.0,
            close=450.0,
            sma_200=440.0
        )
        assert regime == MarketRegime.BULL_TREND

    def test_bear_trend(self):
        """High ADX + price < SMA should be BEAR_TREND."""
        regime = self.filter.classify_regime(
            vix=18.0,
            adx=30.0,
            close=430.0,
            sma_200=440.0
        )
        assert regime == MarketRegime.BEAR_TREND

    def test_high_vol_range(self):
        """Low ADX + high VIX should be HIGH_VOL_RANGE."""
        regime = self.filter.classify_regime(
            vix=25.0,
            adx=15.0
        )
        assert regime == MarketRegime.HIGH_VOL_RANGE

    def test_low_vol_range(self):
        """Low ADX + low VIX should be LOW_VOL_RANGE."""
        regime = self.filter.classify_regime(
            vix=15.0,
            adx=15.0
        )
        assert regime == MarketRegime.LOW_VOL_RANGE

    def test_trading_blocked_crash_mode(self):
        """Trading should be blocked in CRASH_MODE."""
        allowed, reason = self.filter.is_trading_allowed(MarketRegime.CRASH_MODE)
        assert allowed is False
        assert "CRASH" in reason

    def test_trading_blocked_direction_conflict(self):
        """CALL should be blocked in BEAR_TREND."""
        allowed, reason = self.filter.is_trading_allowed(
            MarketRegime.BEAR_TREND,
            direction='CALL'
        )
        assert allowed is False
        assert "CALL blocked" in reason

    def test_direction_bias(self):
        """Bull trend should have positive bias."""
        bias = self.filter.get_direction_bias(MarketRegime.BULL_TREND)
        assert bias > 0

        bias = self.filter.get_direction_bias(MarketRegime.BEAR_TREND)
        assert bias < 0


class TestVolatilityAnalyzer:
    """Tests for VolatilityAnalyzer."""

    def setup_method(self):
        self.analyzer = VolatilityAnalyzer(
            low_vol_threshold=0.10,
            high_vol_threshold=0.25,
            extreme_vol_threshold=0.40
        )
        # Generate sample price data (random walk)
        np.random.seed(42)
        returns = np.random.normal(0.0001, 0.01, 100)
        self.prices = 100 * np.exp(np.cumsum(returns))

    def test_realized_vol_calculation(self):
        """Realized vol should be in reasonable range."""
        rv = self.analyzer.compute_realized_vol(self.prices, window=20)
        assert 0 < rv < 1.0  # Should be reasonable

    def test_realized_vol_insufficient_data(self):
        """Insufficient data should return 0."""
        rv = self.analyzer.compute_realized_vol(np.array([100, 101]), window=20)
        assert rv == 0.0

    def test_skew_calculation(self):
        """Skew should be computed correctly."""
        skew = self.analyzer.compute_skew(
            iv_put=0.20,
            iv_call=0.15,
            iv_atm=0.17
        )
        expected = (0.20 - 0.15) / 0.17
        assert abs(skew - expected) < 0.001

    def test_skew_zero_atm(self):
        """Zero ATM IV should return 0 skew."""
        skew = self.analyzer.compute_skew(0.20, 0.15, 0.0)
        assert skew == 0.0

    def test_steep_skew_detection(self):
        """Steep skew should be detected."""
        assert self.analyzer.is_steep_skew(0.20) is True
        assert self.analyzer.is_steep_skew(0.10) is False

    def test_vrp_calculation(self):
        """VRP should be IV - RV."""
        vrp = self.analyzer.compute_vrp(0.20, 0.15)
        assert abs(vrp - 0.05) < 0.001

    def test_vol_regime_classification(self):
        """Vol regime should be classified correctly."""
        assert self.analyzer.classify_vol_regime(0.05) == "low"
        assert self.analyzer.classify_vol_regime(0.15) == "normal"
        assert self.analyzer.classify_vol_regime(0.30) == "high"
        assert self.analyzer.classify_vol_regime(0.50) == "extreme"

    def test_atr_stop_multiplier(self):
        """ATR stop multiplier should scale with volatility."""
        low_mult = self.analyzer.atr_stop_multiplier(0.003)
        high_mult = self.analyzer.atr_stop_multiplier(0.025)
        assert low_mult < high_mult


class TestQuantorFacade:
    """Tests for the unified QuantorFacade."""

    def setup_method(self):
        self.quantor = QuantorFacade(
            risk_fraction=0.02,
            min_contracts=1,
            max_contracts=10,
            vix_crash_threshold=35.0
        )
        # Generate sample price data
        np.random.seed(42)
        returns = np.random.normal(0.0001, 0.01, 100)
        self.prices = 100 * np.exp(np.cumsum(returns))

    def test_full_analysis(self):
        """Test complete analysis pipeline."""
        result = self.quantor.analyze(
            equity=10000,
            max_loss_per_contract=500,
            direction='CALL',
            close_prices=self.prices,
            rsi=55.0,
            adx=30.0,
            bb_position=0.4,
            stoch_k=45.0,
            vix=18.0,
            ivr=40.0
        )

        assert result.position_size >= 0
        assert result.regime is not None
        assert result.trading_allowed in [True, False]
        assert 0 <= result.fuzzy_confidence <= 1
        assert 0 <= result.overall_confidence <= 1

    def test_should_trade_crash_mode(self):
        """Trading should be blocked in crash mode."""
        should_trade, reason = self.quantor.should_trade(vix=45.0)
        assert should_trade is False
        assert "CRASH" in reason

    def test_should_trade_normal(self):
        """Trading should be allowed in normal conditions."""
        should_trade, reason = self.quantor.should_trade(vix=18.0, adx=15.0)
        assert should_trade is True

    def test_direction_bias(self):
        """Direction bias should reflect regime."""
        bias = self.quantor.get_direction_bias(vix=18.0, adx=30.0, close=450.0, sma_200=440.0)
        assert bias > 0  # Bull trend = positive bias

    def test_simple_position_size(self):
        """Test simplified position sizing interface."""
        market_data = {
            'rsi': 55.0,
            'adx': 30.0,
            'bb_position': 0.5,
            'vix': 18.0,
            'direction': 'CALL'
        }

        # Use equity and max_loss that allow at least 1 contract
        # max_risk = 0.02 * 50000 = 1000, q0 = 1000 // 200 = 5
        size, debug = self.quantor.compute_position_size(
            equity=50000,
            max_loss=200,
            market_data=market_data
        )

        assert size >= 1
        assert 'confidence' in debug


class TestDataAlignment:
    """Tests for data alignment system."""

    def setup_method(self):
        import pandas as pd
        from datetime import datetime, timedelta
        from integrations.quantor import (
            ChainAlignment, AlignmentMode, AlignmentConfig,
            DataAligner, AlignmentDiagnosticsTracker, AlignedStepState
        )
        self.pd = pd
        self.datetime = datetime
        self.timedelta = timedelta
        self.ChainAlignment = ChainAlignment
        self.AlignmentMode = AlignmentMode
        self.AlignmentConfig = AlignmentConfig
        self.DataAligner = DataAligner
        self.AlignmentDiagnosticsTracker = AlignmentDiagnosticsTracker
        self.AlignedStepState = AlignedStepState

        self.config = AlignmentConfig(
            max_lag_sec=600.0,
            iv_decay_half_life_sec=300.0
        )
        self.aligner = DataAligner(self.config)

    def test_iv_confidence_exact(self):
        """Zero lag should give 1.0 confidence."""
        conf = self.aligner.compute_iv_confidence(0)
        assert conf == 1.0

    def test_iv_confidence_half_life(self):
        """At half-life, confidence should be 0.5."""
        conf = self.aligner.compute_iv_confidence(300.0)  # 300s = half-life
        assert abs(conf - 0.5) < 0.01

    def test_iv_confidence_two_half_lives(self):
        """At 2x half-life, confidence should be 0.25."""
        conf = self.aligner.compute_iv_confidence(600.0)
        assert abs(conf - 0.25) < 0.01

    def test_chain_alignment_exact(self):
        """Exact timestamp match should give EXACT mode."""
        now = self.datetime.now()
        chain_df = self.pd.DataFrame({'strike': [100, 105, 110]})

        alignment = self.aligner.align_chain(now, chain_df, now)

        assert alignment.mode == self.AlignmentMode.EXACT
        assert alignment.iv_conf == 1.0
        assert alignment.lag_sec == 0.0
        assert alignment.is_usable

    def test_chain_alignment_prior(self):
        """Small lag should give PRIOR mode with decayed confidence."""
        now = self.datetime.now()
        chain_ts = now - self.timedelta(seconds=60)
        chain_df = self.pd.DataFrame({'strike': [100, 105, 110]})

        alignment = self.aligner.align_chain(now, chain_df, chain_ts)

        assert alignment.mode == self.AlignmentMode.PRIOR
        assert alignment.iv_conf < 1.0
        assert alignment.iv_conf > 0.5  # 60s is less than half-life
        assert abs(alignment.lag_sec - 60.0) < 1.0
        assert alignment.is_usable

    def test_chain_alignment_none_when_no_data(self):
        """No data should give NONE mode."""
        now = self.datetime.now()

        alignment = self.aligner.align_chain(now, None, None)

        assert alignment.mode == self.AlignmentMode.NONE
        assert alignment.iv_conf == 0.0
        assert not alignment.is_usable

    def test_chain_alignment_cutoff(self):
        """Lag beyond max should give NONE mode with DECAY_THEN_CUTOFF policy."""
        now = self.datetime.now()
        chain_ts = now - self.timedelta(seconds=700)  # Beyond 600s max
        chain_df = self.pd.DataFrame({'strike': [100, 105, 110]})

        alignment = self.aligner.align_chain(now, chain_df, chain_ts)

        assert alignment.mode == self.AlignmentMode.NONE
        assert alignment.iv_conf == 0.0

    def test_diagnostics_tracker_counts(self):
        """Diagnostics tracker should count modes correctly."""
        tracker = self.AlignmentDiagnosticsTracker(self.config)
        now = self.datetime.now()

        # Add some alignments
        exact = self.ChainAlignment(
            chain=self.pd.DataFrame(),
            used_ts=now,
            mode=self.AlignmentMode.EXACT,
            lag_sec=0,
            iv_conf=1.0
        )
        prior = self.ChainAlignment(
            chain=self.pd.DataFrame(),
            used_ts=now - self.timedelta(seconds=60),
            mode=self.AlignmentMode.PRIOR,
            lag_sec=60,
            iv_conf=0.87
        )
        none = self.ChainAlignment(
            chain=None,
            used_ts=None,
            mode=self.AlignmentMode.NONE,
            lag_sec=float('inf'),
            iv_conf=0.0
        )

        tracker.record(exact)
        tracker.record(exact)
        tracker.record(prior)
        tracker.record(none)

        stats = tracker.get_stats()
        assert stats.total_bars == 4
        assert stats.exact_count == 2
        assert stats.prior_count == 1
        assert stats.none_count == 1
        assert stats.exact_pct == 50.0

    def test_fail_fast_triggers(self):
        """Fail-fast should trigger when stale rate exceeds threshold."""
        config = self.AlignmentConfig(
            fail_fast_enabled=True,
            fail_fast_stale_threshold=0.3,
            fail_fast_min_bars=10
        )
        tracker = self.AlignmentDiagnosticsTracker(config)

        # Add mostly stale data
        stale = self.ChainAlignment(
            chain=None,
            used_ts=None,
            mode=self.AlignmentMode.STALE,
            lag_sec=500,
            iv_conf=0.1
        )
        exact = self.ChainAlignment(
            chain=self.pd.DataFrame(),
            used_ts=self.datetime.now(),
            mode=self.AlignmentMode.EXACT,
            lag_sec=0,
            iv_conf=1.0
        )

        # Add 5 stale, 5 exact (50% stale > 30% threshold)
        for _ in range(5):
            tracker.record(stale)
        for _ in range(5):
            tracker.record(exact)

        assert tracker.should_fail_fast
        assert "stale" in tracker.fail_fast_reason.lower()

    def test_aligned_step_state_features(self):
        """AlignedStepState should provide correct features."""
        alignment = self.ChainAlignment(
            chain=self.pd.DataFrame(),
            used_ts=self.datetime.now(),
            mode=self.AlignmentMode.PRIOR,
            lag_sec=120,
            iv_conf=0.75
        )

        step_state = self.AlignedStepState(chain_alignment=alignment)

        features = step_state.get_features()
        assert features['iv_conf'] == 0.75
        assert 0 < features['data_lag_normalized'] < 1  # 120/600 = 0.2
        assert features['alignment_exact'] == 0.0
        assert features['alignment_usable'] == 1.0

    def test_confidence_multiplier(self):
        """iv_conf should be used as confidence multiplier."""
        alignment = self.ChainAlignment(
            chain=self.pd.DataFrame(),
            used_ts=self.datetime.now(),
            mode=self.AlignmentMode.PRIOR,
            lag_sec=300,  # Half-life
            iv_conf=0.5
        )

        step_state = self.AlignedStepState(chain_alignment=alignment)

        # Base confidence of 0.8 should become 0.4 after iv_conf multiplier
        adjusted = step_state.apply_to_confidence(0.8)
        assert abs(adjusted - 0.4) < 0.01


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
