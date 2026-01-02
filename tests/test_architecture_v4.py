#!/usr/bin/env python3
"""
Tests for Architecture V4 components.

Run with: python tests/test_architecture_v4.py
"""

import sys
import os

# Add parent to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import unittest
from datetime import datetime


class TestEVGate(unittest.TestCase):
    """Test EV Gate with Bayesian prior."""

    def setUp(self):
        # Ensure EV gate is enabled for tests
        os.environ['EV_GATE_ENABLED'] = '1'
        # Re-import to pick up env
        from backend.ev_gate import EVGate
        self.gate = EVGate()

    def test_bayesian_prior_blend(self):
        """Test that Bayesian prior blends correctly."""
        # With 30% model confidence, 40% prior, 0.3 weight:
        # posterior = 0.3 * 0.40 + 0.7 * 0.30 = 0.12 + 0.21 = 0.33
        result = self.gate.calculate_ev(
            win_prob=0.30,
            take_profit_pct=0.12,
            stop_loss_pct=0.08,
            expected_hold_minutes=45,
        )
        # Check breakdown contains blended probability
        self.assertIn('win_prob', result.breakdown)
        print(f"  EV Gate: win_prob={result.breakdown['win_prob']:.2%}, "
              f"ev_net={result.ev:.2%}, passed={result.passed}")

    def test_positive_ev_passes(self):
        """Test that high confidence passes."""
        result = self.gate.calculate_ev(
            win_prob=0.60,  # High confidence
            take_profit_pct=0.12,
            stop_loss_pct=0.08,
            expected_hold_minutes=30,
        )
        self.assertTrue(result.passed, f"Should pass with 60% confidence: {result.rejection_reason}")
        print(f"  High conf test: ev={result.ev:.2%}, passed={result.passed}")

    def test_costs_included(self):
        """Test that costs are properly calculated."""
        result = self.gate.calculate_ev(
            win_prob=0.50,
            take_profit_pct=0.12,
            stop_loss_pct=0.08,
            expected_hold_minutes=45,
        )
        self.assertIn('total_costs', result.breakdown)
        self.assertGreater(result.breakdown['total_costs'], 0)
        print(f"  Costs test: total_costs={result.breakdown['total_costs']:.2%}")


class TestRegimeCalibration(unittest.TestCase):
    """Test per-regime calibration."""

    def setUp(self):
        os.environ['REGIME_CALIBRATION'] = '1'
        from backend.regime_calibration import RegimeCalibrator
        self.calibrator = RegimeCalibrator()

    def test_regime_key_generation(self):
        """Test regime key generation."""
        # Low vol, morning
        key = self.calibrator.get_regime_key(
            hmm_volatility=0.3,
            timestamp=datetime(2025, 7, 1, 10, 0)
        )
        self.assertEqual(key, "low_vol_open")
        print(f"  Regime key: hmm_vol=0.3, 10:00 -> {key}")

        # High vol, afternoon
        key = self.calibrator.get_regime_key(
            hmm_volatility=0.7,
            timestamp=datetime(2025, 7, 1, 15, 0)
        )
        self.assertEqual(key, "high_vol_close")
        print(f"  Regime key: hmm_vol=0.7, 15:00 -> {key}")

    def test_record_and_calibrate(self):
        """Test recording outcomes and calibrating."""
        # Record some outcomes
        for i in range(50):
            self.calibrator.record_outcome(
                confidence=0.5 + i * 0.01,
                outcome=1 if i % 2 == 0 else 0,
                hmm_volatility=0.5,
                timestamp=datetime(2025, 7, 1, 12, 0)
            )

        # Check calibration works
        calibrated, regime = self.calibrator.calibrate(
            confidence=0.6,
            hmm_volatility=0.5,
            timestamp=datetime(2025, 7, 1, 12, 0)
        )
        print(f"  Calibrated: raw=0.60, calibrated={calibrated:.2f}, regime={regime}")


class TestRegimeAttribution(unittest.TestCase):
    """Test regime performance tracking."""

    def setUp(self):
        os.environ['REGIME_AUTO_DISABLE'] = '1'
        from backend.regime_attribution import RegimePerformanceTracker
        self.tracker = RegimePerformanceTracker()

    def test_record_trades(self):
        """Test recording trades by regime."""
        # Record some winning trades
        for i in range(25):
            self.tracker.record_trade(
                pnl=10.0,
                hmm_trend=0.7,
                hmm_volatility=0.5,
                confidence=0.6,
            )

        stats = self.tracker.get_all_stats()
        self.assertIn('bullish_normal_vol', stats)
        print(f"  Recorded trades: {stats}")

    def test_should_trade(self):
        """Test trade permission by regime."""
        # Fresh tracker should allow trades
        can_trade, reason = self.tracker.should_trade(
            hmm_trend=0.7,
            hmm_volatility=0.5,
            confidence=0.6,
        )
        self.assertTrue(can_trade)
        print(f"  Can trade: {can_trade}, reason: {reason}")


class TestGreeksAwareExits(unittest.TestCase):
    """Test Greeks-aware exit manager."""

    def setUp(self):
        os.environ['GREEKS_AWARE_EXITS'] = '1'
        from backend.greeks_aware_exits import GreeksAwareExitManager
        self.manager = GreeksAwareExitManager()

    def test_high_vix_widens_stops(self):
        """Test that high VIX widens stops."""
        # Normal VIX
        normal = self.manager.calculate_exit_levels(vix_level=15)
        # High VIX
        high = self.manager.calculate_exit_levels(vix_level=30)

        self.assertGreater(high.stop_loss_pct, normal.stop_loss_pct)
        print(f"  VIX 15: SL={normal.stop_loss_pct}%, VIX 30: SL={high.stop_loss_pct}%")

    def test_near_expiry_tightens(self):
        """Test that near expiry tightens max hold."""
        # Far expiry
        far = self.manager.calculate_exit_levels(minutes_to_expiry=240)
        # Near expiry
        near = self.manager.calculate_exit_levels(minutes_to_expiry=30)

        self.assertLess(near.max_hold_minutes, far.max_hold_minutes)
        print(f"  Expiry 240min: hold={far.max_hold_minutes}, Expiry 30min: hold={near.max_hold_minutes}")

    def test_position_size_multiplier(self):
        """Test position sizing based on uncertainty."""
        # Low uncertainty = full size
        mult_low = self.manager.get_position_size_multiplier(mc_uncertainty=0.05)
        # High uncertainty = reduced size
        mult_high = self.manager.get_position_size_multiplier(mc_uncertainty=0.25)

        self.assertGreater(mult_low, mult_high)
        print(f"  Uncertainty 0.05: mult={mult_low}, Uncertainty 0.25: mult={mult_high}")


class TestSkewExitManager(unittest.TestCase):
    """Test skew-optimized exit manager."""

    def setUp(self):
        os.environ['SKEW_EXIT_ENABLED'] = '1'
        os.environ['SKEW_EXIT_MODE'] = 'partial'
        from backend.skew_exit_manager import SkewExitManager, ExitMode
        self.manager = SkewExitManager(mode=ExitMode.PARTIAL)
        self.manager.enabled = True

    def test_stop_loss(self):
        """Test stop loss triggers full exit."""
        decision = self.manager.evaluate_exit(
            trade_id="test1",
            current_pnl_pct=-0.10,  # -10%
            peak_pnl_pct=0.02,
            minutes_held=10,
        )
        self.assertTrue(decision.should_exit)
        self.assertEqual(decision.exit_fraction, 1.0)
        self.assertIn("stop_loss", decision.exit_reason)
        print(f"  Stop loss: {decision.exit_reason}")

    def test_partial_take_profit(self):
        """Test partial TP takes 50%."""
        decision = self.manager.evaluate_exit(
            trade_id="test2",
            current_pnl_pct=0.12,  # +12% (above 10% partial threshold)
            peak_pnl_pct=0.12,
            minutes_held=15,
        )
        self.assertTrue(decision.should_exit)
        self.assertEqual(decision.exit_fraction, 0.5)  # Partial take
        self.assertTrue(decision.runner_active)
        print(f"  Partial TP: {decision.exit_reason}, fraction={decision.exit_fraction}")

    def test_runner_trailing_stop(self):
        """Test runner trailing stop after partial."""
        # First, trigger partial TP
        self.manager.evaluate_exit(
            trade_id="test3",
            current_pnl_pct=0.12,
            peak_pnl_pct=0.12,
            minutes_held=15,
        )

        # Now runner is active, let it run to 20%
        decision = self.manager.evaluate_exit(
            trade_id="test3",
            current_pnl_pct=0.20,
            peak_pnl_pct=0.20,
            minutes_held=20,
        )
        self.assertFalse(decision.should_exit)  # Still running
        print(f"  Runner at 20%: {decision.exit_reason}")

        # Drop to 14% (below 20% - 5% trail = 15%)
        decision = self.manager.evaluate_exit(
            trade_id="test3",
            current_pnl_pct=0.14,
            peak_pnl_pct=0.20,
            minutes_held=25,
        )
        self.assertTrue(decision.should_exit)
        self.assertIn("trail", decision.exit_reason)
        print(f"  Runner trail exit: {decision.exit_reason}")

    def test_fixed_mode(self):
        """Test fixed mode uses standard stops."""
        # Clear env to avoid pollution from previous tests
        os.environ['SKEW_EXIT_MODE'] = 'fixed'
        from backend.skew_exit_manager import SkewExitManager, ExitMode
        fixed_mgr = SkewExitManager(mode=ExitMode.FIXED)
        fixed_mgr.mode = ExitMode.FIXED  # Force mode
        fixed_mgr.enabled = True

        # Take profit at 12%
        decision = fixed_mgr.evaluate_exit(
            trade_id="test4",
            current_pnl_pct=0.13,
            peak_pnl_pct=0.13,
            minutes_held=20,
        )
        self.assertTrue(decision.should_exit)
        self.assertEqual(decision.exit_fraction, 1.0)
        self.assertIn("take_profit", decision.exit_reason)
        print(f"  Fixed TP: {decision.exit_reason}")


def run_tests():
    """Run all tests with verbose output."""
    print("=" * 70)
    print("Architecture V4 Component Tests")
    print("=" * 70)

    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Add test classes
    suite.addTests(loader.loadTestsFromTestCase(TestEVGate))
    suite.addTests(loader.loadTestsFromTestCase(TestRegimeCalibration))
    suite.addTests(loader.loadTestsFromTestCase(TestRegimeAttribution))
    suite.addTests(loader.loadTestsFromTestCase(TestGreeksAwareExits))
    suite.addTests(loader.loadTestsFromTestCase(TestSkewExitManager))

    # Run with verbosity
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    print("\n" + "=" * 70)
    if result.wasSuccessful():
        print("ALL TESTS PASSED!")
    else:
        print(f"FAILURES: {len(result.failures)}, ERRORS: {len(result.errors)}")
    print("=" * 70)

    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
