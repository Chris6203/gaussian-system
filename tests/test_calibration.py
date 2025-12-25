#!/usr/bin/env python3
"""
Unit Tests for Calibration Components
=====================================

Tests for:
1. CalibrationTracker (Platt + Isotonic hybrid)
2. ModelHealthMonitor (drift detection)
3. HealthCheckSystem (pre-cycle checks)
4. FeatureCache (LRU caching)
5. PrioritizedReplayBuffer (PER)
6. SharpeRewardCalculator

Run with:
    python -m pytest tests/test_calibration.py -v
    
Or:
    python tests/test_calibration.py
"""

import sys
import os
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import unittest
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch


class TestCalibrationTracker(unittest.TestCase):
    """Tests for CalibrationTracker class."""
    
    def setUp(self):
        """Set up test fixtures."""
        from backend.calibration_tracker import CalibrationTracker
        self.tracker = CalibrationTracker(
            horizon_minutes=15,
            buffer_size=100,
            min_samples_for_calibration=20,
            refit_interval=10,
            use_hybrid=True
        )
    
    def test_initialization(self):
        """Test tracker initializes correctly."""
        self.assertEqual(self.tracker.horizon_minutes, 15)
        self.assertEqual(self.tracker.buffer_size, 100)
        self.assertTrue(self.tracker.use_hybrid)
        self.assertEqual(len(self.tracker._pending), 0)
    
    def test_record_prediction(self):
        """Test recording predictions."""
        self.tracker.record_prediction(
            confidence=0.7,
            predicted_direction='UP',
            entry_price=100.0
        )
        
        self.assertEqual(len(self.tracker._pending), 1)
        self.assertEqual(self.tracker._total_recorded, 1)
    
    def test_record_prediction_ignores_hold(self):
        """Test that HOLD predictions are ignored."""
        initial_count = len(self.tracker._pending)
        
        # Try to record invalid direction
        self.tracker.record_prediction(
            confidence=0.5,
            predicted_direction='HOLD',
            entry_price=100.0
        )
        
        self.assertEqual(len(self.tracker._pending), initial_count)
    
    def test_settle_predictions(self):
        """Test settling predictions after horizon."""
        # Record a prediction
        past_time = datetime.now() - timedelta(minutes=20)
        self.tracker.record_prediction(
            confidence=0.65,
            predicted_direction='UP',
            entry_price=100.0,
            timestamp=past_time
        )
        
        # Settle with higher price (prediction was correct)
        settled = self.tracker.settle_predictions(current_price=101.0)
        
        self.assertEqual(settled, 1)
        self.assertEqual(len(self.tracker._calibration_buffer), 1)
        
        # Check that direction was correct
        conf, correct = self.tracker._calibration_buffer[-1]
        self.assertEqual(conf, 0.65)
        self.assertTrue(correct)
    
    def test_settle_predictions_incorrect(self):
        """Test settling incorrect predictions."""
        past_time = datetime.now() - timedelta(minutes=20)
        self.tracker.record_prediction(
            confidence=0.8,
            predicted_direction='UP',
            entry_price=100.0,
            timestamp=past_time
        )
        
        # Settle with lower price (prediction was wrong)
        settled = self.tracker.settle_predictions(current_price=99.0)
        
        self.assertEqual(settled, 1)
        conf, correct = self.tracker._calibration_buffer[-1]
        self.assertFalse(correct)
    
    def test_calibrate_returns_raw_without_data(self):
        """Test calibration returns raw confidence without enough data."""
        result = self.tracker.calibrate(0.7)
        self.assertEqual(result, 0.7)
    
    def test_platt_scaling_fitting(self):
        """Test Platt scaling fits correctly."""
        # Add synthetic calibration data
        np.random.seed(42)
        for _ in range(50):
            conf = np.random.uniform(0.4, 0.9)
            # Higher confidence should correlate with correct predictions
            correct = np.random.random() < conf
            self.tracker._calibration_buffer.append((conf, correct))
        
        # Fit calibration
        success = self.tracker._fit_calibration()
        
        self.assertTrue(success)
        self.assertTrue(self.tracker._platt.n_samples > 0)
    
    def test_isotonic_fitting(self):
        """Test isotonic regression fitting."""
        # Add calibration data with clear pattern
        for i in range(50):
            conf = 0.3 + i * 0.01
            correct = conf > 0.5
            self.tracker._calibration_buffer.append((conf, correct))
        
        # Fit calibration
        self.tracker._fit_calibration()
        
        self.assertTrue(self.tracker._isotonic.n_samples > 0)
        self.assertTrue(len(self.tracker._isotonic.x_points) > 0)
    
    def test_hybrid_calibration(self):
        """Test hybrid calibration combines Platt and Isotonic."""
        # Add enough data
        np.random.seed(42)
        for _ in range(60):
            conf = np.random.uniform(0.3, 0.9)
            correct = np.random.random() < conf
            self.tracker._calibration_buffer.append((conf, correct))
        
        self.tracker._fit_calibration()
        
        # Test calibration
        raw = 0.7
        calibrated = self.tracker.calibrate(raw)
        
        # Should use hybrid method
        self.assertEqual(self.tracker._calibration_method, 'hybrid')
        
        # Result should be between 0 and 1
        self.assertGreaterEqual(calibrated, 0.0)
        self.assertLessEqual(calibrated, 1.0)
    
    def test_get_metrics(self):
        """Test metrics calculation."""
        # Add enough data to trigger metric calculation (need > 20)
        np.random.seed(123)
        for i in range(25):
            conf = 0.5 + np.random.uniform(0, 0.4)
            correct = np.random.random() < 0.5  # 50% accuracy
            self.tracker._calibration_buffer.append((conf, correct))
        
        # Fit calibration first
        self.tracker._fit_calibration()
        
        metrics = self.tracker.get_metrics()
        
        self.assertIn('brier_score', metrics)
        self.assertIn('ece', metrics)
        self.assertIn('direction_accuracy', metrics)
        self.assertIn('sample_count', metrics)
        
        # Sample count should match what we added
        self.assertEqual(metrics['sample_count'], 25)


class TestModelHealthMonitor(unittest.TestCase):
    """Tests for ModelHealthMonitor class."""
    
    def setUp(self):
        """Set up test fixtures."""
        from backend.model_health import ModelHealthMonitor
        self.monitor = ModelHealthMonitor(
            error_window=50,
            error_threshold=0.02
        )
    
    def test_initialization(self):
        """Test monitor initializes correctly."""
        self.assertEqual(self.monitor.error_threshold, 0.02)
        self.assertEqual(len(self.monitor._errors), 0)
    
    def test_record_prediction_error(self):
        """Test recording prediction errors."""
        self.monitor.record_prediction_error(
            predicted_return=0.01,
            actual_return=0.005
        )
        
        self.assertEqual(len(self.monitor._errors), 1)
        self.assertAlmostEqual(self.monitor._errors[0], 0.005, places=5)
    
    def test_record_direction(self):
        """Test recording direction accuracy."""
        self.monitor.record_prediction_error(
            predicted_return=0.01,
            actual_return=0.005,
            predicted_direction='UP',
            actual_direction='UP'
        )
        
        self.assertEqual(len(self.monitor._direction_results), 1)
        self.assertTrue(self.monitor._direction_results[0])
    
    def test_health_status_healthy(self):
        """Test health status with good performance."""
        # Add good predictions
        for _ in range(50):
            self.monitor.record_prediction_error(
                predicted_return=0.01,
                actual_return=0.009,  # Small error
                predicted_direction='UP',
                actual_direction='UP'
            )
        
        status = self.monitor.get_health_status()
        
        self.assertIn('health_score', status)
        self.assertIn('status', status)
        self.assertGreater(status['health_score'], 0.5)
    
    def test_health_status_unhealthy(self):
        """Test health status with poor performance."""
        # Add bad predictions with high errors
        for _ in range(50):
            self.monitor.record_prediction_error(
                predicted_return=0.05,
                actual_return=-0.02,  # Large error
                predicted_direction='UP',
                actual_direction='DOWN'
            )
        
        status = self.monitor.get_health_status()
        
        self.assertLess(status['health_score'], 0.7)
    
    def test_record_calibration_metric(self):
        """Test recording calibration metrics."""
        for i in range(25):
            self.monitor.record_calibration_metric(brier_score=0.2 + i * 0.001)
        
        self.assertEqual(len(self.monitor._brier_history), 25)
        self.assertIsNotNone(self.monitor._baseline_brier)


class TestHealthCheckSystem(unittest.TestCase):
    """Tests for HealthCheckSystem class."""
    
    def setUp(self):
        """Set up test fixtures."""
        from backend.health_checks import HealthCheckSystem, AlertManager
        
        self.alert_manager = AlertManager(alert_cooldown_minutes=0)  # No cooldown for tests
        self.health_check = HealthCheckSystem(
            alert_manager=self.alert_manager,
            max_data_age_seconds=300
        )
    
    def test_pre_cycle_checks_pass(self):
        """Test pre-cycle checks with healthy system."""
        result = self.health_check.run_pre_cycle_checks(
            data_timestamp=datetime.now() - timedelta(seconds=60),
            current_positions=2
        )
        
        self.assertTrue(result.can_trade)
        self.assertEqual(len(result.blocking_issues), 0)
    
    def test_pre_cycle_checks_stale_data(self):
        """Test pre-cycle checks block on stale data."""
        result = self.health_check.run_pre_cycle_checks(
            data_timestamp=datetime.now() - timedelta(seconds=600),  # 10 min old
            current_positions=0
        )
        
        self.assertFalse(result.can_trade)
        self.assertGreater(len(result.blocking_issues), 0)
    
    def test_pre_cycle_checks_position_limit(self):
        """Test pre-cycle checks block on position limit."""
        result = self.health_check.run_pre_cycle_checks(
            data_timestamp=datetime.now(),
            current_positions=15  # Over default max of 10
        )
        
        self.assertFalse(result.can_trade)


class TestFeatureCache(unittest.TestCase):
    """Tests for FeatureCache class."""
    
    def setUp(self):
        """Set up test fixtures."""
        from backend.feature_cache import FeatureCache
        self.cache = FeatureCache(max_size=10, ttl_seconds=60)
    
    def test_initialization(self):
        """Test cache initializes correctly."""
        self.assertEqual(self.cache.max_size, 10)
        self.assertEqual(self.cache.ttl_seconds, 60)
    
    def test_put_and_get(self):
        """Test basic put and get."""
        self.cache.put('key1', {'value': 42})
        
        hit, value = self.cache.get('key1')
        
        self.assertTrue(hit)
        self.assertEqual(value['value'], 42)
    
    def test_cache_miss(self):
        """Test cache miss returns False."""
        hit, value = self.cache.get('nonexistent')
        
        self.assertFalse(hit)
        self.assertIsNone(value)
    
    def test_lru_eviction(self):
        """Test LRU eviction when full."""
        # Fill cache
        for i in range(10):
            self.cache.put(f'key{i}', i)
        
        # Add one more (should evict oldest)
        self.cache.put('key10', 10)
        
        # key0 should be evicted
        hit, _ = self.cache.get('key0')
        self.assertFalse(hit)
        
        # key10 should exist
        hit, value = self.cache.get('key10')
        self.assertTrue(hit)
        self.assertEqual(value, 10)
    
    def test_make_key(self):
        """Test key generation."""
        arr1 = np.array([1.0, 2.0, 3.0])
        arr2 = np.array([1.0, 2.0, 3.0])
        arr3 = np.array([1.0, 2.0, 4.0])
        
        key1 = self.cache.make_key('test', arr1)
        key2 = self.cache.make_key('test', arr2)
        key3 = self.cache.make_key('test', arr3)
        
        # Same input should produce same key
        self.assertEqual(key1, key2)
        
        # Different input should produce different key
        self.assertNotEqual(key1, key3)
    
    def test_cached_decorator(self):
        """Test caching decorator."""
        call_count = 0
        
        @self.cache.cached('expensive_func')
        def expensive_func(x):
            nonlocal call_count
            call_count += 1
            return x * 2
        
        # First call
        result1 = expensive_func(5)
        self.assertEqual(result1, 10)
        self.assertEqual(call_count, 1)
        
        # Second call (should be cached)
        result2 = expensive_func(5)
        self.assertEqual(result2, 10)
        self.assertEqual(call_count, 1)  # Not incremented
        
        # Different input
        result3 = expensive_func(10)
        self.assertEqual(result3, 20)
        self.assertEqual(call_count, 2)
    
    def test_stats(self):
        """Test statistics tracking."""
        self.cache.put('a', 1)
        self.cache.get('a')  # Hit
        self.cache.get('b')  # Miss
        
        stats = self.cache.get_stats()
        
        self.assertEqual(stats['hits'], 1)
        self.assertEqual(stats['misses'], 1)
        self.assertEqual(stats['entries'], 1)


class TestPrioritizedReplayBuffer(unittest.TestCase):
    """Tests for PrioritizedReplayBuffer class."""
    
    def setUp(self):
        """Set up test fixtures."""
        from backend.rl_enhancements import PrioritizedReplayBuffer
        self.buffer = PrioritizedReplayBuffer(
            capacity=100,
            alpha=0.6,
            beta=0.4
        )
    
    def test_initialization(self):
        """Test buffer initializes correctly."""
        self.assertEqual(self.buffer.capacity, 100)
        self.assertEqual(len(self.buffer), 0)
    
    def test_add_experience(self):
        """Test adding experiences."""
        state = np.array([1.0, 2.0, 3.0])
        
        self.buffer.add(
            state=state,
            action=1,
            reward=0.5,
            next_state=state,
            done=False
        )
        
        self.assertEqual(len(self.buffer), 1)
    
    def test_sample(self):
        """Test sampling from buffer."""
        # Add experiences
        for i in range(50):
            self.buffer.add(
                state=np.array([i, i+1, i+2]),
                action=i % 5,
                reward=i * 0.01,
                next_state=np.array([i+1, i+2, i+3]),
                done=False
            )
        
        experiences, weights, indices = self.buffer.sample(10)
        
        self.assertEqual(len(experiences), 10)
        self.assertEqual(len(weights), 10)
        self.assertEqual(len(indices), 10)
        
        # Weights should be normalized (max should be close to 1)
        self.assertLessEqual(max(weights), 1.0)
    
    def test_priority_update(self):
        """Test updating priorities."""
        # Add experiences
        for i in range(10):
            self.buffer.add(
                state=np.array([i]),
                action=0,
                reward=0.1,
                next_state=np.array([i+1]),
                done=False
            )
        
        _, _, indices = self.buffer.sample(5)
        
        # Update priorities with TD errors
        td_errors = np.array([0.1, 0.5, 0.2, 0.8, 0.3])
        self.buffer.update_priorities(indices, td_errors)
        
        stats = self.buffer.get_stats()
        self.assertGreater(stats['priority_updates'], 0)


class TestSharpeRewardCalculator(unittest.TestCase):
    """Tests for SharpeRewardCalculator class."""
    
    def setUp(self):
        """Set up test fixtures."""
        from backend.rl_enhancements import SharpeRewardCalculator
        self.calculator = SharpeRewardCalculator()
    
    def test_initialization(self):
        """Test calculator initializes correctly."""
        self.assertIsNotNone(self.calculator.trade_history)
    
    def test_calculate_win(self):
        """Test reward calculation for winning trade."""
        reward = self.calculator.calculate(
            trade_result={'pnl': 50.0, 'time_held_minutes': 30},
            context={'account_balance': 10000.0}
        )
        
        # Should be positive for winning trade
        self.assertGreater(reward, 0)
    
    def test_calculate_loss(self):
        """Test reward calculation for losing trade."""
        reward = self.calculator.calculate(
            trade_result={'pnl': -50.0, 'time_held_minutes': 60},
            context={'account_balance': 10000.0}
        )
        
        # Should be negative for losing trade
        self.assertLess(reward, 0)
    
    def test_sharpe_calculation(self):
        """Test Sharpe ratio calculation."""
        # Add trades with returns
        returns = [0.01, 0.02, -0.005, 0.015, 0.01, -0.01, 0.02, 0.005, 0.01, 0.015]
        
        for ret in returns:
            self.calculator.trade_history.add_return(ret)
        
        sharpe = self.calculator.trade_history.get_sharpe()
        
        # Should be positive with mostly positive returns
        self.assertGreater(sharpe, 0)
    
    def test_calibration_penalty(self):
        """Test penalty for overconfident wrong predictions."""
        # High confidence loss
        reward_high_conf = self.calculator.calculate(
            trade_result={'pnl': -50.0},
            context={'account_balance': 10000.0, 'confidence': 0.9}
        )
        
        # Reset for fair comparison
        self.calculator = type(self.calculator)()
        
        # Low confidence loss
        reward_low_conf = self.calculator.calculate(
            trade_result={'pnl': -50.0},
            context={'account_balance': 10000.0, 'confidence': 0.5}
        )
        
        # High confidence should be penalized more
        self.assertLess(reward_high_conf, reward_low_conf)


class TestIntegration(unittest.TestCase):
    """Integration tests for calibration system."""
    
    def test_calibration_flow(self):
        """Test full calibration flow."""
        from backend.calibration_tracker import CalibrationTracker
        
        tracker = CalibrationTracker(
            horizon_minutes=1,  # Short for testing
            min_samples_for_calibration=10
        )
        
        # Simulate trading cycle
        predictions = []
        for i in range(20):
            conf = 0.5 + np.random.uniform(-0.2, 0.3)
            direction = 'UP' if np.random.random() > 0.5 else 'DOWN'
            price = 100.0 + np.random.uniform(-1, 1)
            
            tracker.record_prediction(
                confidence=conf,
                predicted_direction=direction,
                entry_price=price,
                timestamp=datetime.now() - timedelta(minutes=5)  # Already past horizon
            )
            predictions.append((conf, direction, price))
        
        # Settle all predictions
        current_price = 100.5
        settled = tracker.settle_predictions(current_price)
        
        self.assertEqual(settled, 20)
        
        # Get metrics
        metrics = tracker.get_metrics()
        
        self.assertGreater(metrics['sample_count'], 0)
        self.assertIn('brier_score', metrics)
        
        # Test calibration
        calibrated = tracker.calibrate(0.7)
        self.assertGreaterEqual(calibrated, 0)
        self.assertLessEqual(calibrated, 1)


def run_tests():
    """Run all tests."""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add test classes
    suite.addTests(loader.loadTestsFromTestCase(TestCalibrationTracker))
    suite.addTests(loader.loadTestsFromTestCase(TestModelHealthMonitor))
    suite.addTests(loader.loadTestsFromTestCase(TestHealthCheckSystem))
    suite.addTests(loader.loadTestsFromTestCase(TestFeatureCache))
    suite.addTests(loader.loadTestsFromTestCase(TestPrioritizedReplayBuffer))
    suite.addTests(loader.loadTestsFromTestCase(TestSharpeRewardCalculator))
    suite.addTests(loader.loadTestsFromTestCase(TestIntegration))
    
    # Run with verbosity
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Summary
    print("\n" + "=" * 70)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Skipped: {len(result.skipped)}")
    print("=" * 70)
    
    return result


if __name__ == '__main__':
    run_tests()

