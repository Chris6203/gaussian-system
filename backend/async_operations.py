#!/usr/bin/env python3
"""
Async Operations for Trading Engine
====================================

Provides async/parallel execution capabilities for:
1. Parallel data fetching across multiple symbols
2. Concurrent feature computation
3. Async model inference
4. Background prediction settlement

This module uses asyncio and ThreadPoolExecutor for true parallelism.

Usage:
    # Parallel data fetching
    async_ops = AsyncOperations()
    data = await async_ops.fetch_multiple_symbols(['SPY', 'QQQ', 'IWM'])
    
    # Or sync interface
    data = async_ops.fetch_multiple_symbols_sync(['SPY', 'QQQ', 'IWM'])
"""

import asyncio
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import partial
from typing import Any, Callable, Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import threading

logger = logging.getLogger(__name__)


@dataclass
class TaskResult:
    """Result of an async task."""
    name: str
    success: bool
    result: Any
    error: Optional[str]
    duration_ms: float
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()


class AsyncOperations:
    """
    Async operations manager for trading engine.
    
    Provides both sync and async interfaces for parallel operations.
    """
    
    def __init__(
        self,
        max_workers: int = 4,
        timeout_seconds: float = 30.0
    ):
        """
        Args:
            max_workers: Maximum concurrent workers
            timeout_seconds: Default timeout for operations
        """
        self.max_workers = max_workers
        self.timeout_seconds = timeout_seconds
        
        # Thread pool for CPU-bound and sync operations
        self._executor = ThreadPoolExecutor(max_workers=max_workers)
        
        # Async event loop (created lazily)
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._loop_thread: Optional[threading.Thread] = None
        
        # Statistics
        self._stats = {
            'total_tasks': 0,
            'successful_tasks': 0,
            'failed_tasks': 0,
            'total_duration_ms': 0
        }
        
        logger.info(f"✅ AsyncOperations initialized: {max_workers} workers")
    
    def _get_or_create_loop(self) -> asyncio.AbstractEventLoop:
        """Get or create async event loop."""
        if self._loop is None or not self._loop.is_running():
            # Create new loop in a dedicated thread
            self._loop = asyncio.new_event_loop()
            
            def run_loop():
                asyncio.set_event_loop(self._loop)
                self._loop.run_forever()
            
            self._loop_thread = threading.Thread(target=run_loop, daemon=True)
            self._loop_thread.start()
            
            # Give the loop time to start
            time.sleep(0.1)
        
        return self._loop
    
    # =========================================================================
    # SYNC INTERFACE (easier to use, wraps async)
    # =========================================================================
    
    def fetch_multiple_symbols_sync(
        self,
        symbols: List[str],
        data_source: Any,
        period: str = "7d",
        interval: str = "1m"
    ) -> Dict[str, Any]:
        """
        Fetch data for multiple symbols in parallel (sync interface).
        
        Args:
            symbols: List of symbols to fetch
            data_source: Data source with get_data method
            period: Data period
            interval: Data interval
            
        Returns:
            Dict mapping symbol to data (or None on failure)
        """
        results = {}
        futures = []
        
        def fetch_one(symbol: str):
            try:
                start = time.time()
                data = data_source.get_data(symbol, period=period, interval=interval)
                duration = (time.time() - start) * 1000
                return TaskResult(
                    name=symbol,
                    success=data is not None and not data.empty,
                    result=data,
                    error=None,
                    duration_ms=duration
                )
            except Exception as e:
                return TaskResult(
                    name=symbol,
                    success=False,
                    result=None,
                    error=str(e),
                    duration_ms=0
                )
        
        # Submit all tasks
        with ThreadPoolExecutor(max_workers=min(len(symbols), self.max_workers)) as executor:
            future_to_symbol = {
                executor.submit(fetch_one, symbol): symbol 
                for symbol in symbols
            }
            
            # Collect results
            for future in as_completed(future_to_symbol, timeout=self.timeout_seconds):
                symbol = future_to_symbol[future]
                try:
                    task_result = future.result()
                    results[symbol] = task_result.result
                    self._update_stats(task_result)
                except Exception as e:
                    logger.warning(f"Failed to fetch {symbol}: {e}")
                    results[symbol] = None
        
        return results
    
    def run_parallel_sync(
        self,
        tasks: List[Tuple[str, Callable, tuple, dict]]
    ) -> Dict[str, TaskResult]:
        """
        Run multiple tasks in parallel (sync interface).
        
        Args:
            tasks: List of (name, func, args, kwargs) tuples
            
        Returns:
            Dict mapping task name to TaskResult
        """
        results = {}
        
        def run_task(name: str, func: Callable, args: tuple, kwargs: dict) -> TaskResult:
            try:
                start = time.time()
                result = func(*args, **kwargs)
                duration = (time.time() - start) * 1000
                return TaskResult(
                    name=name,
                    success=True,
                    result=result,
                    error=None,
                    duration_ms=duration
                )
            except Exception as e:
                return TaskResult(
                    name=name,
                    success=False,
                    result=None,
                    error=str(e),
                    duration_ms=0
                )
        
        with ThreadPoolExecutor(max_workers=min(len(tasks), self.max_workers)) as executor:
            future_to_name = {
                executor.submit(run_task, name, func, args, kwargs): name
                for name, func, args, kwargs in tasks
            }
            
            for future in as_completed(future_to_name, timeout=self.timeout_seconds):
                name = future_to_name[future]
                try:
                    task_result = future.result()
                    results[name] = task_result
                    self._update_stats(task_result)
                except Exception as e:
                    results[name] = TaskResult(
                        name=name,
                        success=False,
                        result=None,
                        error=str(e),
                        duration_ms=0
                    )
        
        return results
    
    def compute_features_parallel(
        self,
        feature_functions: List[Tuple[str, Callable, dict]],
        data: Any
    ) -> Dict[str, Any]:
        """
        Compute multiple features in parallel.
        
        Args:
            feature_functions: List of (name, func, kwargs) - func takes data as first arg
            data: Data to pass to each function
            
        Returns:
            Dict mapping feature name to computed value
        """
        tasks = [
            (name, func, (data,), kwargs)
            for name, func, kwargs in feature_functions
        ]
        
        results = self.run_parallel_sync(tasks)
        
        return {
            name: result.result
            for name, result in results.items()
            if result.success
        }
    
    # =========================================================================
    # ASYNC INTERFACE (for advanced usage)
    # =========================================================================
    
    async def fetch_multiple_symbols_async(
        self,
        symbols: List[str],
        data_source: Any,
        period: str = "7d",
        interval: str = "1m"
    ) -> Dict[str, Any]:
        """
        Fetch data for multiple symbols in parallel (async interface).
        
        Args:
            symbols: List of symbols to fetch
            data_source: Data source with get_data method
            period: Data period
            interval: Data interval
            
        Returns:
            Dict mapping symbol to data
        """
        async def fetch_one(symbol: str) -> Tuple[str, Any]:
            loop = asyncio.get_event_loop()
            try:
                data = await loop.run_in_executor(
                    self._executor,
                    partial(data_source.get_data, symbol, period=period, interval=interval)
                )
                return symbol, data
            except Exception as e:
                logger.warning(f"Failed to fetch {symbol}: {e}")
                return symbol, None
        
        # Run all fetches concurrently
        tasks = [fetch_one(symbol) for symbol in symbols]
        results = await asyncio.gather(*tasks)
        
        return dict(results)
    
    async def run_parallel_async(
        self,
        tasks: List[Tuple[str, Callable, tuple, dict]]
    ) -> Dict[str, TaskResult]:
        """
        Run multiple tasks in parallel (async interface).
        
        Args:
            tasks: List of (name, func, args, kwargs) tuples
            
        Returns:
            Dict mapping task name to TaskResult
        """
        async def run_task(name: str, func: Callable, args: tuple, kwargs: dict) -> Tuple[str, TaskResult]:
            loop = asyncio.get_event_loop()
            try:
                start = time.time()
                result = await loop.run_in_executor(
                    self._executor,
                    partial(func, *args, **kwargs)
                )
                duration = (time.time() - start) * 1000
                return name, TaskResult(
                    name=name,
                    success=True,
                    result=result,
                    error=None,
                    duration_ms=duration
                )
            except Exception as e:
                return name, TaskResult(
                    name=name,
                    success=False,
                    result=None,
                    error=str(e),
                    duration_ms=0
                )
        
        # Run all tasks concurrently
        async_tasks = [
            run_task(name, func, args, kwargs)
            for name, func, args, kwargs in tasks
        ]
        results = await asyncio.gather(*async_tasks)
        
        return dict(results)
    
    # =========================================================================
    # BACKGROUND TASKS
    # =========================================================================
    
    def submit_background_task(
        self,
        func: Callable,
        *args,
        callback: Optional[Callable[[TaskResult], None]] = None,
        **kwargs
    ) -> None:
        """
        Submit a task to run in the background.
        
        Args:
            func: Function to run
            *args: Positional arguments
            callback: Optional callback when task completes
            **kwargs: Keyword arguments
        """
        def wrapped():
            try:
                start = time.time()
                result = func(*args, **kwargs)
                duration = (time.time() - start) * 1000
                task_result = TaskResult(
                    name=func.__name__,
                    success=True,
                    result=result,
                    error=None,
                    duration_ms=duration
                )
            except Exception as e:
                task_result = TaskResult(
                    name=func.__name__,
                    success=False,
                    result=None,
                    error=str(e),
                    duration_ms=0
                )
            
            self._update_stats(task_result)
            
            if callback:
                callback(task_result)
        
        self._executor.submit(wrapped)
    
    # =========================================================================
    # UTILITIES
    # =========================================================================
    
    def _update_stats(self, result: TaskResult) -> None:
        """Update statistics with task result."""
        self._stats['total_tasks'] += 1
        self._stats['total_duration_ms'] += result.duration_ms
        
        if result.success:
            self._stats['successful_tasks'] += 1
        else:
            self._stats['failed_tasks'] += 1
    
    def get_stats(self) -> Dict:
        """Get operation statistics."""
        total = self._stats['total_tasks']
        return {
            **self._stats,
            'success_rate': self._stats['successful_tasks'] / total if total > 0 else 0,
            'avg_duration_ms': self._stats['total_duration_ms'] / total if total > 0 else 0
        }
    
    def shutdown(self) -> None:
        """Shutdown the executor."""
        self._executor.shutdown(wait=True)
        
        if self._loop and self._loop.is_running():
            self._loop.call_soon_threadsafe(self._loop.stop)
        
        logger.info("AsyncOperations shutdown complete")


# =============================================================================
# ASYNC CYCLE RUNNER
# =============================================================================

class AsyncCycleRunner:
    """
    Runs trading cycles with async operations.
    
    Allows parallel:
    - Data fetching
    - Feature computation
    - Prediction settlement
    """
    
    def __init__(
        self,
        data_source: Any,
        bot: Any,
        symbols: List[str] = None,
        max_workers: int = 4
    ):
        """
        Args:
            data_source: Data source for fetching
            bot: Trading bot instance
            symbols: Symbols to fetch (default: primary + context)
            max_workers: Max parallel workers
        """
        self.data_source = data_source
        self.bot = bot
        self.symbols = symbols or ['SPY']
        self.async_ops = AsyncOperations(max_workers=max_workers)
        
        # Pre-cycle hooks
        self._pre_cycle_hooks: List[Callable] = []
        self._post_cycle_hooks: List[Callable] = []
    
    def add_pre_cycle_hook(self, func: Callable) -> None:
        """Add function to run before cycle."""
        self._pre_cycle_hooks.append(func)
    
    def add_post_cycle_hook(self, func: Callable) -> None:
        """Add function to run after cycle."""
        self._post_cycle_hooks.append(func)
    
    def run_cycle(self) -> Dict:
        """
        Run one trading cycle with parallel operations.
        
        Returns:
            Cycle results dict
        """
        cycle_start = time.time()
        results = {
            'data': {},
            'features': {},
            'signal': None,
            'errors': [],
            'timing': {}
        }
        
        # Run pre-cycle hooks in parallel
        if self._pre_cycle_hooks:
            hook_tasks = [
                (f"hook_{i}", hook, (), {})
                for i, hook in enumerate(self._pre_cycle_hooks)
            ]
            self.async_ops.run_parallel_sync(hook_tasks)
        
        # Parallel data fetch
        fetch_start = time.time()
        data = self.async_ops.fetch_multiple_symbols_sync(
            symbols=self.symbols,
            data_source=self.data_source
        )
        results['data'] = data
        results['timing']['fetch_ms'] = (time.time() - fetch_start) * 1000
        
        # Check primary symbol
        primary_symbol = self.symbols[0] if self.symbols else 'SPY'
        if data.get(primary_symbol) is None:
            results['errors'].append(f"Failed to fetch {primary_symbol}")
            return results
        
        # Update bot market context with parallel-fetched data
        for symbol, symbol_data in data.items():
            if symbol_data is not None and hasattr(self.bot, 'market_context'):
                try:
                    self.bot.market_context[symbol] = {
                        'close': float(symbol_data['close'].iloc[-1]),
                        'data': symbol_data
                    }
                except (KeyError, IndexError, AttributeError):
                    pass  # Skip symbols with incomplete data
        
        # Generate signal (can't easily parallelize model inference)
        signal_start = time.time()
        try:
            signal = self.bot.generate_unified_signal(primary_symbol)
            results['signal'] = signal
        except Exception as e:
            results['errors'].append(f"Signal generation failed: {e}")
        results['timing']['signal_ms'] = (time.time() - signal_start) * 1000
        
        # Run post-cycle hooks in parallel
        if self._post_cycle_hooks:
            hook_tasks = [
                (f"hook_{i}", hook, (results,), {})
                for i, hook in enumerate(self._post_cycle_hooks)
            ]
            self.async_ops.run_parallel_sync(hook_tasks)
        
        results['timing']['total_ms'] = (time.time() - cycle_start) * 1000
        
        return results


# =============================================================================
# FACTORY FUNCTIONS
# =============================================================================

def create_async_operations(max_workers: int = 4) -> AsyncOperations:
    """Create AsyncOperations instance."""
    return AsyncOperations(max_workers=max_workers)


def create_async_cycle_runner(
    data_source: Any,
    bot: Any,
    symbols: List[str] = None,
    max_workers: int = 4
) -> AsyncCycleRunner:
    """Create AsyncCycleRunner instance."""
    return AsyncCycleRunner(
        data_source=data_source,
        bot=bot,
        symbols=symbols,
        max_workers=max_workers
    )


# =============================================================================
# UTILITY DECORATOR
# =============================================================================

def run_in_background(executor: ThreadPoolExecutor = None):
    """
    Decorator to run function in background thread.
    
    Usage:
        @run_in_background()
        def slow_operation():
            ...
    """
    _executor = executor or ThreadPoolExecutor(max_workers=2)
    
    def decorator(func: Callable) -> Callable:
        def wrapper(*args, **kwargs):
            return _executor.submit(func, *args, **kwargs)
        return wrapper
    
    return decorator

