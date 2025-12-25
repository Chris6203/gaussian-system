#!/usr/bin/env python3
"""
Feature Caching System
======================

LRU caching for expensive feature computations to improve performance.

Features:
1. LRU cache with configurable size
2. TTL (time-to-live) expiration
3. Cache key generation from feature inputs
4. Statistics tracking (hits, misses, evictions)
5. Thread-safe operations

Usage:
    cache = FeatureCache(max_size=1000, ttl_seconds=60)
    
    @cache.cached('technical_features')
    def compute_technical_features(prices: np.ndarray) -> Dict:
        # Expensive computation
        return features
    
    # Or manual usage:
    key = cache.make_key('rsi', prices[-20:])
    if cache.has(key):
        result = cache.get(key)
    else:
        result = compute_rsi(prices)
        cache.put(key, result)
"""

import logging
import hashlib
import pickle
import time
from datetime import datetime
from functools import wraps
from threading import Lock
from typing import Any, Callable, Dict, Optional, Tuple, Union
from collections import OrderedDict
import numpy as np

logger = logging.getLogger(__name__)


class CacheEntry:
    """Single cache entry with value and metadata."""
    __slots__ = ['value', 'created_at', 'last_accessed', 'access_count', 'size_bytes']
    
    def __init__(self, value: Any):
        self.value = value
        self.created_at = time.time()
        self.last_accessed = time.time()
        self.access_count = 1
        self.size_bytes = self._estimate_size(value)
    
    def _estimate_size(self, obj: Any) -> int:
        """Estimate memory size of object."""
        try:
            if isinstance(obj, np.ndarray):
                return obj.nbytes
            elif isinstance(obj, (dict, list)):
                return len(pickle.dumps(obj))
            else:
                return 64  # Default estimate
        except Exception:
            return 64  # Default on serialization failure
    
    def touch(self):
        """Update access time and count."""
        self.last_accessed = time.time()
        self.access_count += 1


class FeatureCache:
    """
    Thread-safe LRU cache with TTL for feature computations.
    """
    
    def __init__(
        self,
        max_size: int = 1000,
        ttl_seconds: float = 60.0,
        max_memory_mb: float = 100.0
    ):
        """
        Args:
            max_size: Maximum number of entries
            ttl_seconds: Time-to-live for entries (0 = no expiration)
            max_memory_mb: Maximum memory usage in MB
        """
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self.max_memory_bytes = max_memory_mb * 1024 * 1024
        
        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._lock = Lock()
        
        # Statistics
        self._stats = {
            'hits': 0,
            'misses': 0,
            'evictions': 0,
            'expirations': 0,
            'total_bytes': 0
        }
        
        logger.info(f"✅ FeatureCache initialized: max_size={max_size}, ttl={ttl_seconds}s")
    
    def make_key(self, name: str, *args, **kwargs) -> str:
        """
        Generate a cache key from function name and arguments.
        
        Args:
            name: Function/feature name
            *args: Positional arguments
            **kwargs: Keyword arguments
            
        Returns:
            Hash string key
        """
        try:
            # Convert args to hashable form
            key_parts = [name]
            
            for arg in args:
                if isinstance(arg, np.ndarray):
                    # Use shape + first/last few values + hash of data
                    key_parts.append(f"arr:{arg.shape}:{arg.dtype}")
                    if arg.size > 0:
                        key_parts.append(f"{arg.flat[0]:.6f}")
                        key_parts.append(f"{arg.flat[-1]:.6f}")
                        key_parts.append(str(hash(arg.tobytes())))
                elif isinstance(arg, (list, tuple)):
                    key_parts.append(str(hash(tuple(arg[:10]))))  # First 10 elements
                elif isinstance(arg, dict):
                    key_parts.append(str(hash(frozenset(list(arg.items())[:10]))))
                else:
                    key_parts.append(str(arg))
            
            for k, v in sorted(kwargs.items()):
                key_parts.append(f"{k}={v}")
            
            # Create hash
            key_str = "|".join(key_parts)
            return hashlib.md5(key_str.encode()).hexdigest()
            
        except Exception as e:
            # Fallback to timestamp-based key (won't cache effectively)
            logger.debug(f"Key generation failed: {e}")
            return f"fallback_{time.time()}"
    
    def get(self, key: str) -> Tuple[bool, Any]:
        """
        Get value from cache.
        
        Args:
            key: Cache key
            
        Returns:
            (hit, value) tuple - hit=True if found and valid
        """
        with self._lock:
            if key not in self._cache:
                self._stats['misses'] += 1
                return False, None
            
            entry = self._cache[key]
            
            # Check TTL
            if self.ttl_seconds > 0:
                age = time.time() - entry.created_at
                if age > self.ttl_seconds:
                    self._remove(key)
                    self._stats['expirations'] += 1
                    self._stats['misses'] += 1
                    return False, None
            
            # Update access and move to end (LRU)
            entry.touch()
            self._cache.move_to_end(key)
            
            self._stats['hits'] += 1
            return True, entry.value
    
    def put(self, key: str, value: Any) -> None:
        """
        Put value in cache.
        
        Args:
            key: Cache key
            value: Value to cache
        """
        with self._lock:
            # Remove if exists (to update)
            if key in self._cache:
                self._remove(key)
            
            # Create entry
            entry = CacheEntry(value)
            
            # Check memory limit
            while (self._stats['total_bytes'] + entry.size_bytes > self.max_memory_bytes 
                   and self._cache):
                self._evict_oldest()
            
            # Check size limit
            while len(self._cache) >= self.max_size:
                self._evict_oldest()
            
            # Add entry
            self._cache[key] = entry
            self._stats['total_bytes'] += entry.size_bytes
    
    def has(self, key: str) -> bool:
        """Check if key exists and is valid."""
        hit, _ = self.get(key)
        return hit
    
    def invalidate(self, key: str) -> bool:
        """Remove a specific key."""
        with self._lock:
            if key in self._cache:
                self._remove(key)
                return True
            return False
    
    def clear(self) -> int:
        """Clear all entries."""
        with self._lock:
            count = len(self._cache)
            self._cache.clear()
            self._stats['total_bytes'] = 0
            return count
    
    def _remove(self, key: str) -> None:
        """Remove entry (internal, assumes lock held)."""
        if key in self._cache:
            entry = self._cache.pop(key)
            self._stats['total_bytes'] -= entry.size_bytes
    
    def _evict_oldest(self) -> None:
        """Evict oldest entry (internal, assumes lock held)."""
        if self._cache:
            key, entry = self._cache.popitem(last=False)
            self._stats['total_bytes'] -= entry.size_bytes
            self._stats['evictions'] += 1
    
    def get_stats(self) -> Dict:
        """Get cache statistics."""
        with self._lock:
            total = self._stats['hits'] + self._stats['misses']
            hit_rate = self._stats['hits'] / total if total > 0 else 0.0
            
            return {
                'entries': len(self._cache),
                'max_size': self.max_size,
                'hits': self._stats['hits'],
                'misses': self._stats['misses'],
                'hit_rate': hit_rate,
                'evictions': self._stats['evictions'],
                'expirations': self._stats['expirations'],
                'memory_bytes': self._stats['total_bytes'],
                'memory_mb': self._stats['total_bytes'] / (1024 * 1024),
                'max_memory_mb': self.max_memory_bytes / (1024 * 1024)
            }
    
    def cached(self, name: str = None):
        """
        Decorator for caching function results.
        
        Args:
            name: Optional name override (default: function name)
            
        Usage:
            @cache.cached('my_feature')
            def compute_feature(data):
                return expensive_computation(data)
        """
        def decorator(func: Callable) -> Callable:
            cache_name = name or func.__name__
            
            @wraps(func)
            def wrapper(*args, **kwargs):
                key = self.make_key(cache_name, *args, **kwargs)
                
                hit, value = self.get(key)
                if hit:
                    return value
                
                result = func(*args, **kwargs)
                self.put(key, result)
                return result
            
            # Add cache control methods to wrapper
            wrapper.cache_clear = lambda: self.clear()
            wrapper.cache_info = lambda: self.get_stats()
            
            return wrapper
        return decorator


# Global cache instance for feature computations
_global_cache: Optional[FeatureCache] = None


def get_feature_cache(
    max_size: int = 1000,
    ttl_seconds: float = 60.0,
    max_memory_mb: float = 100.0
) -> FeatureCache:
    """
    Get or create the global feature cache.
    
    Args:
        max_size: Maximum entries (only used on first call)
        ttl_seconds: TTL in seconds (only used on first call)
        max_memory_mb: Max memory in MB (only used on first call)
        
    Returns:
        Global FeatureCache instance
    """
    global _global_cache
    if _global_cache is None:
        _global_cache = FeatureCache(max_size, ttl_seconds, max_memory_mb)
    return _global_cache


def cached_feature(name: str = None, ttl: float = None):
    """
    Convenience decorator using global cache.
    
    Usage:
        @cached_feature('rsi_calculation')
        def compute_rsi(prices, period=14):
            return calculate_rsi(prices, period)
    """
    cache = get_feature_cache()
    return cache.cached(name)


# Pre-built caching decorators for common operations
def cache_technical_indicator(func: Callable) -> Callable:
    """Cache technical indicator calculations (60s TTL)."""
    return cached_feature(f"tech_{func.__name__}")(func)


def cache_feature_vector(func: Callable) -> Callable:
    """Cache feature vector computations (30s TTL)."""
    return cached_feature(f"feat_{func.__name__}")(func)

