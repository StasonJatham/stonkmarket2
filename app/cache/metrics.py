"""Cache metrics and monitoring.

Provides hit/miss tracking and observability for cache performance.

Usage:
    from app.cache.metrics import cache_metrics

    # Record a cache hit
    cache_metrics.record_hit("ranking")

    # Record a cache miss
    cache_metrics.record_miss("ranking")

    # Get stats
    stats = cache_metrics.get_stats()
    # {"ranking": {"hits": 150, "misses": 10, "hit_rate": 0.9375}, ...}
"""

from __future__ import annotations

import time
from collections import defaultdict
from dataclasses import dataclass

from app.core.logging import get_logger


logger = get_logger("cache.metrics")


@dataclass
class CacheStats:
    """Statistics for a single cache prefix."""

    hits: int = 0
    misses: int = 0
    errors: int = 0
    total_get_time_ms: float = 0.0
    total_set_time_ms: float = 0.0
    get_count: int = 0
    set_count: int = 0
    last_hit: float | None = None
    last_miss: float | None = None

    @property
    def hit_rate(self) -> float:
        """Calculate cache hit rate."""
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0

    @property
    def avg_get_time_ms(self) -> float:
        """Average GET operation time in milliseconds."""
        return self.total_get_time_ms / self.get_count if self.get_count > 0 else 0.0

    @property
    def avg_set_time_ms(self) -> float:
        """Average SET operation time in milliseconds."""
        return self.total_set_time_ms / self.set_count if self.set_count > 0 else 0.0

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "hits": self.hits,
            "misses": self.misses,
            "errors": self.errors,
            "hit_rate": round(self.hit_rate, 4),
            "avg_get_time_ms": round(self.avg_get_time_ms, 2),
            "avg_set_time_ms": round(self.avg_set_time_ms, 2),
            "total_operations": self.hits + self.misses,
        }


class CacheMetrics:
    """Global cache metrics collector.
    
    Thread-safe metrics collection for cache operations.
    In production, this could be replaced with Prometheus metrics.
    """

    def __init__(self):
        self._stats: dict[str, CacheStats] = defaultdict(CacheStats)
        self._start_time = time.time()

    def record_hit(self, prefix: str, duration_ms: float = 0.0) -> None:
        """Record a cache hit."""
        stats = self._stats[prefix]
        stats.hits += 1
        stats.last_hit = time.time()
        if duration_ms > 0:
            stats.total_get_time_ms += duration_ms
            stats.get_count += 1

    def record_miss(self, prefix: str, duration_ms: float = 0.0) -> None:
        """Record a cache miss."""
        stats = self._stats[prefix]
        stats.misses += 1
        stats.last_miss = time.time()
        if duration_ms > 0:
            stats.total_get_time_ms += duration_ms
            stats.get_count += 1

    def record_set(self, prefix: str, duration_ms: float = 0.0) -> None:
        """Record a cache set operation."""
        stats = self._stats[prefix]
        if duration_ms > 0:
            stats.total_set_time_ms += duration_ms
            stats.set_count += 1

    def record_error(self, prefix: str) -> None:
        """Record a cache error."""
        self._stats[prefix].errors += 1

    def get_stats(self, prefix: str | None = None) -> dict:
        """Get cache statistics.
        
        Args:
            prefix: Optional specific prefix to get stats for.
                    If None, returns all stats.
        
        Returns:
            Dictionary of cache statistics
        """
        if prefix:
            if prefix in self._stats:
                return {prefix: self._stats[prefix].to_dict()}
            return {}

        return {p: s.to_dict() for p, s in self._stats.items()}

    def get_summary(self) -> dict:
        """Get summary statistics across all caches."""
        total_hits = sum(s.hits for s in self._stats.values())
        total_misses = sum(s.misses for s in self._stats.values())
        total_errors = sum(s.errors for s in self._stats.values())
        total = total_hits + total_misses

        return {
            "total_hits": total_hits,
            "total_misses": total_misses,
            "total_errors": total_errors,
            "overall_hit_rate": round(total_hits / total, 4) if total > 0 else 0.0,
            "uptime_seconds": round(time.time() - self._start_time, 0),
            "caches_tracked": len(self._stats),
            "by_cache": self.get_stats(),
        }

    def reset(self) -> None:
        """Reset all statistics."""
        self._stats.clear()
        self._start_time = time.time()
        logger.info("Cache metrics reset")

    def log_stats(self) -> None:
        """Log current cache statistics."""
        summary = self.get_summary()
        logger.info(
            f"Cache stats: {summary['total_hits']} hits, "
            f"{summary['total_misses']} misses, "
            f"{summary['overall_hit_rate']:.1%} hit rate",
            extra={"cache_stats": summary},
        )


# Global singleton instance
cache_metrics = CacheMetrics()


# Context manager for timing cache operations
class CacheTimer:
    """Context manager for timing cache operations.
    
    Usage:
        with CacheTimer("ranking") as timer:
            value = await cache.get(key)
            timer.was_hit = value is not None
    """

    def __init__(self, prefix: str):
        self.prefix = prefix
        self.start_time: float = 0.0
        self.was_hit: bool = False
        self.is_set: bool = False

    def __enter__(self) -> CacheTimer:
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        duration_ms = (time.perf_counter() - self.start_time) * 1000

        if exc_type:
            cache_metrics.record_error(self.prefix)
        elif self.is_set:
            cache_metrics.record_set(self.prefix, duration_ms)
        elif self.was_hit:
            cache_metrics.record_hit(self.prefix, duration_ms)
        else:
            cache_metrics.record_miss(self.prefix, duration_ms)
