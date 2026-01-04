"""
Resilience patterns for external API calls.

This module provides:
1. Circuit Breaker - Fail fast after consecutive failures
2. Request Coalescing - Deduplicate concurrent requests
3. Retry with Tenacity - Exponential backoff with jitter

Usage:
    from app.services.data_providers.resilience import (
        CircuitBreaker,
        RequestCoalescer,
        with_retry,
    )
    
    # Circuit breaker
    breaker = CircuitBreaker(
        failure_threshold=5,
        recovery_timeout=30.0,
        name="yfinance"
    )
    
    async def fetch_data():
        await breaker.guard()  # Raises CircuitOpenError if open
        try:
            result = await do_fetch()
            breaker.record_success()
            return result
        except Exception as e:
            breaker.record_failure(e)
            raise
    
    # Request coalescing
    coalescer = RequestCoalescer()
    
    async def get_ticker(symbol: str):
        async with coalescer.coalesce(f"ticker:{symbol}"):
            return await fetch_ticker(symbol)
    
    # Retry decorator
    @with_retry(max_attempts=3, base_delay=1.0)
    async def reliable_fetch():
        return await fetch_something()
"""

from __future__ import annotations

import asyncio
import functools
import random
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, TypeVar

from app.core.logging import get_logger

logger = get_logger("resilience")

# Type for decorated async functions
T = TypeVar("T")


# =============================================================================
# Exceptions
# =============================================================================


class CircuitOpenError(Exception):
    """Raised when circuit breaker is open and blocking calls."""

    def __init__(self, name: str, message: str = "Circuit breaker is open"):
        self.name = name
        self.message = message
        super().__init__(f"{name}: {message}")


class RetryExhaustedError(Exception):
    """Raised when all retry attempts are exhausted."""

    def __init__(self, attempts: int, last_error: Exception | None = None):
        self.attempts = attempts
        self.last_error = last_error
        message = f"All {attempts} retry attempts exhausted"
        if last_error:
            message += f": {last_error}"
        super().__init__(message)


# =============================================================================
# Circuit Breaker
# =============================================================================


class CircuitState(Enum):
    """Circuit breaker states."""

    CLOSED = "closed"  # Normal operation
    OPEN = "open"  # Failing fast, blocking calls
    HALF_OPEN = "half_open"  # Testing if service recovered


@dataclass
class CircuitBreaker:
    """
    Circuit breaker pattern for fail-fast protection.
    
    States:
    - CLOSED: Normal operation, counting failures
    - OPEN: After threshold failures, block all calls
    - HALF_OPEN: After recovery timeout, allow test request
    
    Args:
        failure_threshold: Number of consecutive failures before opening
        recovery_timeout: Seconds to wait before testing (half-open)
        name: Identifier for logging
        excluded_exceptions: Exception types that shouldn't trigger circuit
    """

    failure_threshold: int = 5
    recovery_timeout: float = 30.0
    name: str = "circuit"
    excluded_exceptions: tuple[type, ...] = ()

    # Internal state
    _state: CircuitState = field(default=CircuitState.CLOSED, init=False)
    _failure_count: int = field(default=0, init=False)
    _last_failure_time: float | None = field(default=None, init=False)
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock, init=False)

    @property
    def state(self) -> CircuitState:
        """Current circuit state (may transition from OPEN to HALF_OPEN)."""
        if self._state == CircuitState.OPEN and self._last_failure_time:
            if time.monotonic() - self._last_failure_time >= self.recovery_timeout:
                return CircuitState.HALF_OPEN
        return self._state

    @property
    def is_closed(self) -> bool:
        """True if circuit is closed (normal operation)."""
        return self.state == CircuitState.CLOSED

    @property
    def is_open(self) -> bool:
        """True if circuit is open (blocking calls)."""
        return self.state == CircuitState.OPEN

    async def guard(self) -> None:
        """
        Guard entry to protected code. Raises CircuitOpenError if open.
        
        Call this before attempting the protected operation.
        """
        state = self.state

        if state == CircuitState.OPEN:
            raise CircuitOpenError(
                self.name,
                f"Circuit open after {self._failure_count} failures, "
                f"retry in {self.recovery_timeout - (time.monotonic() - (self._last_failure_time or 0)):.1f}s",
            )

        # HALF_OPEN allows one test request
        if state == CircuitState.HALF_OPEN:
            logger.info(f"[{self.name}] Circuit half-open, allowing test request")

    def record_success(self) -> None:
        """Record a successful call, reset failure count."""
        if self._state != CircuitState.CLOSED:
            logger.info(f"[{self.name}] Circuit closed after successful recovery")

        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._last_failure_time = None

    def record_failure(self, error: Exception | None = None) -> None:
        """
        Record a failed call. Opens circuit after threshold failures.
        
        Args:
            error: The exception that caused the failure
        """
        # Don't count excluded exceptions
        if error and isinstance(error, self.excluded_exceptions):
            return

        self._failure_count += 1
        self._last_failure_time = time.monotonic()

        if self._failure_count >= self.failure_threshold:
            if self._state != CircuitState.OPEN:
                logger.warning(
                    f"[{self.name}] Circuit OPEN after {self._failure_count} failures"
                )
            self._state = CircuitState.OPEN
        else:
            logger.debug(
                f"[{self.name}] Failure {self._failure_count}/{self.failure_threshold}"
            )

    def reset(self) -> None:
        """Force reset to closed state."""
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._last_failure_time = None

    def get_stats(self) -> dict[str, Any]:
        """Get circuit breaker statistics."""
        return {
            "name": self.name,
            "state": self.state.value,
            "failure_count": self._failure_count,
            "failure_threshold": self.failure_threshold,
            "recovery_timeout": self.recovery_timeout,
            "last_failure_age": (
                time.monotonic() - self._last_failure_time
                if self._last_failure_time
                else None
            ),
        }


# =============================================================================
# Request Coalescing
# =============================================================================


@dataclass
class _PendingRequest:
    """Tracks an in-flight request that others can wait on."""

    future: asyncio.Future[Any]
    created_at: float = field(default_factory=time.monotonic)


class RequestCoalescer:
    """
    Coalesces concurrent requests for the same resource.
    
    When multiple callers request the same key simultaneously, only one
    actual request is made and all callers receive the same result.
    
    Usage:
        coalescer = RequestCoalescer()
        
        async def get_data(key: str):
            async with coalescer.coalesce(key):
                return await expensive_fetch(key)
    """

    def __init__(self, max_wait: float = 10.0):
        """
        Args:
            max_wait: Maximum seconds to wait for coalesced request
        """
        self._pending: dict[str, _PendingRequest] = {}
        self._locks: dict[str, asyncio.Lock] = {}
        self._max_wait = max_wait

    def _get_lock(self, key: str) -> asyncio.Lock:
        """Get or create lock for a key."""
        if key not in self._locks:
            self._locks[key] = asyncio.Lock()
        return self._locks[key]

    @asynccontextmanager
    async def coalesce(self, key: str):
        """
        Context manager for request coalescing.
        
        If another request for the same key is in flight, wait for it.
        Otherwise, execute the block and share result with waiters.
        
        Args:
            key: Unique identifier for the request
            
        Yields:
            Control to caller's code block
        """
        lock = self._get_lock(key)

        # Fast path: check if request is already pending
        if key in self._pending:
            pending = self._pending[key]
            try:
                result = await asyncio.wait_for(
                    asyncio.shield(pending.future),
                    timeout=self._max_wait,
                )
                # Return the coalesced result
                # We need a way to signal "use this result instead"
                # Using a simple wrapper that caller can check
                raise _CoalescedResult(result)
            except asyncio.TimeoutError:
                logger.warning(f"Coalesce timeout for {key}, proceeding independently")
            except _CoalescedResult:
                raise
            except Exception:
                # Original request failed, we'll proceed with our own
                pass

        # Slow path: acquire lock and either wait or execute
        async with lock:
            # Check again after acquiring lock
            if key in self._pending:
                pending = self._pending[key]
                try:
                    result = await asyncio.wait_for(
                        asyncio.shield(pending.future),
                        timeout=self._max_wait,
                    )
                    raise _CoalescedResult(result)
                except asyncio.TimeoutError:
                    pass
                except _CoalescedResult:
                    raise

            # We're the leader - create pending request
            loop = asyncio.get_running_loop()
            future: asyncio.Future[Any] = loop.create_future()
            self._pending[key] = _PendingRequest(future=future)

            try:
                yield  # Let caller execute their code
            except Exception as e:
                # Propagate error to waiters
                if not future.done():
                    future.set_exception(e)
                raise
            finally:
                # Clean up
                if key in self._pending:
                    del self._pending[key]

            # Note: Caller must call coalescer.set_result(key, result) to share result
            # Or we can improve this API...

    async def execute(
        self, key: str, func: Callable[[], Any]
    ) -> Any:
        """
        Execute function with request coalescing.
        
        Simpler API than context manager - handles result sharing automatically.
        
        Args:
            key: Unique identifier for the request
            func: Async callable to execute
            
        Returns:
            Result from func (either executed or coalesced)
        """
        lock = self._get_lock(key)

        # Check if already pending
        if key in self._pending:
            pending = self._pending[key]
            try:
                return await asyncio.wait_for(
                    asyncio.shield(pending.future),
                    timeout=self._max_wait,
                )
            except asyncio.TimeoutError:
                logger.warning(f"Coalesce timeout for {key}")
            except asyncio.CancelledError:
                raise
            except Exception:
                pass  # Original failed, try our own

        async with lock:
            # Double-check after lock
            if key in self._pending:
                pending = self._pending[key]
                try:
                    return await asyncio.wait_for(
                        asyncio.shield(pending.future),
                        timeout=self._max_wait,
                    )
                except (asyncio.TimeoutError, Exception):
                    pass

            # Create pending entry
            loop = asyncio.get_running_loop()
            future: asyncio.Future[Any] = loop.create_future()
            self._pending[key] = _PendingRequest(future=future)

            try:
                result = await func()
                if not future.done():
                    future.set_result(result)
                return result
            except Exception as e:
                if not future.done():
                    future.set_exception(e)
                raise
            finally:
                self._pending.pop(key, None)

    def get_pending_count(self) -> int:
        """Return number of pending coalesced requests."""
        return len(self._pending)


class _CoalescedResult(Exception):
    """Internal exception to signal coalesced result."""

    def __init__(self, result: Any):
        self.result = result
        super().__init__()


# =============================================================================
# Retry with Exponential Backoff
# =============================================================================

# Default exceptions that should trigger retry
DEFAULT_RETRY_EXCEPTIONS = (
    ConnectionError,
    TimeoutError,
    asyncio.TimeoutError,
)


def with_retry(
    max_attempts: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 10.0,
    exponential_base: float = 2.0,
    jitter: float = 0.5,
    retry_on: tuple[type[Exception], ...] = DEFAULT_RETRY_EXCEPTIONS,
    on_retry: Callable[[int, Exception], None] | None = None,
) -> Callable:
    """
    Decorator for retry with exponential backoff and jitter.
    
    Args:
        max_attempts: Maximum number of attempts (including initial)
        base_delay: Initial delay in seconds
        max_delay: Maximum delay cap
        exponential_base: Base for exponential growth (default 2.0)
        jitter: Random jitter factor (0.5 = ±50% of delay)
        retry_on: Exception types that trigger retry
        on_retry: Callback(attempt_number, exception) before each retry
        
    Returns:
        Decorated async function with retry logic
    """

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs) -> T:
            last_error: Exception | None = None

            for attempt in range(1, max_attempts + 1):
                try:
                    return await func(*args, **kwargs)
                except retry_on as e:
                    last_error = e

                    if attempt >= max_attempts:
                        logger.warning(
                            f"Retry exhausted for {func.__name__} after {max_attempts} attempts: {e}"
                        )
                        raise RetryExhaustedError(max_attempts, last_error) from e

                    # Calculate delay with exponential backoff
                    delay = min(
                        base_delay * (exponential_base ** (attempt - 1)),
                        max_delay,
                    )

                    # Add jitter
                    if jitter > 0:
                        delay *= 1 + (random.random() - 0.5) * 2 * jitter

                    logger.debug(
                        f"Retry {attempt}/{max_attempts} for {func.__name__} "
                        f"after {delay:.2f}s: {e}"
                    )

                    if on_retry:
                        on_retry(attempt, e)

                    await asyncio.sleep(delay)

            # Should not reach here, but satisfy type checker
            raise RetryExhaustedError(max_attempts, last_error)

        return wrapper

    return decorator


async def retry_async(
    func: Callable[[], T],
    max_attempts: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 10.0,
    exponential_base: float = 2.0,
    jitter: float = 0.5,
    retry_on: tuple[type[Exception], ...] = DEFAULT_RETRY_EXCEPTIONS,
) -> T:
    """
    Retry an async function with exponential backoff.
    
    Functional API alternative to the decorator.
    
    Args:
        func: Async callable to retry
        max_attempts: Maximum attempts
        base_delay: Initial delay
        max_delay: Max delay cap
        exponential_base: Exponential growth base
        jitter: Jitter factor
        retry_on: Exceptions to retry on
        
    Returns:
        Result from successful func call
        
    Raises:
        RetryExhaustedError: If all attempts fail
    """
    last_error: Exception | None = None

    for attempt in range(1, max_attempts + 1):
        try:
            return await func()
        except retry_on as e:
            last_error = e

            if attempt >= max_attempts:
                raise RetryExhaustedError(max_attempts, last_error) from e

            delay = min(
                base_delay * (exponential_base ** (attempt - 1)),
                max_delay,
            )
            if jitter > 0:
                delay *= 1 + (random.random() - 0.5) * 2 * jitter

            await asyncio.sleep(delay)

    raise RetryExhaustedError(max_attempts, last_error)


# =============================================================================
# Combined Resilient Executor
# =============================================================================


class ResilientExecutor:
    """
    Combines circuit breaker, coalescing, and retry into single executor.
    
    Usage:
        executor = ResilientExecutor(name="yfinance")
        
        async def fetch():
            return await executor.execute(
                key="ticker:AAPL",
                func=lambda: fetch_ticker("AAPL"),
            )
    """

    def __init__(
        self,
        name: str = "resilient",
        failure_threshold: int = 5,
        recovery_timeout: float = 30.0,
        max_retries: int = 3,
        retry_delay: float = 1.0,
        coalesce_timeout: float = 10.0,
        retry_on: tuple[type[Exception], ...] = DEFAULT_RETRY_EXCEPTIONS,
    ):
        self.name = name
        self.circuit = CircuitBreaker(
            failure_threshold=failure_threshold,
            recovery_timeout=recovery_timeout,
            name=name,
            excluded_exceptions=retry_on,  # Don't open circuit on retriable errors
        )
        self.coalescer = RequestCoalescer(max_wait=coalesce_timeout)
        self._max_retries = max_retries
        self._retry_delay = retry_delay
        self._retry_on = retry_on

    async def execute(
        self,
        key: str,
        func: Callable[[], T],
        skip_coalesce: bool = False,
    ) -> T:
        """
        Execute function with full resilience stack.
        
        Order: Coalescing → Circuit Breaker → Retry
        
        Args:
            key: Unique key for coalescing
            func: Async callable to execute
            skip_coalesce: If True, skip coalescing (for writes)
            
        Returns:
            Result from func
        """

        async def _with_circuit_and_retry() -> T:
            # Check circuit
            await self.circuit.guard()

            try:
                result = await retry_async(
                    func,
                    max_attempts=self._max_retries,
                    base_delay=self._retry_delay,
                    retry_on=self._retry_on,
                )
                self.circuit.record_success()
                return result
            except RetryExhaustedError as e:
                # All retries failed - record as circuit failure
                self.circuit.record_failure(e.last_error)
                raise
            except Exception as e:
                self.circuit.record_failure(e)
                raise

        if skip_coalesce:
            return await _with_circuit_and_retry()

        return await self.coalescer.execute(key, _with_circuit_and_retry)

    def get_stats(self) -> dict[str, Any]:
        """Get executor statistics."""
        return {
            "name": self.name,
            "circuit": self.circuit.get_stats(),
            "pending_requests": self.coalescer.get_pending_count(),
        }
