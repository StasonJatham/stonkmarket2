"""
Tests for resilience patterns (circuit breaker, coalescing, retry).
"""

import asyncio
import time

import pytest

from app.services.data_providers.resilience import (
    CircuitBreaker,
    CircuitOpenError,
    CircuitState,
    RequestCoalescer,
    ResilientExecutor,
    RetryExhaustedError,
    retry_async,
    with_retry,
)


# =============================================================================
# Circuit Breaker Tests
# =============================================================================


class TestCircuitBreaker:
    """Tests for CircuitBreaker."""

    def test_initial_state_is_closed(self):
        """Circuit starts in closed state."""
        breaker = CircuitBreaker(name="test")
        assert breaker.state == CircuitState.CLOSED
        assert breaker.is_closed
        assert not breaker.is_open

    def test_guard_passes_when_closed(self):
        """Guard allows calls when circuit is closed."""
        breaker = CircuitBreaker(name="test")

        async def run():
            await breaker.guard()  # Should not raise

        asyncio.run(run())

    def test_opens_after_threshold_failures(self):
        """Circuit opens after reaching failure threshold."""
        breaker = CircuitBreaker(failure_threshold=3, name="test")

        breaker.record_failure()
        assert breaker.state == CircuitState.CLOSED

        breaker.record_failure()
        assert breaker.state == CircuitState.CLOSED

        breaker.record_failure()
        assert breaker.state == CircuitState.OPEN
        assert breaker.is_open

    def test_guard_raises_when_open(self):
        """Guard raises CircuitOpenError when circuit is open."""
        breaker = CircuitBreaker(failure_threshold=2, name="test")

        breaker.record_failure()
        breaker.record_failure()
        assert breaker.is_open

        async def run():
            with pytest.raises(CircuitOpenError) as exc_info:
                await breaker.guard()
            assert exc_info.value.name == "test"

        asyncio.run(run())

    def test_success_resets_failure_count(self):
        """Success resets failure count and closes circuit."""
        breaker = CircuitBreaker(failure_threshold=3, name="test")

        breaker.record_failure()
        breaker.record_failure()
        breaker.record_success()

        assert breaker._failure_count == 0
        assert breaker.state == CircuitState.CLOSED

    def test_half_open_after_recovery_timeout(self):
        """Circuit transitions to half-open after recovery timeout."""
        breaker = CircuitBreaker(
            failure_threshold=2,
            recovery_timeout=0.1,  # 100ms
            name="test",
        )

        breaker.record_failure()
        breaker.record_failure()
        assert breaker.state == CircuitState.OPEN

        time.sleep(0.15)  # Wait for recovery timeout
        assert breaker.state == CircuitState.HALF_OPEN

    def test_success_in_half_open_closes_circuit(self):
        """Success in half-open state closes the circuit."""
        breaker = CircuitBreaker(
            failure_threshold=2,
            recovery_timeout=0.05,
            name="test",
        )

        breaker.record_failure()
        breaker.record_failure()
        time.sleep(0.1)
        assert breaker.state == CircuitState.HALF_OPEN

        breaker.record_success()
        assert breaker.state == CircuitState.CLOSED
        assert breaker._failure_count == 0

    def test_failure_in_half_open_reopens_circuit(self):
        """Failure in half-open state reopens the circuit."""
        breaker = CircuitBreaker(
            failure_threshold=2,
            recovery_timeout=0.05,
            name="test",
        )

        breaker.record_failure()
        breaker.record_failure()
        time.sleep(0.1)
        assert breaker.state == CircuitState.HALF_OPEN

        breaker.record_failure()
        assert breaker.state == CircuitState.OPEN

    def test_excluded_exceptions_not_counted(self):
        """Excluded exception types don't increment failure count."""
        breaker = CircuitBreaker(
            failure_threshold=2,
            name="test",
            excluded_exceptions=(ValueError,),
        )

        breaker.record_failure(ValueError("ignored"))
        breaker.record_failure(ValueError("also ignored"))
        assert breaker._failure_count == 0
        assert breaker.state == CircuitState.CLOSED

        breaker.record_failure(RuntimeError("counted"))
        assert breaker._failure_count == 1

    def test_reset_closes_circuit(self):
        """Reset forces circuit to closed state."""
        breaker = CircuitBreaker(failure_threshold=2, name="test")

        breaker.record_failure()
        breaker.record_failure()
        assert breaker.is_open

        breaker.reset()
        assert breaker.state == CircuitState.CLOSED
        assert breaker._failure_count == 0

    def test_get_stats(self):
        """Stats returns current circuit state."""
        breaker = CircuitBreaker(
            failure_threshold=5,
            recovery_timeout=30.0,
            name="test",
        )

        breaker.record_failure()
        breaker.record_failure()

        stats = breaker.get_stats()
        assert stats["name"] == "test"
        assert stats["state"] == "closed"
        assert stats["failure_count"] == 2
        assert stats["failure_threshold"] == 5
        assert stats["recovery_timeout"] == 30.0


# =============================================================================
# Request Coalescer Tests
# =============================================================================


class TestRequestCoalescer:
    """Tests for RequestCoalescer."""

    @pytest.mark.asyncio
    async def test_single_request_executes(self):
        """Single request executes normally."""
        coalescer = RequestCoalescer()
        call_count = 0

        async def fetch():
            nonlocal call_count
            call_count += 1
            return "result"

        result = await coalescer.execute("key1", fetch)

        assert result == "result"
        assert call_count == 1

    @pytest.mark.asyncio
    async def test_concurrent_requests_coalesced(self):
        """Concurrent requests for same key share result."""
        coalescer = RequestCoalescer()
        call_count = 0

        async def fetch():
            nonlocal call_count
            call_count += 1
            await asyncio.sleep(0.1)  # Simulate slow operation
            return f"result-{call_count}"

        # Start multiple concurrent requests
        results = await asyncio.gather(
            coalescer.execute("key1", fetch),
            coalescer.execute("key1", fetch),
            coalescer.execute("key1", fetch),
        )

        # All should get the same result
        assert all(r == "result-1" for r in results)
        # Function should only be called once
        assert call_count == 1

    @pytest.mark.asyncio
    async def test_different_keys_not_coalesced(self):
        """Requests for different keys execute independently."""
        coalescer = RequestCoalescer()
        call_count = 0

        async def fetch():
            nonlocal call_count
            call_count += 1
            return f"result-{call_count}"

        results = await asyncio.gather(
            coalescer.execute("key1", fetch),
            coalescer.execute("key2", fetch),
        )

        assert results == ["result-1", "result-2"]
        assert call_count == 2

    @pytest.mark.asyncio
    async def test_error_propagated_to_waiters(self):
        """Errors are propagated to all waiting requests."""
        coalescer = RequestCoalescer()

        async def fetch():
            await asyncio.sleep(0.05)
            raise ValueError("test error")

        # All concurrent requests should get the error
        with pytest.raises(ValueError, match="test error"):
            await asyncio.gather(
                coalescer.execute("key1", fetch),
                coalescer.execute("key1", fetch),
            )

    @pytest.mark.asyncio
    async def test_sequential_requests_not_coalesced(self):
        """Sequential requests execute independently."""
        coalescer = RequestCoalescer()
        call_count = 0

        async def fetch():
            nonlocal call_count
            call_count += 1
            return f"result-{call_count}"

        result1 = await coalescer.execute("key1", fetch)
        result2 = await coalescer.execute("key1", fetch)

        assert result1 == "result-1"
        assert result2 == "result-2"
        assert call_count == 2

    @pytest.mark.asyncio
    async def test_pending_count(self):
        """Pending count reflects in-flight requests."""
        coalescer = RequestCoalescer()

        assert coalescer.get_pending_count() == 0

        async def slow_fetch():
            await asyncio.sleep(0.1)
            return "done"

        # Start request but don't await
        task = asyncio.create_task(coalescer.execute("key1", slow_fetch))
        await asyncio.sleep(0.01)  # Let it start

        assert coalescer.get_pending_count() == 1

        await task
        assert coalescer.get_pending_count() == 0


# =============================================================================
# Retry Tests
# =============================================================================


class TestRetryDecorator:
    """Tests for @with_retry decorator."""

    @pytest.mark.asyncio
    async def test_success_on_first_attempt(self):
        """Function returns immediately on success."""
        call_count = 0

        @with_retry(max_attempts=3)
        async def func():
            nonlocal call_count
            call_count += 1
            return "success"

        result = await func()

        assert result == "success"
        assert call_count == 1

    @pytest.mark.asyncio
    async def test_retry_on_failure(self):
        """Function is retried on retryable exceptions."""
        call_count = 0

        @with_retry(max_attempts=3, base_delay=0.01)
        async def func():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ConnectionError("transient")
            return "success"

        result = await func()

        assert result == "success"
        assert call_count == 3

    @pytest.mark.asyncio
    async def test_exhausted_after_max_attempts(self):
        """RetryExhaustedError raised after max attempts."""
        call_count = 0

        @with_retry(max_attempts=3, base_delay=0.01)
        async def func():
            nonlocal call_count
            call_count += 1
            raise ConnectionError("always fails")

        with pytest.raises(RetryExhaustedError) as exc_info:
            await func()

        assert exc_info.value.attempts == 3
        assert isinstance(exc_info.value.last_error, ConnectionError)
        assert call_count == 3

    @pytest.mark.asyncio
    async def test_no_retry_on_non_matching_exception(self):
        """Non-retryable exceptions are raised immediately."""
        call_count = 0

        @with_retry(max_attempts=3, retry_on=(ConnectionError,))
        async def func():
            nonlocal call_count
            call_count += 1
            raise ValueError("not retryable")

        with pytest.raises(ValueError, match="not retryable"):
            await func()

        assert call_count == 1

    @pytest.mark.asyncio
    async def test_on_retry_callback(self):
        """on_retry callback is called before each retry."""
        retries = []

        def on_retry(attempt, error):
            retries.append((attempt, str(error)))

        @with_retry(max_attempts=3, base_delay=0.01, on_retry=on_retry)
        async def func():
            raise ConnectionError("fail")

        with pytest.raises(RetryExhaustedError):
            await func()

        assert len(retries) == 2  # Called before retries 2 and 3
        assert retries[0][0] == 1
        assert retries[1][0] == 2


class TestRetryAsync:
    """Tests for retry_async function."""

    @pytest.mark.asyncio
    async def test_success(self):
        """Successful function returns result."""
        async def func():
            return "result"

        result = await retry_async(func)
        assert result == "result"

    @pytest.mark.asyncio
    async def test_retry_and_succeed(self):
        """Function is retried until success."""
        attempt = 0

        async def func():
            nonlocal attempt
            attempt += 1
            if attempt < 2:
                raise TimeoutError()
            return "success"

        result = await retry_async(func, max_attempts=3, base_delay=0.01)
        assert result == "success"
        assert attempt == 2

    @pytest.mark.asyncio
    async def test_exponential_backoff(self):
        """Delays increase exponentially."""
        times = []

        async def func():
            times.append(time.monotonic())
            raise ConnectionError()

        with pytest.raises(RetryExhaustedError):
            await retry_async(
                func,
                max_attempts=3,
                base_delay=0.05,
                exponential_base=2.0,
                jitter=0,  # No jitter for predictable timing
            )

        # Check delays (approximately)
        delay1 = times[1] - times[0]  # Should be ~0.05
        delay2 = times[2] - times[1]  # Should be ~0.10

        assert 0.04 < delay1 < 0.08
        assert 0.08 < delay2 < 0.15


# =============================================================================
# Resilient Executor Tests
# =============================================================================


class TestResilientExecutor:
    """Tests for ResilientExecutor (combined patterns)."""

    @pytest.mark.asyncio
    async def test_successful_execution(self):
        """Successful call goes through full stack."""
        executor = ResilientExecutor(name="test")

        async def fetch():
            return "data"

        result = await executor.execute("key1", fetch)
        assert result == "data"

    @pytest.mark.asyncio
    async def test_coalescing_works(self):
        """Concurrent requests are coalesced."""
        executor = ResilientExecutor(name="test")
        call_count = 0

        async def fetch():
            nonlocal call_count
            call_count += 1
            await asyncio.sleep(0.05)
            return "data"

        results = await asyncio.gather(
            executor.execute("key1", fetch),
            executor.execute("key1", fetch),
        )

        assert all(r == "data" for r in results)
        assert call_count == 1

    @pytest.mark.asyncio
    async def test_skip_coalesce(self):
        """skip_coalesce=True bypasses coalescing."""
        executor = ResilientExecutor(name="test")
        call_count = 0

        async def fetch():
            nonlocal call_count
            call_count += 1
            await asyncio.sleep(0.02)
            return f"data-{call_count}"

        results = await asyncio.gather(
            executor.execute("key1", fetch, skip_coalesce=True),
            executor.execute("key1", fetch, skip_coalesce=True),
        )

        # Both should execute independently
        assert call_count == 2

    @pytest.mark.asyncio
    async def test_retry_on_transient_error(self):
        """Transient errors trigger retry."""
        executor = ResilientExecutor(
            name="test",
            max_retries=3,
            retry_delay=0.01,
        )
        call_count = 0

        async def fetch():
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise ConnectionError("transient")
            return "success"

        result = await executor.execute("key1", fetch)
        assert result == "success"
        assert call_count == 2

    @pytest.mark.asyncio
    async def test_circuit_opens_after_failures(self):
        """Circuit opens after repeated failures."""
        executor = ResilientExecutor(
            name="test",
            failure_threshold=2,
            max_retries=1,  # Fail fast
            retry_delay=0.01,
        )

        async def always_fail():
            raise RuntimeError("permanent failure")

        # First call fails and opens circuit
        with pytest.raises((RetryExhaustedError, RuntimeError)):
            await executor.execute("key1", always_fail)

        with pytest.raises((RetryExhaustedError, RuntimeError)):
            await executor.execute("key2", always_fail)

        # Circuit should be open now
        assert executor.circuit.is_open

        # Next call should fail immediately with CircuitOpenError
        with pytest.raises(CircuitOpenError):
            await executor.execute("key3", always_fail)

    @pytest.mark.asyncio
    async def test_get_stats(self):
        """Stats include circuit and coalescing info."""
        executor = ResilientExecutor(name="test")

        stats = executor.get_stats()

        assert stats["name"] == "test"
        assert "circuit" in stats
        assert "pending_requests" in stats
        assert stats["circuit"]["state"] == "closed"
