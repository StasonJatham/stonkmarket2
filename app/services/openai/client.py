"""
OpenAI async client manager with connection pooling and circuit breaker.

Provides a robust, thread-safe client with:
- Connection pooling via httpx
- Circuit breaker pattern for fault tolerance
- Automatic client refresh on TTL expiry
- Proper async context management
"""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, AsyncGenerator

import httpx
from openai import AsyncOpenAI

from app.core.logging import get_logger
from app.services.openai.config import OpenAISettings, get_settings

if TYPE_CHECKING:
    pass


logger = get_logger("openai.client")


@dataclass
class CircuitBreakerState:
    """State for circuit breaker pattern."""
    failures: int = 0
    last_failure: datetime | None = None
    is_open: bool = False
    opened_at: datetime | None = None
    
    def record_failure(self) -> None:
        """Record a failure and potentially open the circuit."""
        self.failures += 1
        self.last_failure = datetime.now(UTC)
    
    def record_success(self) -> None:
        """Record a success and reset failure count."""
        self.failures = 0
        self.is_open = False
        self.opened_at = None
    
    def open_circuit(self) -> None:
        """Open the circuit breaker."""
        self.is_open = True
        self.opened_at = datetime.now(UTC)
        logger.warning(f"Circuit breaker opened after {self.failures} failures")
    
    def should_allow_request(self, timeout_seconds: int) -> bool:
        """Check if a request should be allowed through."""
        if not self.is_open:
            return True
        
        # Allow a test request after timeout (half-open state)
        if self.opened_at:
            elapsed = (datetime.now(UTC) - self.opened_at).total_seconds()
            if elapsed >= timeout_seconds:
                logger.info("Circuit breaker half-open, allowing test request")
                return True
        
        return False


class OpenAIClientManager:
    """
    Manages OpenAI client lifecycle with connection pooling and circuit breaker.
    
    Features:
    - Thread-safe client initialization via asyncio.Lock
    - Automatic client refresh after TTL expiry
    - Circuit breaker to prevent cascading failures
    - Connection pooling via httpx
    
    Usage:
        manager = OpenAIClientManager()
        client = await manager.get_client()
        response = await client.responses.create(...)
    """
    
    def __init__(self, settings: OpenAISettings | None = None):
        self._settings = settings or get_settings()
        self._client: AsyncOpenAI | None = None
        self._created_at: datetime | None = None
        self._lock = asyncio.Lock()
        self._circuit_breaker = CircuitBreakerState()
        self._http_client: httpx.AsyncClient | None = None
    
    @property
    def settings(self) -> OpenAISettings:
        """Get current settings."""
        return self._settings
    
    def _is_client_expired(self) -> bool:
        """Check if the current client has exceeded its TTL."""
        if not self._created_at:
            return True
        return datetime.now(UTC) - self._created_at > self._settings.client_ttl
    
    async def get_client(self) -> AsyncOpenAI | None:
        """
        Get or create an OpenAI client.
        
        Returns None if:
        - API key is not configured
        - Circuit breaker is open
        
        Thread-safe via asyncio.Lock.
        """
        # Check circuit breaker first
        if not self._circuit_breaker.should_allow_request(
            self._settings.circuit_breaker_timeout
        ):
            logger.warning("Circuit breaker open, rejecting request")
            return None
        
        async with self._lock:
            # Double-check after acquiring lock
            if self._client is not None and not self._is_client_expired():
                return self._client
            
            # Get API key from settings or database
            api_key = await self._get_api_key()
            if not api_key:
                logger.warning("OpenAI API key not configured")
                return None
            
            # Close old client if exists
            await self._close_client()
            
            # Create new HTTP client with connection pooling
            self._http_client = httpx.AsyncClient(
                limits=httpx.Limits(
                    max_connections=self._settings.max_connections,
                    max_keepalive_connections=self._settings.max_connections // 2,
                ),
                timeout=httpx.Timeout(60.0, connect=10.0),
            )
            
            # Create new OpenAI client
            self._client = AsyncOpenAI(
                api_key=api_key,
                http_client=self._http_client,
            )
            self._created_at = datetime.now(UTC)
            
            logger.debug("Created new OpenAI client")
            return self._client
    
    async def _get_api_key(self) -> str | None:
        """Get API key from settings or database."""
        # First check settings (environment variable)
        if self._settings.api_key:
            return self._settings.api_key
        
        # Fall back to database setting
        try:
            from app.core.config import get_runtime_setting
            return get_runtime_setting("openai_api_key")
        except Exception:
            return None
    
    async def _close_client(self) -> None:
        """Close the current client and HTTP client."""
        if self._http_client:
            try:
                await self._http_client.aclose()
            except Exception as e:
                logger.debug(f"Error closing HTTP client: {e}")
            self._http_client = None
        
        self._client = None
        self._created_at = None
    
    def record_success(self) -> None:
        """Record a successful API call."""
        self._circuit_breaker.record_success()
    
    def record_failure(self) -> None:
        """Record a failed API call and potentially open circuit breaker."""
        self._circuit_breaker.record_failure()
        
        if self._circuit_breaker.failures >= self._settings.circuit_breaker_threshold:
            self._circuit_breaker.open_circuit()
    
    def is_circuit_open(self) -> bool:
        """Check if circuit breaker is open."""
        return self._circuit_breaker.is_open
    
    async def close(self) -> None:
        """Close the client manager and release resources."""
        async with self._lock:
            await self._close_client()
    
    async def verify_connection(self) -> bool:
        """Verify the API key and connection are valid."""
        try:
            client = await self.get_client()
            if not client:
                return False
            
            # Simple API call to verify connection
            await client.models.list()
            self.record_success()
            return True
        except Exception as e:
            logger.warning(f"Connection verification failed: {e}")
            self.record_failure()
            return False


# Global client manager instance
_manager: OpenAIClientManager | None = None
_manager_lock = asyncio.Lock()


async def get_client_manager() -> OpenAIClientManager:
    """Get or create the global client manager."""
    global _manager
    
    if _manager is None:
        async with _manager_lock:
            if _manager is None:
                _manager = OpenAIClientManager()
    
    return _manager


async def get_client() -> AsyncOpenAI | None:
    """Get the OpenAI client from the global manager."""
    manager = await get_client_manager()
    return await manager.get_client()


@asynccontextmanager
async def openai_client() -> AsyncGenerator[AsyncOpenAI | None, None]:
    """
    Context manager for OpenAI client with automatic success/failure tracking.
    
    Usage:
        async with openai_client() as client:
            if client:
                response = await client.responses.create(...)
    """
    manager = await get_client_manager()
    client = await manager.get_client()
    
    try:
        yield client
        if client:
            manager.record_success()
    except Exception:
        manager.record_failure()
        raise


async def check_api_key() -> tuple[bool, str | None]:
    """
    Verify the OpenAI API key is valid.
    
    Returns:
        Tuple of (is_valid, error_message)
    """
    client = await get_client()
    if not client:
        return False, "API key not configured"
    
    try:
        await client.models.list()
        return True, None
    except Exception as e:
        return False, str(e)


async def get_available_models() -> list[str]:
    """
    List available GPT models from OpenAI.
    
    Returns:
        Sorted list of GPT-4/GPT-5 model IDs
    """
    client = await get_client()
    if not client:
        return []
    
    try:
        models = await client.models.list()
        return sorted([
            m.id for m in models.data
            if m.id.startswith(("gpt-4", "gpt-5", "o1", "o3"))
        ])
    except Exception as e:
        logger.warning(f"Failed to list models: {e}")
        return []
        raise
