"""Pytest configuration and fixtures."""

from __future__ import annotations

import asyncio
from typing import AsyncGenerator, Generator

import pytest
import pytest_asyncio
from fastapi.testclient import TestClient
from httpx import AsyncClient

# Configure asyncio
pytest_plugins = ["pytest_asyncio"]


@pytest.fixture(scope="session")
def event_loop() -> Generator:
    """Create an event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def client() -> Generator[TestClient, None, None]:
    """Create a test client for the FastAPI app."""
    from app.api.app import create_api_app
    
    app = create_api_app()
    with TestClient(app) as test_client:
        yield test_client


@pytest_asyncio.fixture
async def async_client() -> AsyncGenerator[AsyncClient, None]:
    """Create an async test client for the FastAPI app."""
    from app.api.app import create_api_app
    
    app = create_api_app()
    async with AsyncClient(app=app, base_url="http://test") as ac:
        yield ac


@pytest.fixture
def auth_token() -> str:
    """Create a valid JWT token for testing."""
    from app.core.security import create_access_token
    return create_access_token(username="test_user", is_admin=False)


@pytest.fixture
def admin_token() -> str:
    """Create an admin JWT token for testing."""
    from app.core.security import create_access_token
    return create_access_token(username="test_admin", is_admin=True)


@pytest.fixture
def auth_headers(auth_token: str) -> dict:
    """Create authorization headers with a regular user token."""
    return {"Authorization": f"Bearer {auth_token}"}


@pytest.fixture
def admin_headers(admin_token: str) -> dict:
    """Create authorization headers with an admin token."""
    return {"Authorization": f"Bearer {admin_token}"}


@pytest.fixture
def sample_prices() -> list[float]:
    """Sample price series for testing."""
    # 100 days of prices with a dip pattern
    import numpy as np
    
    np.random.seed(42)
    
    # Start at 100, trend up to 150, then dip to 120
    prices = []
    price = 100.0
    
    # Rising phase (days 0-50)
    for i in range(50):
        price = price * (1 + 0.005 + np.random.randn() * 0.01)
        prices.append(price)
    
    # Peak and decline (days 50-80)
    peak_price = price
    for i in range(30):
        price = price * (1 - 0.01 + np.random.randn() * 0.008)
        prices.append(price)
    
    # Stabilize (days 80-100)
    for i in range(20):
        price = price * (1 + np.random.randn() * 0.005)
        prices.append(price)
    
    return prices


@pytest.fixture
def sample_prices_array(sample_prices) -> "np.ndarray":
    """Sample prices as numpy array."""
    import numpy as np
    return np.array(sample_prices)
