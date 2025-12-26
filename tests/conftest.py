"""Pytest configuration and fixtures."""

from __future__ import annotations

import asyncio
import gc
import warnings
from typing import AsyncGenerator, Generator

import pytest
import pytest_asyncio
from fastapi.testclient import TestClient
from httpx import AsyncClient, ASGITransport

# Configure asyncio
pytest_plugins = ["pytest_asyncio"]


def _force_cleanup():
    """Force cleanup of pending async resources."""
    # Suppress ResourceWarning during GC - these are expected during test cleanup
    # when async resources are collected outside their event loop
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ResourceWarning)
        gc.collect()


async def _async_cleanup():
    """Async cleanup of all resources."""
    from app.database.connection import close_database, _pool as db_pool, _engine, close_sqlalchemy_engine
    from app.cache.client import close_valkey_client, _pool as cache_pool
    
    # Close SQLAlchemy engine (ORM)
    try:
        await close_sqlalchemy_engine()
    except Exception:
        pass
    
    # Close database
    try:
        await close_database()
    except Exception:
        pass
    
    # Close cache
    try:
        await close_valkey_client()
    except Exception:
        pass
    
    # Allow pending tasks to complete
    await asyncio.sleep(0.05)


def pytest_sessionfinish(session, exitstatus):
    """Called after whole test run finished, right before returning exit status."""
    # Create a fresh event loop for final cleanup
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(_async_cleanup())
        # Give time for sockets to close
        loop.run_until_complete(asyncio.sleep(0.1))
    except Exception:
        pass
    finally:
        # Cancel all pending tasks
        try:
            pending = asyncio.all_tasks(loop)
            for task in pending:
                task.cancel()
            if pending:
                loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
        except Exception:
            pass
        
        # Close the loop
        try:
            loop.run_until_complete(loop.shutdown_asyncgens())
        except Exception:
            pass
        
        loop.close()
    
    # Final garbage collection
    gc.collect()


@pytest.fixture(scope="function", autouse=True)
def cleanup_after_test():
    """Force garbage collection after each test to clean up connections."""
    # Reset SQLAlchemy engine before test to avoid event loop issues
    import app.database.connection as db_conn
    import app.services.data_providers.yfinance_service as yf_service
    import app.cache.client as cache_client
    
    # Close existing connections before resetting
    async def _close_all():
        try:
            if db_conn._engine is not None:
                await db_conn._engine.dispose()
        except Exception:
            pass
        try:
            if db_conn._pool is not None:
                await db_conn._pool.close()
        except Exception:
            pass
        try:
            if cache_client._pool is not None:
                await cache_client._pool.disconnect()
        except Exception:
            pass
    
    # Run cleanup in a new event loop if needed
    try:
        loop = asyncio.get_running_loop()
        loop.run_until_complete(_close_all())
    except RuntimeError:
        # No running loop - create one for cleanup
        try:
            asyncio.run(_close_all())
        except Exception:
            pass
    
    db_conn._engine = None
    db_conn._session_factory = None
    db_conn._pool = None
    yf_service._instance = None
    cache_client._pool = None
    cache_client._client = None
    
    yield
    
    # Close and reset again after test
    try:
        asyncio.run(_close_all())
    except Exception:
        pass
    
    db_conn._engine = None
    db_conn._session_factory = None
    db_conn._pool = None
    yf_service._instance = None
    cache_client._pool = None
    cache_client._client = None
    
    _force_cleanup()


@pytest.fixture
def client() -> Generator[TestClient, None, None]:
    """Create a test client for the FastAPI app."""
    from app.api.app import create_api_app
    
    app = create_api_app()
    with TestClient(app, raise_server_exceptions=False) as test_client:
        yield test_client
    _force_cleanup()


@pytest_asyncio.fixture
async def async_client() -> AsyncGenerator[AsyncClient, None]:
    """Create an async test client for the FastAPI app."""
    from app.api.app import create_api_app
    
    app = create_api_app()
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
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


# ============================================================================
# Mock fixtures for database-independent testing
# ============================================================================

@pytest.fixture
def mock_db(mocker):
    """Mock database operations for unit tests."""
    mock = mocker.patch("app.database.connection.fetch_one", new_callable=AsyncMock)
    mock.return_value = None
    
    mock_fetch_all = mocker.patch("app.database.connection.fetch_all", new_callable=AsyncMock)
    mock_fetch_all.return_value = []
    
    mock_execute = mocker.patch("app.database.connection.execute", new_callable=AsyncMock)
    mock_execute.return_value = None
    
    return {
        "fetch_one": mock,
        "fetch_all": mock_fetch_all,
        "execute": mock_execute,
    }


@pytest.fixture
def mock_cache(mocker):
    """Mock cache operations for unit tests."""
    mock_get = mocker.patch("app.cache.cache.Cache.get", new_callable=AsyncMock)
    mock_get.return_value = None
    
    mock_set = mocker.patch("app.cache.cache.Cache.set", new_callable=AsyncMock)
    mock_set.return_value = True
    
    mock_delete = mocker.patch("app.cache.cache.Cache.delete", new_callable=AsyncMock)
    mock_delete.return_value = True
    
    return {
        "get": mock_get,
        "set": mock_set,
        "delete": mock_delete,
    }


@pytest.fixture
def sample_stock_info() -> dict:
    """Sample stock info for testing."""
    return {
        "symbol": "AAPL",
        "name": "Apple Inc.",
        "sector": "Technology",
        "industry": "Consumer Electronics",
        "market_cap": 3000000000000,
        "pe_ratio": 28.5,
        "forward_pe": 25.2,
        "dividend_yield": 0.005,
        "beta": 1.2,
        "avg_volume": 75000000,
        "summary": "Apple Inc. designs, manufactures, and markets smartphones, personal computers, tablets, wearables, and accessories worldwide.",
        "website": "https://www.apple.com",
        "recommendation": "buy",
    }


@pytest.fixture
def sample_ranking_entry() -> dict:
    """Sample ranking entry for testing."""
    return {
        "symbol": "NVDA",
        "name": "NVIDIA Corporation",
        "depth": 0.15,
        "last_price": 450.0,
        "previous_close": 455.0,
        "change_percent": -1.1,
        "days_since_dip": 5,
        "high_52w": 530.0,
        "low_52w": 350.0,
        "market_cap": 1100000000000,
        "sector": "Technology",
        "pe_ratio": 65.0,
        "volume": 45000000,
        "updated_at": "2025-12-23T00:00:00",
    }


@pytest.fixture
def sample_chart_data() -> list[dict]:
    """Sample chart data points for testing."""
    import datetime
    
    base_date = datetime.date(2025, 9, 1)
    data = []
    price = 100.0
    ref_high = 120.0
    
    for i in range(90):
        date = base_date + datetime.timedelta(days=i)
        price = price * (1 + (0.002 if i < 30 else -0.003))
        drawdown = (price - ref_high) / ref_high
        
        data.append({
            "date": date.isoformat(),
            "close": round(price, 2),
            "ref_high": ref_high,
            "threshold": ref_high * 0.9,
            "drawdown": round(drawdown, 4),
            "since_dip": None,
            "ref_high_date": "2025-09-15",
            "dip_start_date": "2025-10-01",
        })
    
    return data

