"""
System Test Configuration - pytest fixtures for Docker log interception.

This conftest creates a "grey-box" testing environment where:
1. Tests run against real Docker containers (no mocks for DB/Celery)
2. All container logs are monitored during test execution
3. Any error/warning in any container fails the test
4. Celery task completion is verified before test passes
5. AI-friendly debug reports are generated on failure

CRITICAL: This file must be in system_tests/ to apply only to system tests.
The existing tests/ folder continues to use its own conftest with mocks.
"""

from __future__ import annotations

import os
import time
from datetime import datetime, UTC
from pathlib import Path
from typing import Generator

import docker
import httpx
import pytest
from docker.models.containers import Container

from system_tests.config import SystemTestConfig, get_config
from system_tests.fixtures.docker_logs import DockerLogWatcher
from system_tests.fixtures.celery_sync import CeleryTaskTracker
from system_tests.fixtures.db_state import DatabaseSnapshot
from system_tests.fixtures.stack_health import StackHealthChecker
from system_tests.reporters.ai_debug_report import AIDebugReport


# =============================================================================
# CONFIGURATION
# =============================================================================


@pytest.fixture(scope="session")
def system_config() -> SystemTestConfig:
    """Load system test configuration from environment."""
    return get_config()


# =============================================================================
# DOCKER CLIENT AND CONTAINERS
# =============================================================================


@pytest.fixture(scope="session")
def docker_client() -> Generator[docker.DockerClient, None, None]:
    """Create Docker client for container interaction."""
    client = docker.from_env()
    yield client
    client.close()


@pytest.fixture(scope="session")
def containers(
    docker_client: docker.DockerClient,
    system_config: SystemTestConfig,
) -> dict[str, Container]:
    """
    Get references to all monitored containers.

    Returns a dict mapping logical names to Container objects.
    Fails if any required container is not running.
    """
    result = {}
    missing = []

    for name, container_name in system_config.containers.items():
        try:
            container = docker_client.containers.get(container_name)
            container.reload()
            result[name] = container
        except docker.errors.NotFound:
            missing.append(container_name)

    if missing:
        pytest.fail(
            f"Required containers not found: {missing}\n\n"
            f"Start the test stack with:\n"
            f"  docker compose -f docker-compose.test.yml up -d\n\n"
            f"Or use the dev stack:\n"
            f"  SYSTEM_TEST_STRICT=0 docker compose -f docker-compose.dev.yml up -d"
        )

    return result


# =============================================================================
# STACK HEALTH CHECK (Session-scoped)
# =============================================================================


@pytest.fixture(scope="session")
def stack_health(
    containers: dict[str, Container],
    system_config: SystemTestConfig,
) -> StackHealthChecker:
    """Create stack health checker."""
    return StackHealthChecker(containers, system_config)


@pytest.fixture(scope="session", autouse=True)
def verify_stack_healthy(
    stack_health: StackHealthChecker,
    system_config: SystemTestConfig,
):
    """
    Verify entire stack is healthy before running any tests.

    This runs once at the start of the test session.
    """
    print("\n" + "=" * 60)
    print("🔍 HOLISTIC SYSTEM VERIFICATION FRAMEWORK")
    print("=" * 60)
    print(f"  Strict Mode: {'ON' if system_config.strict_mode else 'OFF'}")
    print(f"  API URL: {system_config.api_base_url}")
    print("=" * 60)

    try:
        stack_health.wait_for_healthy(timeout=30)
        print("\n✅ All containers healthy, starting system tests...\n")
    except AssertionError as e:
        stack_health.print_status()
        pytest.fail(str(e))


# =============================================================================
# DOCKER LOG WATCHER (Per-test, autouse)
# =============================================================================


@pytest.fixture(autouse=True)
def docker_log_watcher(
    request: pytest.FixtureRequest,
    containers: dict[str, Container],
    system_config: SystemTestConfig,
) -> Generator[DockerLogWatcher, None, None]:
    """
    Intercept Docker logs before and after each test.

    This fixture:
    1. Marks the current log position before the test
    2. Allows the test to run
    3. Captures all new logs after the test
    4. Scans for errors/warnings
    5. Fails the test if any issues found

    CRITICAL: This runs for EVERY test automatically (autouse=True)
    """
    watcher = DockerLogWatcher(containers, system_config)

    # === BEFORE TEST ===
    watcher.mark_positions()
    test_start = datetime.now(UTC)
    test_name = request.node.name
    test_id = request.node.nodeid

    yield watcher

    # === AFTER TEST ===
    # Give async operations time to complete (Celery tasks, etc.)
    time.sleep(0.3)

    # Capture all logs since the mark
    captures = watcher.capture_since_mark()

    # Check for issues
    issues = watcher.get_issues_summary(
        captures,
        strict_mode=system_config.strict_mode,
    )

    if issues:
        # Generate AI-friendly debug report
        report = AIDebugReport(
            test_id=test_id,
            test_name=test_name,
            test_start=test_start,
            test_end=datetime.now(UTC),
            container_logs={c.container: c.logs for c in captures},
            errors=[e for c in captures for e in c.errors],
            tracebacks=[t for c in captures for t in c.tracebacks],
            warnings=[w for c in captures for w in c.warnings],
        )

        # Save report
        report_path = report.save()

        # Fail with detailed message
        pytest.fail(
            f"\n{'=' * 70}\n"
            f"🚨 SYSTEM HEALTH FAILURE - Test triggered errors in Docker stack\n"
            f"{'=' * 70}\n"
            f"Test: {test_name}\n"
            f"Debug Report: {report_path}\n"
            f"\n{issues}\n"
            f"{'=' * 70}"
        )


# =============================================================================
# CELERY TASK SYNCHRONIZATION
# =============================================================================


@pytest.fixture
def celery_tracker(
    containers: dict[str, Container],
    system_config: SystemTestConfig,
) -> CeleryTaskTracker:
    """
    Track Celery task execution to ensure async work completes.

    Usage in tests:
        def test_something(celery_tracker):
            # Trigger action that queues Celery task
            response = client.post("/api/cronjobs/prices_daily/run")

            # Wait for all tasks to complete
            celery_tracker.wait_for_drain(timeout=30)

            # Now verify results (task has completed)
    """
    # Try worker, fall back to worker-batch
    worker = containers.get("worker") or containers.get("worker-batch")
    if not worker:
        pytest.skip("No Celery worker container available")

    return CeleryTaskTracker(
        worker_container=worker,
        timeout=system_config.celery_drain_timeout,
    )


# =============================================================================
# DATABASE STATE VERIFICATION
# =============================================================================


@pytest.fixture
def db_snapshot(containers: dict[str, Container]) -> DatabaseSnapshot:
    """
    Capture database state for before/after comparison.

    Usage:
        def test_voting(db_snapshot):
            before = db_snapshot.query_scalar("SELECT COUNT(*) FROM community_votes")

            # Perform action
            response = client.post("/api/swipe/vote", ...)

            after = db_snapshot.query_scalar("SELECT COUNT(*) FROM community_votes")
            assert after > before
    """
    postgres = containers.get("postgres")
    if not postgres:
        pytest.skip("No PostgreSQL container available")

    return DatabaseSnapshot(postgres)


# =============================================================================
# HTTP CLIENT
# =============================================================================


@pytest.fixture
def api_client(system_config: SystemTestConfig) -> Generator[httpx.Client, None, None]:
    """
    HTTP client configured for the API.

    Usage:
        def test_endpoint(api_client):
            response = api_client.get("/health")
            assert response.status_code == 200
    """
    with httpx.Client(
        base_url=system_config.api_base_url,
        timeout=system_config.api_timeout,
    ) as client:
        yield client


@pytest.fixture
def auth_client(
    api_client: httpx.Client,
    system_config: SystemTestConfig,
) -> httpx.Client:
    """
    HTTP client with authentication token.

    Creates a test user and returns authenticated client.
    """
    # Try to get token from environment first
    token = os.getenv("SYSTEM_TEST_TOKEN")

    if not token:
        # Create test user and get token
        # This assumes a test user creation endpoint exists
        try:
            response = api_client.post(
                "/api/auth/register",
                json={
                    "email": "system_test@test.local",
                    "password": "TestPassword123!",
                    "name": "System Test User",
                },
            )
            if response.status_code in (200, 201):
                token = response.json().get("access_token")
            elif response.status_code == 409:
                # User exists, try login
                response = api_client.post(
                    "/api/auth/login",
                    json={
                        "email": "system_test@test.local",
                        "password": "TestPassword123!",
                    },
                )
                token = response.json().get("access_token")
        except Exception:
            pytest.skip("Could not authenticate for system tests")

    if not token:
        pytest.skip("No authentication token available")

    # Set auth header
    api_client.headers["Authorization"] = f"Bearer {token}"
    return api_client


# =============================================================================
# PYTEST HOOKS FOR ENHANCED REPORTING
# =============================================================================


def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers",
        "workflow: End-to-end workflow test that mixes UI/API/DB verification",
    )
    config.addinivalue_line(
        "markers",
        "smoke: Quick sanity check for critical functionality",
    )
    config.addinivalue_line(
        "markers",
        "slow: Test that takes more than 10 seconds",
    )


def pytest_collection_modifyitems(config, items):
    """Add markers based on test location."""
    for item in items:
        # Add smoke marker to tests in smoke/ directory
        if "/smoke/" in str(item.fspath):
            item.add_marker(pytest.mark.smoke)

        # Add workflow marker to tests in workflows/ directory
        if "/workflows/" in str(item.fspath):
            item.add_marker(pytest.mark.workflow)
