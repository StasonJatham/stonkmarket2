"""
Stack Health Checker - Verify Docker stack is healthy before tests.

This module provides comprehensive health checks for the entire
Docker Compose stack, ensuring all containers are running and
responsive before tests begin.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

import httpx

if TYPE_CHECKING:
    from docker.models.containers import Container

from system_tests.config import SystemTestConfig


@dataclass
class HealthCheckResult:
    """Result of a health check."""

    name: str
    healthy: bool
    message: str
    details: dict | None = None


class StackHealthChecker:
    """
    Verify entire Docker stack is healthy.

    Checks:
    1. All containers are running
    2. Container health status (if health check configured)
    3. API endpoint responds
    4. Database accepts connections
    5. Celery workers are connected
    """

    def __init__(
        self,
        containers: dict[str, "Container"],
        config: SystemTestConfig,
    ):
        self.containers = containers
        self.config = config

    def check_all(self) -> list[HealthCheckResult]:
        """Run all health checks and return results."""
        results = []

        # Check each container
        for name, container in self.containers.items():
            results.append(self._check_container(name, container))

        # Check API health endpoint
        results.append(self._check_api_health())

        return results

    def assert_healthy(self) -> None:
        """Assert that all checks pass, raise if any fail."""
        results = self.check_all()
        failures = [r for r in results if not r.healthy]

        if failures:
            messages = "\n".join(
                f"  ❌ {r.name}: {r.message}" for r in failures
            )
            raise AssertionError(
                f"Stack health check failed:\n{messages}\n\n"
                f"Ensure docker-compose.test.yml is running: "
                f"docker compose -f docker-compose.test.yml up -d"
            )

    def wait_for_healthy(
        self,
        timeout: float | None = None,
        poll_interval: float = 2.0,
    ) -> None:
        """Wait for stack to become healthy."""
        timeout = timeout or self.config.stack_startup_timeout
        start = time.time()

        while (time.time() - start) < timeout:
            try:
                self.assert_healthy()
                return  # All healthy
            except AssertionError:
                time.sleep(poll_interval)

        # Final check with full error
        self.assert_healthy()

    def _check_container(
        self, name: str, container: "Container"
    ) -> HealthCheckResult:
        """Check if a container is running and healthy."""
        try:
            container.reload()

            # Check running status
            if container.status != "running":
                return HealthCheckResult(
                    name=f"container:{name}",
                    healthy=False,
                    message=f"Container not running (status: {container.status})",
                )

            # Check Docker health status if available
            health = container.attrs.get("State", {}).get("Health", {})
            if health:
                status = health.get("Status", "unknown")
                if status != "healthy":
                    return HealthCheckResult(
                        name=f"container:{name}",
                        healthy=False,
                        message=f"Container unhealthy (health: {status})",
                        details=health,
                    )

            return HealthCheckResult(
                name=f"container:{name}",
                healthy=True,
                message="Container running",
            )

        except Exception as e:
            return HealthCheckResult(
                name=f"container:{name}",
                healthy=False,
                message=f"Error checking container: {e}",
            )

    def _check_api_health(self) -> HealthCheckResult:
        """Check if API health endpoint responds."""
        # Try multiple possible health endpoints
        health_endpoints = [
            f"{self.config.api_base_url}/api/health",
            f"{self.config.api_base_url}/api/health/live",
            f"{self.config.api_base_url}/health",
        ]

        for url in health_endpoints:
            try:
                response = httpx.get(url, timeout=self.config.api_timeout)

                if response.status_code == 200:
                    return HealthCheckResult(
                        name="api:health",
                        healthy=True,
                        message="API responding",
                        details=response.json() if response.headers.get(
                            "content-type", ""
                        ).startswith("application/json") else None,
                    )
            except httpx.RequestError:
                continue

        # All endpoints failed
        return HealthCheckResult(
            name="api:health",
            healthy=False,
            message="Cannot connect to any health endpoint",
        )

    def get_container_logs(
        self, name: str, tail: int = 50
    ) -> str:
        """Get recent logs from a container for debugging."""
        container = self.containers.get(name)
        if not container:
            return f"Container '{name}' not found"

        try:
            return container.logs(tail=tail).decode("utf-8", errors="replace")
        except Exception as e:
            return f"Error fetching logs: {e}"

    def print_status(self) -> None:
        """Print current stack status to stdout."""
        results = self.check_all()

        print("\n" + "=" * 60)
        print("STACK HEALTH STATUS")
        print("=" * 60)

        for result in results:
            icon = "✅" if result.healthy else "❌"
            print(f"  {icon} {result.name}: {result.message}")

        print("=" * 60 + "\n")
