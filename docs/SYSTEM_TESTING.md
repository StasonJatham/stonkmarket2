# Holistic System Verification Framework

A comprehensive testing framework that validates the **entire Docker Compose stack**, not just individual endpoints. If any container emits an error during a test, that test fails.

## Philosophy

Traditional tests mock databases and external services, which can hide real integration issues. This framework:

1. **Runs against real containers** - No mocks for DB, Celery, or Redis
2. **Monitors all container logs** - Fails if any container emits ERROR/WARNING
3. **Waits for async operations** - Ensures Celery tasks complete before assertions
4. **Generates AI-friendly reports** - Structured JSON reports for automated debugging

## Quick Start

### Using the Dev Stack (Recommended)

If you already have the dev stack running:

```bash
# Run all system tests
SYSTEM_TEST_USE_DEV=1 python -m pytest system_tests/ -v

# Run only smoke tests (fast)
SYSTEM_TEST_USE_DEV=1 python -m pytest system_tests/smoke/ -v

# Run with lenient mode (don't fail on warnings)
SYSTEM_TEST_USE_DEV=1 SYSTEM_TEST_STRICT=0 python -m pytest system_tests/ -v
```

### Using the Isolated Test Stack

```bash
# Start the test stack
docker compose -f docker-compose.test.yml up -d

# Run migrations
docker compose -f docker-compose.test.yml exec api alembic upgrade head

# Run tests
python -m pytest system_tests/ -v

# Cleanup
docker compose -f docker-compose.test.yml down -v
```

### Using the Runner Script

```bash
# Run all system tests using dev stack
./scripts/run_system_tests.sh --dev

# Run smoke tests only
./scripts/run_system_tests.sh --dev smoke

# Run with strict mode off
./scripts/run_system_tests.sh --dev --lenient
```

## Project Structure

```
system_tests/
├── conftest.py           # Main fixtures: Docker log watcher, Celery sync
├── config.py             # Configuration: container names, patterns
│
├── fixtures/
│   ├── docker_logs.py    # DockerLogWatcher - monitors container logs
│   ├── celery_sync.py    # CeleryTaskTracker - waits for task completion
│   ├── db_state.py       # DatabaseSnapshot - before/after state verification
│   └── stack_health.py   # StackHealthChecker - verifies all containers healthy
│
├── reporters/
│   ├── ai_debug_report.py # Structured JSON failure reports
│   └── log_extractor.py   # Extract error context from logs
│
├── smoke/
│   └── test_stack_healthy.py  # Fast sanity checks
│
└── workflows/
    ├── test_price_fetch_flow.py  # End-to-end price data tests
    ├── test_dipfinder_flow.py    # DipFinder job tests
    └── test_voting_flow.py       # Voting workflow tests
```

## Key Features

### 1. Docker Log Watcher (Auto-enabled)

Every test automatically monitors all container logs. If any container emits an error during the test, the test fails:

```python
def test_something(api_client):
    # This test will FAIL if any container logs an error
    response = api_client.post("/api/cronjobs/dipfinder/run")
    # Even if response.status_code == 200, we fail if Celery worker crashed
```

### 2. Celery Task Synchronization

Wait for background tasks to complete before making assertions:

```python
def test_job_flow(auth_client, celery_tracker):
    # Trigger job
    response = auth_client.post("/api/cronjobs/prices_daily/run")
    
    # Wait for Celery to finish
    celery_tracker.wait_for_drain(timeout=60)
    
    # Verify no tasks failed
    celery_tracker.assert_no_failed_tasks()
```

### 3. Database State Verification

Capture and compare database state:

```python
def test_vote_persisted(auth_client, db_snapshot):
    before = db_snapshot.query_scalar("SELECT COUNT(*) FROM dip_votes")
    
    auth_client.post("/api/swipe/vote", json={"symbol": "AAPL", "vote": "up"})
    
    after = db_snapshot.query_scalar("SELECT COUNT(*) FROM dip_votes")
    assert after > before
```

### 4. AI-Friendly Debug Reports

On failure, structured JSON reports are saved to `test-results/debug-reports/`:

```json
{
  "test_id": "system_tests/workflows/test_price_fetch_flow.py::test_chart_endpoint",
  "summary": "Exception: KeyError: 'price' | 2 error(s) in logs",
  "tracebacks": ["Traceback (most recent call last):\n  ..."],
  "container_logs": {
    "api": ["2026-01-04 14:30:01 ERROR: ..."],
    "worker": ["2026-01-04 14:30:02 Task failed: ..."]
  }
}
```

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `SYSTEM_TEST_USE_DEV` | `0` | Use dev stack container names |
| `SYSTEM_TEST_STRICT` | `1` | Fail on warnings (not just errors) |
| `CELERY_TIMEOUT` | `30` | Seconds to wait for Celery drain |
| `API_BASE_URL` | `http://localhost:8000` | API base URL |

### Ignored Patterns

Some log patterns are ignored by default (false positives):

- `DEBUG` - Debug logs are fine
- `TzCache` - yfinance timezone cache warnings
- `healthcheck` - Health check logs
- `rate limit` - Expected rate limiting
- `404.*fundamentals` - ETFs don't have fundamentals

Add more in `system_tests/config.py`.

## Writing New Tests

### Smoke Tests

Quick sanity checks in `system_tests/smoke/`:

```python
class TestCriticalEndpoints:
    def test_health(self, api_client):
        response = api_client.get("/api/health")
        assert response.status_code == 200
```

### Workflow Tests

End-to-end flows in `system_tests/workflows/`:

```python
class TestPriceFlow:
    @pytest.mark.slow  # Mark long-running tests
    def test_prices_daily_job(self, auth_client, celery_tracker, db_snapshot):
        # Capture before state
        before_count = db_snapshot.query_scalar("SELECT COUNT(*) FROM price_history")
        
        # Trigger job
        response = auth_client.post("/api/cronjobs/prices_daily/run")
        assert response.status_code == 200
        
        # Wait for completion
        celery_tracker.wait_for_drain(timeout=60)
        celery_tracker.assert_no_failed_tasks()
        
        # Docker log watcher automatically fails if any errors occurred
```

### Available Fixtures

| Fixture | Scope | Description |
|---------|-------|-------------|
| `docker_log_watcher` | function (autouse) | Monitors container logs |
| `api_client` | function | HTTP client for API calls |
| `auth_client` | function | Authenticated HTTP client |
| `db_snapshot` | function | Database state capture |
| `celery_tracker` | function | Celery task synchronization |
| `stack_health` | session | Stack health checker |
| `containers` | session | Docker container references |

## Markers

- `@pytest.mark.smoke` - Quick sanity tests (auto-added for `smoke/` directory)
- `@pytest.mark.workflow` - End-to-end workflow tests (auto-added for `workflows/`)
- `@pytest.mark.slow` - Tests that take >10 seconds

Run specific markers:

```bash
pytest system_tests/ -m smoke      # Only smoke tests
pytest system_tests/ -m "not slow" # Skip slow tests
```

## Troubleshooting

### "Container not found" error

Ensure the Docker stack is running:

```bash
# For dev stack
docker compose -f docker-compose.dev.yml up -d

# For test stack
docker compose -f docker-compose.test.yml up -d
```

### Tests fail due to expected warnings

Use lenient mode or add the pattern to `ignore_patterns` in config:

```bash
SYSTEM_TEST_STRICT=0 pytest system_tests/ -v
```

### Debug reports not generated

Check the `test-results/debug-reports/` directory. Reports are only generated when tests fail due to container log issues.
