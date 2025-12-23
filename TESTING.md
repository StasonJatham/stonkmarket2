# Testing & Quality Guide

This document covers the testing infrastructure, code quality tools, and best practices for the StonkMarket project.

## Table of Contents

- [Quick Start](#quick-start)
- [Backend Testing (pytest)](#backend-testing-pytest)
- [Frontend E2E Testing (Playwright)](#frontend-e2e-testing-playwright)
- [Code Quality](#code-quality)
- [Web Vitals Monitoring](#web-vitals-monitoring)
- [CI/CD Pipeline](#cicd-pipeline)
- [Writing Tests](#writing-tests)

---

## Quick Start

### Prerequisites

```bash
# Install dev dependencies
pip install -r requirements-dev.txt

# Install Playwright browsers (for E2E)
cd new && npm ci && npx playwright install
```

### Run All Checks

```bash
make ci          # Run full CI suite locally
```

Or run individual checks:

```bash
make lint        # Lint Python code
make format      # Format Python code
make typecheck   # Type check with mypy
make test        # Run pytest
make e2e         # Run Playwright E2E tests
```

---

## Backend Testing (pytest)

### Test Structure

```
tests/
├── conftest.py           # Shared fixtures
├── test_auth.py          # Authentication endpoints
├── test_dips.py          # Dip ranking/charts
├── test_symbols.py       # Symbol management
├── test_health.py        # Health checks
├── test_suggestions.py   # Search suggestions
├── test_cronjobs.py      # Cron job management
├── test_stock_tinder.py  # Tinder card voting
├── test_mfa.py           # Multi-factor auth
└── test_api_keys.py      # API key management
```

### Running Tests

```bash
# Run all tests
make test

# Run with coverage report
make test-cov

# Run specific test file
pytest tests/test_auth.py -v

# Run specific test function
pytest tests/test_auth.py::test_login_success -v

# Run tests matching a pattern
pytest -k "login" -v

# Run only unit tests (fast, no external deps)
make test-unit

# Run only integration tests
make test-integration

# Run tests in parallel
pytest tests/ -n auto
```

### Test Markers

Tests are organized with pytest markers:

```python
@pytest.mark.unit          # Fast, isolated unit tests
@pytest.mark.integration   # Tests requiring database/cache
@pytest.mark.slow          # Long-running tests
@pytest.mark.auth          # Authentication tests
```

Run specific markers:

```bash
pytest -m "unit and not slow"
pytest -m "auth"
```

### Key Fixtures

Located in `tests/conftest.py`:

| Fixture | Description |
|---------|-------------|
| `client` | Sync `TestClient` for FastAPI |
| `async_client` | Async `AsyncClient` for async tests |
| `auth_token` | Valid JWT token for authenticated requests |
| `admin_token` | JWT token with admin privileges |
| `mock_db` | Mocked database session |
| `mock_cache` | Mocked Valkey cache |
| `sample_stock_info` | Sample stock data for testing |
| `sample_ranking_entry` | Sample ranking data |

### Example Test

```python
import pytest
from httpx import AsyncClient

@pytest.mark.integration
async def test_get_ranking(async_client: AsyncClient, auth_token: str):
    """Test fetching dip ranking with authentication."""
    response = await async_client.get(
        "/dips/ranking",
        headers={"Authorization": f"Bearer {auth_token}"}
    )
    assert response.status_code == 200
    data = response.json()
    assert "items" in data
```

---

## Frontend E2E Testing (Playwright)

### Test Structure

```
new/
├── playwright.config.ts   # Playwright configuration
└── e2e/
    ├── dashboard.spec.ts   # Dashboard page tests
    ├── auth.spec.ts        # Authentication flow tests
    └── accessibility.spec.ts # A11y compliance tests
```

### Running E2E Tests

```bash
cd new

# Run all E2E tests (headless)
npx playwright test

# Run with UI mode (interactive)
npx playwright test --ui

# Run specific test file
npx playwright test e2e/dashboard.spec.ts

# Run in specific browser
npx playwright test --project=chromium
npx playwright test --project=firefox
npx playwright test --project=webkit

# Run mobile viewport tests
npx playwright test --project="Mobile Chrome"

# Show HTML report
npx playwright show-report
```

Or use Make targets:

```bash
make e2e         # Run headless
make e2e-ui      # Run with UI mode
```

### Test Browsers

Configured in `playwright.config.ts`:

- **Chromium** (Desktop Chrome)
- **Firefox** (Desktop Firefox)
- **WebKit** (Desktop Safari)
- **Mobile Chrome** (Pixel 5 viewport)
- **Mobile Safari** (iPhone 12 viewport)

### Example E2E Test

```typescript
import { test, expect } from '@playwright/test';

test.describe('Dashboard', () => {
  test('displays stock cards', async ({ page }) => {
    await page.goto('/');
    
    // Wait for data to load
    await expect(page.getByTestId('stock-card')).toBeVisible();
    
    // Verify card content
    const cards = page.locator('[data-testid="stock-card"]');
    await expect(cards).toHaveCount.greaterThan(0);
  });
});
```

### Page Object Pattern

For complex tests, use page objects:

```typescript
// e2e/pages/DashboardPage.ts
export class DashboardPage {
  constructor(private page: Page) {}
  
  async goto() {
    await this.page.goto('/');
  }
  
  async getStockCards() {
    return this.page.locator('[data-testid="stock-card"]');
  }
}
```

---

## Code Quality

### Linting (ruff)

ruff provides fast Python linting with auto-fix capabilities.

```bash
# Check for issues
make lint

# Auto-fix issues
make lint-fix

# Check specific file
ruff check app/api/routes/dips.py
```

**Configured rules** (see `pyproject.toml`):

- `E`, `W` - pycodestyle errors/warnings
- `F` - Pyflakes
- `I` - isort (import sorting)
- `B` - flake8-bugbear
- `C4` - flake8-comprehensions
- `UP` - pyupgrade
- `ARG` - flake8-unused-arguments
- `SIM` - flake8-simplify

### Formatting (ruff format)

```bash
# Format all Python files
make format

# Check formatting without changes
ruff format --check .
```

### Type Checking (mypy)

```bash
# Run mypy
make typecheck

# Check specific module
mypy app/services/ --ignore-missing-imports
```

**Type checking configuration** (see `pyproject.toml`):

- Strict mode enabled
- Missing imports ignored (for third-party libs)
- Implicit Optional disabled

---

## Web Vitals Monitoring

The application includes built-in Web Vitals measurement for performance monitoring.

### Metrics Collected

| Metric | Description | Good | Needs Improvement | Poor |
|--------|-------------|------|-------------------|------|
| **LCP** | Largest Contentful Paint | ≤2500ms | ≤4000ms | >4000ms |
| **INP** | Interaction to Next Paint | ≤200ms | ≤500ms | >500ms |
| **CLS** | Cumulative Layout Shift | ≤0.1 | ≤0.25 | >0.25 |
| **FCP** | First Contentful Paint | ≤1800ms | ≤3000ms | >3000ms |
| **TTFB** | Time to First Byte | ≤800ms | ≤1800ms | >1800ms |

### How It Works

1. **Frontend** (`src/lib/webVitals.ts`):
   - Uses PerformanceObserver API to collect metrics
   - Sends data on page visibility change

2. **Backend** (`/metrics/vitals`):
   - Receives and aggregates metrics
   - Provides summary statistics (p50, p75, p95)

### API Endpoints

```bash
# Get aggregated vitals summary
GET /metrics/vitals/summary

# Response:
{
  "LCP": {
    "count": 150,
    "p50": 1850.5,
    "p75": 2100.0,
    "p95": 2800.0,
    "avg": 1920.3,
    "good_pct": 85.3,
    "poor_pct": 2.0
  },
  ...
}

# Clear metrics buffer (admin only)
DELETE /metrics/vitals
```

### Viewing Metrics

The Web Vitals summary can be accessed at `/metrics/vitals/summary` or through the admin dashboard.

---

## CI/CD Pipeline

### GitHub Actions Workflow

The CI pipeline (`.github/workflows/ci.yml`) runs on every push and PR:

```
┌─────────────────────────────────────────────────────────────┐
│                         CI Pipeline                          │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│   ┌─────────┐   ┌───────────┐   ┌──────────────────┐       │
│   │  Lint   │   │ TypeCheck │   │  Frontend Lint   │       │
│   └────┬────┘   └─────┬─────┘   └────────┬─────────┘       │
│        │              │                   │                  │
│        └──────────────┼───────────────────┘                  │
│                       ▼                                      │
│               ┌───────────────┐                              │
│               │    Pytest     │                              │
│               │  (with Postgres, Valkey)                     │
│               └───────┬───────┘                              │
│                       │                                      │
│                       ▼                                      │
│               ┌───────────────┐                              │
│               │  Playwright   │                              │
│               │    E2E        │                              │
│               └───────┬───────┘                              │
│                       │                                      │
│                       ▼                                      │
│               ┌───────────────┐   ┌───────────────┐         │
│               │    Build      │   │   Security    │         │
│               │    Check      │   │    Scan       │         │
│               └───────────────┘   └───────────────┘         │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### Jobs

| Job | Description | Dependencies |
|-----|-------------|--------------|
| `lint` | ruff linter + format check | - |
| `typecheck` | mypy type checking | - |
| `test` | pytest with coverage | - |
| `frontend-lint` | ESLint + TypeScript | - |
| `e2e` | Playwright browser tests | lint, test, frontend-lint |
| `security` | pip-audit vulnerability scan | - |
| `build` | Build frontend + Docker | lint, typecheck, test, frontend-lint |

### Running CI Locally

```bash
# Run full CI suite
make ci

# This runs:
# 1. ruff check
# 2. ruff format --check
# 3. mypy
# 4. pytest with coverage
```

---

## Writing Tests

### Best Practices

1. **Use descriptive test names**
   ```python
   def test_login_with_invalid_password_returns_401():
       ...
   ```

2. **One assertion per test** (when practical)
   ```python
   def test_ranking_returns_items():
       ...  # Only check that items are returned
   
   def test_ranking_items_have_required_fields():
       ...  # Only check field presence
   ```

3. **Use fixtures for setup**
   ```python
   @pytest.fixture
   def sample_user(db):
       return create_test_user(db)
   ```

4. **Mock external services**
   ```python
   @pytest.fixture
   def mock_stock_api(mocker):
       return mocker.patch("app.services.stock_info.fetch_price")
   ```

5. **Test edge cases**
   - Empty inputs
   - Invalid data
   - Missing permissions
   - Rate limiting

### Test Coverage Goals

| Module | Target Coverage |
|--------|-----------------|
| `app/api/routes/` | ≥80% |
| `app/services/` | ≥85% |
| `app/repositories/` | ≥75% |
| `app/core/` | ≥70% |

View coverage report:

```bash
make test-cov
# Opens htmlcov/index.html
```

---

## Troubleshooting

### Common Issues

**1. Tests fail with database connection error**
```bash
# Ensure services are running
docker compose up -d postgres valkey
```

**2. Playwright tests timeout**
```bash
# Increase timeout in playwright.config.ts
# Or run in headed mode to debug
npx playwright test --headed
```

**3. Import errors in tests**
```bash
# Ensure you're in the project root
# and PYTHONPATH is set
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

**4. mypy reports missing stubs**
```bash
# Install type stubs
pip install types-redis types-python-dateutil
```

---

## Resources

- [pytest documentation](https://docs.pytest.org/)
- [Playwright documentation](https://playwright.dev/)
- [ruff documentation](https://docs.astral.sh/ruff/)
- [Web Vitals](https://web.dev/vitals/)
- [mypy documentation](https://mypy.readthedocs.io/)
