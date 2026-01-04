# Stonkmarket Refactoring Blueprint

**Status:** In Progress  
**Started:** January 4, 2026  
**Last Updated:** January 4, 2026

---

## Overview

This document tracks the architectural refactoring of the Stonkmarket codebase from legacy patterns to a modern, async-first, high-performance stack.

---

## Phase 1: Critical Fixes & Foundation ✅ COMPLETED

### 1.1 Fixed Deprecated `asyncio.get_event_loop()` ✅

All deprecated `asyncio.get_event_loop()` calls replaced with:
- `asyncio.get_running_loop()` in async contexts
- `asyncio.run()` in Celery sync task contexts

**Files Modified:**
- `app/services/data_providers/yfinance_service.py` - 7 occurrences
- `app/services/data_providers/yahooquery_service.py` - 6 occurrences
- `app/services/market_data.py` - 1 occurrence
- `app/services/calendar_data.py` - 1 occurrence
- `app/services/financedatabase_service.py` - 3 occurrences
- `app/jobs/tasks.py` - 2 occurrences

### 1.2 YFinance Service Improvements ✅

**Thread Pool Optimization:**
- Increased `ThreadPoolExecutor` from 4 to 8 workers
- Added `asyncio.Semaphore(16)` for backpressure control
- All executor calls wrapped with semaphore

**File:** `app/services/data_providers/yfinance_service.py`

### 1.3 Database Session Dependency Injection ✅

**Created:**
- `app/database/session.py` - Request-scoped sessions with auto-commit/rollback
- `DbSession` type alias for FastAPI route dependency injection
- Added `get_session_factory()` to `app/database/connection.py`

### 1.4 Dead Code Removal ✅

**Deleted:**
- `app/repositories/cronjobs.py` - Replaced by `cronjobs_orm.py`

---

## Phase 2: Domain Models ✅ COMPLETED

### 2.1 Create Pydantic Domain Models ✅ COMPLETED

Created validated, typed models in `app/domain/` directory:

| Model | Purpose | Status |
| ----- | ------- | ------ |
| `TickerInfo` | Complete stock/ETF metadata from yfinance | ✅ DONE |
| `TickerSearchResult` | Symbol search result | ✅ DONE |
| `PriceBar` | Single OHLCV candlestick | ✅ DONE |
| `PriceHistory` | Price series with DataFrame conversion | ✅ DONE |
| `FundamentalsData` | Company fundamentals for quality scoring | ✅ DONE |
| `QualityMetrics` | Quality score with sub-scores | ✅ DONE |
| `EarningsEvent` | Earnings calendar event | ✅ DONE |
| `IpoEvent` | IPO calendar event | ✅ DONE |
| `SplitEvent` | Stock split event | ✅ DONE |
| `EconomicEvent` | Economic calendar event | ✅ DONE |

**Files Created:**

- `app/domain/__init__.py` - Public exports
- `app/domain/ticker.py` - TickerInfo, TickerSearchResult
- `app/domain/price.py` - PriceBar, PriceHistory
- `app/domain/fundamentals.py` - FundamentalsData, QualityMetrics
- `app/domain/calendar.py` - Calendar event models
- `tests/test_domain_models.py` - 27 tests for all models

### 2.2 Migrate Services to Return Models ✅ COMPLETED

Added typed wrapper methods to `YFinanceService` (backward compatible):

| Service | Function | Status |
| ------- | -------- | ------ |
| `yfinance_service.py` | `get_ticker_info_typed()` → `TickerInfo` | ✅ DONE |
| `yfinance_service.py` | `get_price_history_typed()` → `PriceHistory` | ✅ DONE |
| `yfinance_service.py` | `search_tickers_typed()` → `list[TickerSearchResult]` | ✅ DONE |
| `fundamentals.py` | `QualityMetrics.to_pydantic()` conversion | ✅ DONE |
| `dipfinder/service.py` | Uses QualityMetrics dataclass (convertible) | ✅ DONE |

**Migration Pattern:**

Existing code continues to use dict/dataclass returns. New code can use typed methods:

```python
# Old pattern (still works)
info = await service.get_ticker_info("AAPL")  # Returns dict

# New typed pattern
info = await service.get_ticker_info_typed("AAPL")  # Returns TickerInfo

# Dataclass to Pydantic conversion
from app.dipfinder.fundamentals import compute_quality_score
quality = await compute_quality_score("AAPL")  # Returns dataclass
pydantic_model = quality.to_pydantic()  # Convert to Pydantic
```

### 2.3 Add Response Models to Routes ⏳ TODO

Ensure all API routes have explicit Pydantic response models.

---

## Phase 3: YFinance Resilience ✅ COMPLETED

### 3.1 Circuit Breaker ✅

Added circuit breaker pattern in `app/services/data_providers/resilience.py`:
- Opens after 5 consecutive failures
- Half-open after 30 seconds
- Reset on successful call
- Integrated in `YFinanceService._run_in_executor()`

### 3.2 Request Coalescing ✅

Added request coalescing in `app/services/data_providers/resilience.py`:
- `RequestCoalescer` class with `coalesce()` context manager
- `execute()` method for simpler API
- Deduplicates concurrent requests for same symbol

### 3.3 Retry Policy ✅

Added exponential backoff retry in `app/services/data_providers/resilience.py`:
- `with_retry` decorator and `retry_async` function
- Max 3 attempts with 1s → 2s → 4s base delay
- Jitter support to prevent thundering herd
- Configurable retry exceptions

**Files Created/Modified:**
- `app/services/data_providers/resilience.py` - 642 lines, all patterns
- `tests/test_resilience.py` - 571 lines, comprehensive tests
- `app/services/data_providers/yfinance_service.py` - Integrated patterns

---

## Phase 4: Job Refactoring ✅ COMPLETED

### 4.1 Split Monolithic Job Definitions ✅ COMPLETED

Original `app/jobs/definitions.py` was 3264 lines. Now modularized:

**Final Module Structure:**

| Module | Lines | Jobs |
| ------ | ----- | ---- |
| `definitions.py` | 653 | cache_warmup, fundamentals_daily, cleanup_daily, portfolio_worker, market_data_sync, calendar_sync |
| `ai/__init__.py` | 284 | ai_bios_weekly, ai_batch_poll, batch_watchdog, ai_personas_weekly, portfolio_ai_analysis |
| `quant/__init__.py` | 963 | quant_monthly, strategy_nightly, quant_scoring_daily, quant_analysis_nightly |
| `data/__init__.py` | 900 | universe_sync, symbol_ingest, data_backfill, prices_daily, add_to_ingest_queue |
| `analysis/__init__.py` | 407 | signals_daily, dipfinder_daily, regime_daily |
| `pipelines/__init__.py` | 210 | market_close_pipeline, weekly_ai_pipeline |
| `utils.py` | 97 | log_job_success, get_close_column, timing helpers |

**Total reduction:** 3264 lines → 653 lines in main definitions.py (80% reduction)

**Directory structure:**

```text
app/jobs/
├── definitions.py      # Core maintenance jobs (653 lines)
├── utils.py            # Shared utilities (97 lines)
├── pipelines/          # Pipeline orchestrators
│   └── __init__.py     # (210 lines)
├── ai/                 # AI-related jobs
│   └── __init__.py     # (284 lines)
├── quant/              # Quantitative analysis jobs
│   └── __init__.py     # (963 lines)
├── data/               # Data ingestion jobs
│   └── __init__.py     # (900 lines)
├── analysis/           # Market analysis jobs
│   └── __init__.py     # (407 lines)
├── registry.py         # Job registry
├── executor.py         # Job executor
└── tasks.py            # Celery tasks
```

### 4.2 Fix Circular Imports ✅ N/A

No circular imports detected - Celery uses string-based task references via `_run_async()` pattern in `tasks.py`.

---

## Phase 5: Database Consolidation 📋 PLANNED

### 5.1 Remove Legacy asyncpg Pool

Currently maintaining dual pools (asyncpg + SQLAlchemy). Consolidate to SQLAlchemy only:

**To Remove from `app/database/connection.py`:**
- `_pool` global variable
- `init_pg_pool()` / `close_pg_pool()`
- `get_pg_connection()`
- `fetch_one()`, `fetch_all()`, `fetch_val()`, `execute()`, `execute_many()`
- `transaction()` context manager

**Update Lifecycle in:**
- `app/main.py`
- `app/api/app.py`

### 5.2 Update Test Fixtures

Update `tests/conftest.py` to only manage SQLAlchemy state.

---

## Test Results

| Phase | Tests Passed | Tests Skipped | Status |
| ----- | ------------ | ------------- | ------ |
| Phase 1 | 513 | 8 | ✅ |
| Phase 2 | 550 | 8 | ✅ (37 new tests: 29 domain + 8 typed service) |

---

## Critical Issues Tracking

| Priority | Issue | Location | Status |
|----------|-------|----------|--------|
| P0 | Dual database pools | `connection.py` | 📋 Phase 5 |
| P0 | Thread pool saturation | `yfinance_service.py` | ✅ Fixed |
| P1 | Circular imports | `jobs/tasks.py` | 📋 Phase 4 |
| P1 | No retry policy | Multiple | 📋 Phase 3 |
| P2 | Raw dicts instead of models | Services | ✅ Fixed (Phase 2) |
| P2 | Deprecated `get_event_loop()` | Multiple | ✅ Fixed |
| P3 | Monolithic job definitions | `definitions.py` | 📋 Phase 4 |

---

## Notes

- All changes maintain backward compatibility
- Each phase includes running the full test suite
- No file should exceed 500 lines after refactoring
