# Static Code Analysis Report

**Generated:** 2025-12-26
**Python Version:** 3.14.1
**Tools Used:** ruff, bandit, mypy, vulture, radon

---

## Executive Summary

| Tool | Issues Found | Severity |
|------|--------------|----------|
| Ruff (Linting) | 3,996 | Mostly auto-fixable |
| Bandit (Security) | 23 | ~~3 High~~, 2 Medium, 21 Low |
| mypy (Type Checking) | 248 | Type errors |
| Vulture (Dead Code) | 4 | False positives (context manager args) |
| Radon (Complexity) | 15 high-complexity | Functions need refactoring |

### Overall Code Quality Score: **B+**

The codebase is functional with comprehensive test coverage (406 tests passing), but has room for improvement in type safety, complexity reduction, and security hardening.

### Fixes Applied (2025-12-26)
- ✅ Fixed 3 high-severity MD5 hash issues (added `usedforsecurity=False`)
- ✅ Fixed unreachable code in `batch_scheduler.py`
- ✅ Removed 10 unused imports across 6 files
- ✅ Fixed unused variable warning in `jobs/__init__.py`

---

## 1. Ruff Linting Analysis

**Total Issues:** 3,996 (most auto-fixable with `ruff check --fix`)

### Top Issues by Count

| Issue | Count | Description |
|-------|-------|-------------|
| W293 | 2,149 | Blank lines contain whitespace |
| UP045 | 1,073 | Use `X \| None` instead of `Optional[X]` |
| UP017 | 141 | Use `datetime.UTC` instead of `timezone.utc` |
| I001 | 127 | Import blocks unsorted |
| UP006 | 127 | Use `dict`/`list` instead of `Dict`/`List` |
| UP037 | 40 | Remove unnecessary quotes from annotations |
| UP035 | 28 | Import from `collections.abc` instead |
| W291 | 26 | Trailing whitespace |
| RUF022 | 12 | `__all__` not sorted |
| E712 | 12 | Avoid `== True`, use `is` |

### Recommended Actions
```bash
# Auto-fix most issues
ruff check app/ --fix

# Auto-fix with unsafe changes (review carefully)
ruff check app/ --fix --unsafe-fixes
```

### Priority Fixes (Manual)
- **E402**: 2 module-level imports not at top of file
- **ERA001**: 11 commented-out code blocks to review/remove
- **SIM102**: Nested `if` statements that can be collapsed

---

## 2. Bandit Security Analysis

**Total Issues:** 26 (3 High, 2 Medium, 21 Low)

### High Severity (3)

#### B324: Weak MD5 Hash (3 occurrences)
| File | Line | Context |
|------|------|---------|
| `app/cache/cache.py` | 278 | Cache key generation |
| `app/cache/http_cache.py` | 52 | ETag generation |
| `app/services/logo_service.py` | 167 | Avatar color generation |

**Fix:** Add `usedforsecurity=False` parameter:
```python
# Before
hashlib.md5(content.encode()).hexdigest()

# After (for non-security uses)
hashlib.md5(content.encode(), usedforsecurity=False).hexdigest()
```

### Medium Severity (2)

#### B104: Binding to All Interfaces
- **File:** `app/main.py:178`
- **Context:** `host="0.0.0.0"` in uvicorn config
- **Action:** Expected for containerized deployment, add `# nosec B104` comment

#### B608: SQL Injection Vector
- **File:** `app/services/symbol_search.py:136`
- **Context:** Dynamic SQL query construction
- **Action:** Already uses parameterized queries with `:query` placeholders - false positive

### Low Severity (21)

| Issue | Count | Description |
|-------|-------|-------------|
| B110 | 12 | `try/except/pass` patterns |
| B106 | 2 | False positive on `token_type="bearer"` |
| B101 | 7 | Use of `assert` (disabled in production anyway) |

---

## 3. mypy Type Checking Analysis

**Total Errors:** 248 in 55 files (130 source files checked)

### Critical Patterns

#### 1. SQLAlchemy `rowcount` Attribute (14 occurrences)
```python
# Error: "Result[Any]" has no attribute "rowcount"
result = await session.execute(stmt)
return result.rowcount  # ❌

# Fix: Use CursorResult instead
from sqlalchemy.engine import CursorResult
result: CursorResult = await session.execute(stmt)
return result.rowcount  # ✅
```

**Files affected:**
- `app/repositories/jobs_orm.py` (5 occurrences)
- `app/repositories/cleanup_orm.py` (3 occurrences)
- `app/repositories/dip_history_orm.py` (1 occurrence)
- `app/repositories/portfolios_orm.py` (4 occurrences)
- `app/repositories/suggestions_orm.py` (1 occurrence)

#### 2. Implicit Optional Types (15+ occurrences)
```python
# Error: PEP 484 prohibits implicit Optional
def func(param: int = None):  # ❌

# Fix: Explicit Optional
def func(param: int | None = None):  # ✅
```

#### 3. `no-any-return` Errors (20+ occurrences)
```python
# Error: Returning Any from function declared to return "int"
return result.rowcount  # When rowcount is Any

# Fix: Cast or type narrow
from typing import cast
return cast(int, result.rowcount)
```

#### 4. Type Annotation Mismatches
- `Decimal` vs `float` in `SymbolResponse`
- Missing `task_id` argument in symbol routes
- Incompatible default arguments

### Recommended Actions
1. Add type stubs for `croniter`: `pip install types-croniter`
2. Create `py.typed` marker file for package
3. Add gradual typing with `# type: ignore` where needed
4. Configure `pyproject.toml` with stricter mypy settings

---

## 4. Vulture Dead Code Analysis

**Total Issues:** 14 (high confidence)

### Unused Imports
| File | Import |
|------|--------|
| `app/api/routes/suggestions.py:33` | `_get_ipo_year`, `_get_stock_info`, `_get_stock_info_full` |
| `app/dipfinder/domain_scoring.py:21` | `get_domain_metadata` |
| `app/repositories/dips_orm.py:17` | `joinedload` |
| `app/services/openai_client.py:48` | `TypedDict` |
| `app/services/swipe.py:9` | `joinedload` |
| `app/services/symbol_search.py:42` | `joinedload` |

### Unused Variables
| File | Variable |
|------|----------|
| `app/cache/distributed_lock.py:153` | `exc_tb`, `exc_val` |
| `app/cache/metrics.py:189` | `exc_tb`, `exc_val` |
| `app/jobs/__init__.py:17` | `cron_expression` |

### Unreachable Code
| File | Line | Description |
|------|------|-------------|
| `app/services/batch_scheduler.py` | 377 | Code after `return` statement |

---

## 5. Radon Complexity Analysis

**Average Complexity:** A (4.15) - Excellent
**Total Blocks Analyzed:** 1,275

### High Complexity Functions (Grade D-F)

These functions exceed recommended complexity thresholds and should be refactored:

| File | Function | Grade | Complexity |
|------|----------|-------|------------|
| `app/dipfinder/stability.py` | `_compute_fundamental_stability_score` | F | 54 |
| `app/dipfinder/domain_scoring.py` | `OperatingCompanyAdapter.compute_quality_score` | F | 44 |
| `app/services/data_providers/yfinance_service.py` | `_fetch_ticker_info_sync` | F | 42 |
| `app/hedge_fund/data/yfinance_service.py` | `get_fundamentals` | E | 39 |
| `app/dipfinder/domain_scoring.py` | `RetailAdapter.compute_quality_score` | E | 34 |
| `app/dipfinder/domain_scoring.py` | `SemiconductorAdapter.compute_quality_score` | D | 29 |
| `app/dipfinder/domain_scoring.py` | `BankAdapter.compute_quality_score` | D | 27 |
| `app/hedge_fund/agents/technicals.py` | `_calculate_indicators` | D | 26 |
| `app/hedge_fund/agents/valuation.py` | `_calculate_valuation_grade` | D | 26 |

### Recommended Refactoring
1. **Extract helper methods** - Break large scoring functions into smaller units
2. **Use strategy pattern** - Domain adapters have similar patterns, extract shared logic
3. **Configuration objects** - Replace nested conditionals with config-driven logic
4. **Early returns** - Reduce nesting with guard clauses

---

## 6. Recommended Fixes Priority

### Immediate (Security & Correctness)
1. ✅ Add `usedforsecurity=False` to MD5 calls (non-security uses)
2. ✅ Fix `rowcount` type annotations in repositories
3. ✅ Remove unreachable code in `batch_scheduler.py`

### Short-term (Code Quality)
1. Run `ruff check app/ --fix` to auto-fix 3,000+ issues
2. Remove unused imports identified by vulture
3. Add explicit type annotations where mypy complains

### Medium-term (Maintainability)
1. Refactor high-complexity functions (Grade D-F)
2. Add comprehensive type hints
3. Configure strict mypy in CI pipeline

### Long-term (Technical Debt)
1. Create domain-specific adapter base class
2. Implement proper error handling instead of `try/except/pass`
3. Consider switching to pydantic v2 syntax throughout

---

## 7. CI/CD Integration

Add these commands to your CI pipeline:

```yaml
# .github/workflows/lint.yml
jobs:
  lint:
    steps:
      - name: Ruff Check
        run: ruff check app/ --output-format=github
        
      - name: Bandit Security
        run: bandit -r app/ -f sarif -o bandit.sarif --severity-level=medium
        
      - name: mypy Type Check
        run: mypy app/ --ignore-missing-imports
        
      - name: Complexity Check
        run: radon cc app/ --min=D --total-average --show-complexity
```

---

## 8. Configuration Files

### pyproject.toml additions
```toml
[tool.ruff]
line-length = 100
target-version = "py311"
select = ["E", "F", "W", "I", "UP", "C90"]
ignore = ["E501"]  # Let formatter handle line length

[tool.ruff.per-file-ignores]
"tests/*" = ["S101"]  # Allow asserts in tests

[tool.mypy]
python_version = "3.11"
warn_return_any = true
warn_unused_ignores = true
no_implicit_optional = true

[tool.bandit]
exclude_dirs = ["tests"]
skips = ["B101"]  # Skip assert warnings
```

---

## Appendix: Quick Fix Commands

```bash
# Auto-fix ruff issues
ruff check app/ --fix

# Run bandit with sarif output
bandit -r app/ -f sarif -o bandit.sarif

# Run mypy with HTML report
mypy app/ --html-report mypy_report

# Find all high-complexity functions
radon cc app/ --min=C -s -a

# Remove unused imports automatically
autoflake --remove-all-unused-imports -i -r app/
```
