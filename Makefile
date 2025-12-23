# ============================================================================
# stonkmarket Makefile
# ============================================================================
.PHONY: help install install-dev lint format typecheck test test-cov test-unit test-integration e2e clean

PYTHON := python3
PIP := pip
PYTEST := pytest
RUFF := ruff
MYPY := mypy

# Colors for output
BLUE := \033[0;34m
GREEN := \033[0;32m
YELLOW := \033[0;33m
NC := \033[0m # No Color

help: ## Show this help message
	@echo "$(BLUE)stonkmarket Development Commands$(NC)"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  $(GREEN)%-18s$(NC) %s\n", $$1, $$2}'

# ============================================================================
# Installation
# ============================================================================
install: ## Install production dependencies
	$(PIP) install -r requirements.txt

install-dev: install ## Install development dependencies
	$(PIP) install -r requirements-dev.txt
	playwright install chromium

# ============================================================================
# Code Quality
# ============================================================================
lint: ## Run linter (ruff)
	$(RUFF) check app/ tests/

lint-fix: ## Run linter and auto-fix issues
	$(RUFF) check --fix app/ tests/

format: ## Format code with ruff
	$(RUFF) format app/ tests/

format-check: ## Check code formatting without changes
	$(RUFF) format --check app/ tests/

typecheck: ## Run type checker (mypy)
	$(MYPY) app/

quality: lint format-check typecheck ## Run all code quality checks

# ============================================================================
# Testing
# ============================================================================
test: ## Run all tests
	$(PYTEST) tests/ -v

test-cov: ## Run tests with coverage report
	$(PYTEST) tests/ --cov=app --cov-report=term-missing --cov-report=html

test-unit: ## Run unit tests only
	$(PYTEST) tests/ -m "unit or not integration" -v

test-integration: ## Run integration tests only
	$(PYTEST) tests/ -m integration -v

test-fast: ## Run tests in parallel (faster)
	$(PYTEST) tests/ -n auto -v

test-watch: ## Run tests and watch for changes
	$(PYTEST) tests/ --watch

# ============================================================================
# E2E Testing (Playwright)
# ============================================================================
e2e: ## Run Playwright E2E tests
	cd frontend && npx playwright test

e2e-ui: ## Run Playwright tests with UI
	cd frontend && npx playwright test --ui

e2e-headed: ## Run Playwright tests in headed mode
	cd frontend && npx playwright test --headed

e2e-report: ## Show last Playwright test report
	cd frontend && npx playwright show-report

# ============================================================================
# Development
# ============================================================================
dev: ## Start development server
	docker compose -f docker-compose.dev.yml up -d

dev-logs: ## Show development logs
	docker compose -f docker-compose.dev.yml logs -f api

dev-down: ## Stop development server
	docker compose -f docker-compose.dev.yml down

# ============================================================================
# CI
# ============================================================================
ci: quality test ## Run all CI checks (quality + tests)

ci-full: quality test-cov e2e ## Run full CI pipeline with E2E

# ============================================================================
# Cleanup
# ============================================================================
clean: ## Clean up generated files
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".ruff_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "htmlcov" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name ".coverage" -delete 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
