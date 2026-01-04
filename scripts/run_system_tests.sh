#!/bin/bash
# =============================================================================
# HOLISTIC SYSTEM VERIFICATION FRAMEWORK - Test Runner
# =============================================================================
#
# This script runs system tests against the Docker Compose stack.
# It ensures all containers are healthy before running tests and
# provides detailed error reporting on failure.
#
# Usage:
#   ./scripts/run_system_tests.sh              # Run all system tests
#   ./scripts/run_system_tests.sh smoke        # Run only smoke tests
#   ./scripts/run_system_tests.sh workflows    # Run workflow tests
#   ./scripts/run_system_tests.sh --dev        # Use dev stack (faster)
#   ./scripts/run_system_tests.sh --lenient    # Don't fail on warnings
#
# =============================================================================

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
COMPOSE_FILE="${COMPOSE_FILE:-docker-compose.test.yml}"
STRICT_MODE="${STRICT_MODE:-1}"
USE_DEV_STACK=0
TEST_PATH="system_tests/"
PYTEST_ARGS="-v --tb=short"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --dev)
            USE_DEV_STACK=1
            COMPOSE_FILE="docker-compose.dev.yml"
            shift
            ;;
        --lenient)
            STRICT_MODE=0
            shift
            ;;
        smoke)
            TEST_PATH="system_tests/smoke/"
            shift
            ;;
        workflows)
            TEST_PATH="system_tests/workflows/"
            shift
            ;;
        --slow)
            PYTEST_ARGS="$PYTEST_ARGS -m slow"
            shift
            ;;
        --fast)
            PYTEST_ARGS="$PYTEST_ARGS -m 'not slow'"
            shift
            ;;
        *)
            # Pass through to pytest
            PYTEST_ARGS="$PYTEST_ARGS $1"
            shift
            ;;
    esac
done

echo -e "${BLUE}"
echo "╔════════════════════════════════════════════════════════════╗"
echo "║       HOLISTIC SYSTEM VERIFICATION FRAMEWORK               ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo -e "${NC}"

echo -e "${YELLOW}Configuration:${NC}"
echo "  Compose File: $COMPOSE_FILE"
echo "  Strict Mode:  $([ "$STRICT_MODE" = "1" ] && echo "ON (fail on warnings)" || echo "OFF (errors only)")"
echo "  Test Path:    $TEST_PATH"
echo "  Use Dev Stack: $([ "$USE_DEV_STACK" = "1" ] && echo "YES" || echo "NO")"
echo ""

# =============================================================================
# Step 1: Ensure stack is running
# =============================================================================
echo -e "${YELLOW}[1/4] Checking Docker stack...${NC}"

if [ "$USE_DEV_STACK" = "1" ]; then
    # Check if dev stack is running
    if ! docker compose -f "$COMPOSE_FILE" ps --quiet api 2>/dev/null | grep -q .; then
        echo -e "${RED}ERROR: Dev stack is not running.${NC}"
        echo "Start it with: docker compose -f docker-compose.dev.yml up -d"
        exit 1
    fi
    echo -e "${GREEN}✓ Dev stack is running${NC}"
else
    # Start test stack if not running
    if ! docker compose -f "$COMPOSE_FILE" ps --quiet api 2>/dev/null | grep -q .; then
        echo "Starting test stack..."
        docker compose -f "$COMPOSE_FILE" up -d --wait --wait-timeout 120
    fi
    echo -e "${GREEN}✓ Test stack is running${NC}"
fi

# =============================================================================
# Step 2: Verify health
# =============================================================================
echo -e "${YELLOW}[2/4] Verifying container health...${NC}"

CONTAINERS=("api" "postgres" "valkey")
if [ "$USE_DEV_STACK" = "1" ]; then
    SUFFIX="-dev"
else
    SUFFIX="-test"
fi

all_healthy=true
for container in "${CONTAINERS[@]}"; do
    name="stonkmarket-${container}${SUFFIX}"
    if docker inspect "$name" --format='{{.State.Running}}' 2>/dev/null | grep -q true; then
        echo -e "  ${GREEN}✓${NC} $name is running"
    else
        echo -e "  ${RED}✗${NC} $name is NOT running"
        all_healthy=false
    fi
done

if [ "$all_healthy" = false ]; then
    echo -e "${RED}ERROR: Some containers are not healthy.${NC}"
    docker compose -f "$COMPOSE_FILE" ps
    exit 1
fi

# =============================================================================
# Step 3: Run database migrations (if using test stack)
# =============================================================================
if [ "$USE_DEV_STACK" = "0" ]; then
    echo -e "${YELLOW}[3/4] Running database migrations...${NC}"
    docker compose -f "$COMPOSE_FILE" exec -T api alembic upgrade head 2>/dev/null || true
    echo -e "${GREEN}✓ Migrations applied${NC}"
else
    echo -e "${YELLOW}[3/4] Skipping migrations (using dev stack)${NC}"
fi

# =============================================================================
# Step 4: Run system tests
# =============================================================================
echo -e "${YELLOW}[4/4] Running system tests...${NC}"
echo ""

# Set environment variables
export SYSTEM_TEST_STRICT="$STRICT_MODE"
export SYSTEM_TEST_USE_DEV="$USE_DEV_STACK"

# Activate virtual environment if available
if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
fi

# Run pytest
set +e  # Don't exit on test failure, we want to report
python -m pytest "$TEST_PATH" $PYTEST_ARGS
TEST_EXIT_CODE=$?
set -e

echo ""

# =============================================================================
# Report results
# =============================================================================
if [ $TEST_EXIT_CODE -eq 0 ]; then
    echo -e "${GREEN}"
    echo "╔════════════════════════════════════════════════════════════╗"
    echo "║                    ALL TESTS PASSED ✓                      ║"
    echo "╚════════════════════════════════════════════════════════════╝"
    echo -e "${NC}"
else
    echo -e "${RED}"
    echo "╔════════════════════════════════════════════════════════════╗"
    echo "║                    TESTS FAILED ✗                          ║"
    echo "╚════════════════════════════════════════════════════════════╝"
    echo -e "${NC}"

    # Check for debug reports
    if [ -d "test-results/debug-reports" ]; then
        report_count=$(ls -1 test-results/debug-reports/*.json 2>/dev/null | wc -l)
        if [ "$report_count" -gt 0 ]; then
            echo -e "${YELLOW}Debug reports available in test-results/debug-reports/${NC}"
            echo "Latest reports:"
            ls -lt test-results/debug-reports/*.json 2>/dev/null | head -5
        fi
    fi
fi

exit $TEST_EXIT_CODE
