#!/bin/bash
set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}   StonkMarket - Starting Application   ${NC}"
echo -e "${GREEN}========================================${NC}"

# Wait for database to be ready
echo -e "${YELLOW}Waiting for database to be ready...${NC}"
MAX_RETRIES=30
RETRY_COUNT=0

while [ $RETRY_COUNT -lt $MAX_RETRIES ]; do
    if python -c "
import asyncio
import asyncpg
import os

async def check_db():
    try:
        conn = await asyncpg.connect(os.environ.get('DATABASE_URL', 'postgresql://postgres:postgres@db:5432/stonkmarket'))
        await conn.execute('SELECT 1')
        await conn.close()
        return True
    except Exception as e:
        return False

exit(0 if asyncio.run(check_db()) else 1)
" 2>/dev/null; then
        echo -e "${GREEN}Database is ready!${NC}"
        break
    fi
    
    RETRY_COUNT=$((RETRY_COUNT + 1))
    echo -e "${YELLOW}Database not ready yet... retry $RETRY_COUNT/$MAX_RETRIES${NC}"
    sleep 2
done

if [ $RETRY_COUNT -eq $MAX_RETRIES ]; then
    echo -e "${RED}ERROR: Database connection failed after $MAX_RETRIES attempts${NC}"
    exit 1
fi

# Run Alembic migrations
echo -e "${YELLOW}Running database migrations...${NC}"

if [ -d "alembic" ]; then
    # Check if this is a fresh database (no alembic_version table)
    IS_FRESH=$(python -c "
import asyncio
import asyncpg
import os

async def check_fresh():
    conn = await asyncpg.connect(os.environ.get('DATABASE_URL', 'postgresql://postgres:postgres@db:5432/stonkmarket'))
    result = await conn.fetchval(\"\"\"
        SELECT EXISTS (
            SELECT FROM information_schema.tables 
            WHERE table_name = 'alembic_version'
        )
    \"\"\")
    await conn.close()
    print('existing' if result else 'fresh')

asyncio.run(check_fresh())
" 2>/dev/null)

    if [ "$IS_FRESH" = "fresh" ]; then
        # Check if auth_user table exists (schema from init.sql)
        HAS_SCHEMA=$(python -c "
import asyncio
import asyncpg
import os

async def check_schema():
    conn = await asyncpg.connect(os.environ.get('DATABASE_URL', 'postgresql://postgres:postgres@db:5432/stonkmarket'))
    result = await conn.fetchval(\"\"\"
        SELECT EXISTS (
            SELECT FROM information_schema.tables 
            WHERE table_name = 'auth_user'
        )
    \"\"\")
    await conn.close()
    print('yes' if result else 'no')

asyncio.run(check_schema())
" 2>/dev/null)

        if [ "$HAS_SCHEMA" = "yes" ]; then
            # Existing database without Alembic tracking - stamp it
            echo -e "${YELLOW}Existing database detected without Alembic version tracking.${NC}"
            echo -e "${YELLOW}Stamping baseline migration...${NC}"
            alembic stamp 001_baseline
        else
            # Fresh database - run all migrations
            echo -e "${YELLOW}Fresh database detected. Running all migrations...${NC}"
            alembic upgrade head
        fi
    else
        # Existing Alembic-tracked database - run pending migrations
        echo -e "${YELLOW}Running pending migrations...${NC}"
        alembic upgrade head
    fi
    
    echo -e "${GREEN}Migrations complete!${NC}"
else
    echo -e "${YELLOW}No alembic directory found, skipping migrations${NC}"
fi

# Show current migration version
if [ -d "alembic" ]; then
    echo -e "${YELLOW}Current migration version:${NC}"
    alembic current
fi

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}   Starting application server...       ${NC}"
echo -e "${GREEN}========================================${NC}"

# Execute the main command
exec "$@"
