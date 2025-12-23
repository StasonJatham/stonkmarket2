"""Main application entry point with app factory and lifespan management."""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI

from app.api.app import create_api_app
from app.cache.client import close_valkey_client, get_valkey_client
from app.core.config import settings
from app.core.logging import get_logger, setup_logging
from app.database.connection import init_pg_pool, close_pg_pool, execute, fetch_one
from app.jobs import start_scheduler, stop_scheduler
from app.services.runtime_settings import init_runtime_settings
from app.repositories.api_keys import seed_api_keys_from_env
import app.jobs.definitions  # noqa: F401 - register jobs

logger = get_logger("main")


async def run_migrations() -> None:
    """Run all pending database migrations from the migrations folder."""
    # Ensure migrations tracking table exists
    await execute("""
        CREATE TABLE IF NOT EXISTS schema_migrations (
            version VARCHAR(50) PRIMARY KEY,
            applied_at TIMESTAMPTZ DEFAULT NOW()
        );
    """)
    
    # Find migrations directory (relative to app)
    migrations_dir = Path(__file__).parent.parent / "migrations"
    if not migrations_dir.exists():
        logger.info("No migrations directory found, skipping migrations")
        return
    
    # Get all .sql files sorted by name
    migration_files = sorted(migrations_dir.glob("*.sql"))
    if not migration_files:
        logger.info("No migration files found")
        return
    
    applied_count = 0
    for migration_file in migration_files:
        version = migration_file.stem  # e.g., "005_add_symbol_type_and_dip_start"
        
        # Check if already applied
        result = await fetch_one(
            "SELECT version FROM schema_migrations WHERE version = $1",
            version
        )
        if result:
            continue
        
        # Run migration
        logger.info(f"Running migration: {version}")
        try:
            sql = migration_file.read_text()
            await execute(sql)
            
            # Mark as applied
            await execute(
                "INSERT INTO schema_migrations (version) VALUES ($1)",
                version
            )
            applied_count += 1
            logger.info(f"Migration {version} applied successfully")
        except Exception as e:
            logger.error(f"Migration {version} failed: {e}")
            # Don't raise - allow app to start, admin can fix manually
            break
    
    if applied_count > 0:
        logger.info(f"Applied {applied_count} migration(s)")
    else:
        logger.info("All migrations already applied")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    # Startup
    setup_logging()
    logger.info(f"Starting {settings.app_name} v{settings.app_version}")
    logger.info(f"Environment: {settings.environment}")

    # Initialize database
    await init_pg_pool()
    logger.info("PostgreSQL pool initialized")

    # Run any pending migrations
    await run_migrations()

    # Initialize runtime settings from database
    await init_runtime_settings()
    logger.info("Runtime settings initialized")

    # Seed API keys from environment variables (if not already in db)
    await seed_api_keys_from_env()

    # Initialize Valkey connection
    try:
        await get_valkey_client()
        logger.info("Valkey connection established")
    except Exception as e:
        logger.warning(f"Valkey connection failed (cache disabled): {e}")

    # Start scheduler
    if settings.scheduler_enabled:
        await start_scheduler()
        logger.info("Job scheduler started")

    yield

    # Shutdown
    logger.info("Shutting down...")

    # Stop scheduler
    await stop_scheduler()

    # Close Valkey connection
    await close_valkey_client()

    # Close database connections
    await close_pg_pool()

    logger.info("Shutdown complete")


def create_app() -> FastAPI:
    """Create the main FastAPI application."""
    # Create API app
    api_app = create_api_app()

    # Create main app with lifespan
    app = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        lifespan=lifespan,
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
    )

    # Mount API
    app.mount("/api", api_app)

    # Root redirect to API docs (if enabled)
    @app.get("/")
    async def root():
        return {
            "name": settings.app_name,
            "version": settings.app_version,
            "docs": "/api/docs" if settings.debug else None,
            "health": "/api/health",
        }

    return app


# Application instance
app = create_app()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.debug,
        log_level=settings.log_level.lower(),
    )
