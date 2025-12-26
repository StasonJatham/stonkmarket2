"""Main application entry point with app factory and lifespan management."""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from sqlalchemy import text, select

from app.api.app import create_api_app
from app.cache.client import close_valkey_client, get_valkey_client
from app.core.config import settings
from app.core.logging import get_logger, setup_logging
from app.database.connection import init_pg_pool, close_pg_pool, get_session
from app.database.orm import SchemaMigration
from app.services.runtime_settings import init_runtime_settings
from app.repositories.api_keys import seed_api_keys_from_env
from app.repositories.auth_user import seed_admin_from_env
import app.jobs.definitions  # noqa: F401 - register jobs

logger = get_logger("main")


async def run_migrations() -> None:
    """Run all pending database migrations from the migrations folder.
    
    Uses PostgreSQL advisory lock to ensure only one worker runs migrations at a time.
    """
    async with get_session() as session:
        # Try to acquire advisory lock (non-blocking)
        # Lock ID 1 is reserved for migrations
        lock_result = await session.execute(text("SELECT pg_try_advisory_lock(1) as acquired"))
        row = lock_result.mappings().first()
        if not row or not row.get("acquired"):
            logger.info("Another worker is running migrations, skipping")
            return
        
        try:
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
                
                # Check if already applied using ORM
                existing = await session.execute(
                    select(SchemaMigration).where(SchemaMigration.version == version)
                )
                if existing.scalar_one_or_none():
                    continue
                
                # Run migration
                logger.info(f"Running migration: {version}")
                try:
                    sql = migration_file.read_text()
                    await session.execute(text(sql))
                    
                    # Mark as applied using ORM
                    migration_record = SchemaMigration(version=version)
                    session.add(migration_record)
                    await session.commit()
                    
                    applied_count += 1
                    logger.info(f"Migration {version} applied successfully")
                except Exception as e:
                    await session.rollback()
                    logger.error(f"Migration {version} failed: {e}")
                    # Don't raise - allow app to start, admin can fix manually
                    break
            
            if applied_count > 0:
                logger.info(f"Applied {applied_count} migration(s)")
            else:
                logger.info("All migrations already applied")
        finally:
            # Always release the advisory lock
            await session.execute(text("SELECT pg_advisory_unlock(1)"))


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

    # Seed admin user from environment variables (if not already in db)
    await seed_admin_from_env()

    # Seed API keys from environment variables (if not already in db)
    await seed_api_keys_from_env()

    # Initialize Valkey connection
    try:
        await get_valkey_client()
        logger.info("Valkey connection established")
    except Exception as e:
        logger.warning(f"Valkey connection failed (cache disabled): {e}")

    yield

    # Shutdown
    logger.info("Shutting down...")

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
