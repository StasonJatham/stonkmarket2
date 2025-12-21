"""Main application entry point with app factory and lifespan management."""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api.app import create_api_app
from app.cache.client import close_valkey_client, get_valkey_client
from app.core.config import settings
from app.core.logging import get_logger, setup_logging
from app.database import close_db, init_db
from app.jobs import start_scheduler, stop_scheduler
from app.jobs.definitions import analysis_job, data_grab_job  # noqa: F401 - register jobs

logger = get_logger("main")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    # Startup
    setup_logging()
    logger.info(f"Starting {settings.app_name} v{settings.app_version}")
    logger.info(f"Environment: {settings.environment}")

    # Initialize database
    init_db()
    logger.info("Database initialized")

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
    close_db()

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
