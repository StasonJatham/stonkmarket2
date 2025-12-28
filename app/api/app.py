"""API application factory."""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware

from app.core.config import settings
from app.core.exceptions import register_exception_handlers
from app.core.logging import get_logger, request_id_var
from app.schemas.common import ErrorResponse

from .routes import (
    admin_settings,
    ai_personas,
    api_keys,
    auth,
    celery,
    cronjobs,
    dip_changes,
    dip_entry,
    dipfinder,
    dips,
    health,
    logos,
    metrics,
    mfa,
    portfolios,
    quant_engine,
    seo,
    strategy,
    suggestions,
    swipe,
    symbols,
    user_api_keys,
)


logger = get_logger("api")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifecycle - initialize and cleanup resources."""
    from app.cache.client import close_valkey_client, init_valkey_pool
    from app.database.connection import close_pg_pool, init_pg_pool

    # Initialize resources on startup
    try:
        await init_pg_pool()
        await init_valkey_pool()
    except Exception as e:
        logger.warning(f"Resource initialization failed (may be ok in tests): {e}")

    yield

    # Cleanup resources on shutdown
    try:
        await close_pg_pool()
        await close_valkey_client()
    except Exception as e:
        logger.warning(f"Resource cleanup failed: {e}")


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Add security headers to all responses."""

    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)

        # Security headers
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        response.headers["Permissions-Policy"] = (
            "geolocation=(), microphone=(), camera=()"
        )
        # Content Security Policy (adjust as needed for your CDN/analytics)
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; "
            "script-src 'self' 'unsafe-inline' https://umami.karlcom.de; "
            "style-src 'self' 'unsafe-inline'; "
            "img-src 'self' data: https:; "
            "font-src 'self' data:; "
            "connect-src 'self' https://umami.karlcom.de wss:; "
            "frame-ancestors 'none'; "
            "base-uri 'self'; "
            "form-action 'self'"
        )

        # HSTS (only when HTTPS is enabled)
        if settings.https_enabled:
            response.headers["Strict-Transport-Security"] = (
                "max-age=31536000; includeSubDomains"
            )

        return response


class RequestIDMiddleware(BaseHTTPMiddleware):
    """Add request ID to all requests for tracing."""

    async def dispatch(self, request: Request, call_next):
        import uuid

        request_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())
        request.state.request_id = request_id
        request_id_var.set(request_id)

        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id

        return response


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Log all requests with sensitive data scrubbing."""

    async def dispatch(self, request: Request, call_next):
        import time

        start_time = time.monotonic()

        response = await call_next(request)

        duration = time.monotonic() - start_time

        # Log path only (not query params which may contain sensitive data)
        path = request.url.path

        logger.info(
            f"{request.method} {path} -> {response.status_code} ({duration:.3f}s)",
            extra={
                "method": request.method,
                "path": path,
                "status_code": response.status_code,
                "duration_ms": int(duration * 1000),
            },
        )

        return response


def create_api_app() -> FastAPI:
    """Create and configure the API application."""
    app = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        description="Stock dip tracking and analysis API",
        root_path=settings.root_path,
        docs_url="/docs" if settings.debug else None,
        redoc_url="/redoc" if settings.debug else None,
        openapi_url="/openapi.json" if settings.debug else None,
        lifespan=lifespan,
        responses={
            400: {"model": ErrorResponse, "description": "Bad Request"},
            401: {"model": ErrorResponse, "description": "Unauthorized"},
            403: {"model": ErrorResponse, "description": "Forbidden"},
            404: {"model": ErrorResponse, "description": "Not Found"},
            422: {"model": ErrorResponse, "description": "Validation Error"},
            429: {"model": ErrorResponse, "description": "Rate Limit Exceeded"},
            500: {"model": ErrorResponse, "description": "Internal Server Error"},
        },
    )

    # Add middlewares (order matters - first added is outermost)
    app.add_middleware(RequestLoggingMiddleware)
    app.add_middleware(RequestIDMiddleware)
    app.add_middleware(SecurityHeadersMiddleware)

    # CORS - strict configuration (no wildcards with credentials)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["GET", "POST", "PUT", "DELETE", "PATCH"],
        allow_headers=["Authorization", "Content-Type", "X-Request-ID", "X-CSRF-Token"],
        expose_headers=["X-Request-ID"],
        max_age=600,
    )

    # Register exception handlers
    register_exception_handlers(app)

    # Include routers
    app.include_router(health.router, tags=["Health"])
    app.include_router(auth.router, prefix="/auth", tags=["Authentication"])
    app.include_router(mfa.router, prefix="/auth/mfa", tags=["MFA"])
    app.include_router(api_keys.router, prefix="/admin/api-keys", tags=["API Keys"])
    app.include_router(
        admin_settings.router, prefix="/admin/settings", tags=["Admin Settings"]
    )
    app.include_router(user_api_keys.router, tags=["User API Keys"])
    app.include_router(symbols.router, prefix="/symbols", tags=["Symbols"])
    app.include_router(dips.router, prefix="/dips", tags=["Dips"])
    app.include_router(dip_changes.router, tags=["Dip Changes"])
    app.include_router(swipe.router, prefix="/swipe", tags=["Swipe"])
    app.include_router(dipfinder.router, prefix="/dipfinder", tags=["DipFinder"])
    app.include_router(cronjobs.router, prefix="/cronjobs", tags=["CronJobs"])
    app.include_router(suggestions.router, tags=["Suggestions"])
    app.include_router(metrics.router, tags=["Metrics"])
    app.include_router(logos.router, tags=["Logos"])
    app.include_router(seo.router, tags=["SEO"])  # No prefix - serves at root for crawlers
    app.include_router(portfolios.router, tags=["Portfolios"])
    app.include_router(quant_engine.router, tags=["Quant Engine"])
    app.include_router(quant_engine.global_router, tags=["Quant Engine"])
    app.include_router(strategy.router, prefix="/signals/strategy", tags=["Strategy Signals"])
    app.include_router(dip_entry.router, prefix="/signals/dip-entry", tags=["Dip Entry"])
    app.include_router(celery.router, tags=["Celery"])
    app.include_router(ai_personas.router, tags=["AI Personas"])

    return app
