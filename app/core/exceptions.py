"""Custom exceptions and centralized exception handlers."""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse


class AppException(Exception):
    """Base application exception with structured error response."""

    status_code: int = status.HTTP_500_INTERNAL_SERVER_ERROR
    error_code: str = "INTERNAL_ERROR"
    message: str = "An unexpected error occurred"

    def __init__(
        self,
        message: str | None = None,
        error_code: str | None = None,
        status_code: int | None = None,
        details: dict[str, Any] | None = None,
    ):
        self.message = message or self.message
        self.error_code = error_code or self.error_code
        self.status_code = status_code or self.status_code
        self.details = details or {}
        super().__init__(self.message)

    def to_dict(self) -> dict[str, Any]:
        """Convert to RFC 7807 problem+json style response."""
        return {
            "error": self.error_code,
            "message": self.message,
            "status": self.status_code,
            **({"details": self.details} if self.details else {}),
        }


class NotFoundError(AppException):
    """Resource not found."""

    status_code = status.HTTP_404_NOT_FOUND
    error_code = "NOT_FOUND"
    message = "Resource not found"


class BadRequestError(AppException):
    """Bad request."""

    status_code = status.HTTP_400_BAD_REQUEST
    error_code = "BAD_REQUEST"
    message = "Bad request"


class AuthenticationError(AppException):
    """Authentication failed."""

    status_code = status.HTTP_401_UNAUTHORIZED
    error_code = "AUTHENTICATION_FAILED"
    message = "Authentication required"


class AuthorizationError(AppException):
    """Authorization failed."""

    status_code = status.HTTP_403_FORBIDDEN
    error_code = "FORBIDDEN"
    message = "You don't have permission to access this resource"


class ValidationError(AppException):
    """Validation failed."""

    status_code = status.HTTP_422_UNPROCESSABLE_CONTENT
    error_code = "VALIDATION_ERROR"
    message = "Validation failed"


class RateLimitError(AppException):
    """Rate limit exceeded."""

    status_code = status.HTTP_429_TOO_MANY_REQUESTS
    error_code = "RATE_LIMIT_EXCEEDED"
    message = "Too many requests. Please try again later."


class ExternalServiceError(AppException):
    """External service error."""

    status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    error_code = "EXTERNAL_SERVICE_ERROR"
    message = "External service temporarily unavailable"


class ConflictError(AppException):
    """Resource conflict."""

    status_code = status.HTTP_409_CONFLICT
    error_code = "CONFLICT"
    message = "Resource conflict"


class CacheError(AppException):
    """Cache operation failed."""

    status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
    error_code = "CACHE_ERROR"
    message = "Cache operation failed"


class JobError(AppException):
    """Job execution failed."""

    status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
    error_code = "JOB_ERROR"
    message = "Job execution failed"


def register_exception_handlers(app: FastAPI) -> None:
    """Register centralized exception handlers."""

    @app.exception_handler(AppException)
    async def app_exception_handler(
        request: Request, exc: AppException
    ) -> JSONResponse:
        return JSONResponse(
            status_code=exc.status_code,
            content=exc.to_dict(),
            headers={"X-Request-ID": getattr(request.state, "request_id", "unknown")},
        )

    @app.exception_handler(Exception)
    async def generic_exception_handler(
        request: Request, exc: Exception
    ) -> JSONResponse:
        # Log the actual error but return generic message
        import logging

        logger = logging.getLogger("stonkmarket.error")
        logger.exception(
            "Unhandled exception",
            extra={
                "request_id": getattr(request.state, "request_id", "unknown"),
                "path": request.url.path,
                "method": request.method,
            },
        )

        # Don't expose internal error details in production
        from .config import settings

        if settings.debug:
            message = str(exc)
        else:
            message = "An unexpected error occurred"

        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "error": "INTERNAL_ERROR",
                "message": message,
                "status": 500,
            },
            headers={"X-Request-ID": getattr(request.state, "request_id", "unknown")},
        )
