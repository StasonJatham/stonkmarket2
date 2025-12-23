"""Application settings with Pydantic validation and environment loading."""

from __future__ import annotations

from functools import lru_cache
from typing import List, Optional

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
        populate_by_name=True,
    )

    # Application
    app_name: str = "Stonkmarket API"
    app_version: str = "1.0.0"
    debug: bool = Field(
        default=False, description="Enable debug mode (disable in production)"
    )
    root_path: str = Field(default="", description="Root path for reverse proxy")
    environment: str = Field(
        default="production",
        description="Environment: development, staging, production",
    )

    # Database
    database_url: str = Field(
        default="postgresql://stonkmarket:stonkmarket@localhost:5432/stonkmarket",
        description="PostgreSQL connection URL",
    )
    db_pool_min_size: int = Field(
        default=5, ge=1, le=20, description="Minimum database pool connections"
    )
    db_pool_max_size: int = Field(
        default=20, ge=5, le=100, description="Maximum database pool connections"
    )

    # Security
    auth_secret: str = Field(
        default="dev-secret-please-change-in-production-min-32-chars",
        description="Secret key for JWT signing (min 32 chars in production)",
    )
    access_token_expire_minutes: int = Field(
        default=60 * 24 * 7, ge=1, description="JWT expiration in minutes"
    )
    password_min_length: int = Field(
        default=8, ge=6, description="Minimum password length"
    )

    # Admin - env vars are ADMIN_USER and ADMIN_PASS
    admin_user: str = Field(default="admin", min_length=3, alias="ADMIN_USER")
    admin_pass: str = Field(default="changeme", min_length=6, alias="ADMIN_PASS")

    # Domain and HTTPS
    domain: Optional[str] = Field(default=None, description="Cookie domain")
    https_enabled: bool = Field(default=False, description="Enable secure cookies")

    # CORS
    cors_origins: List[str] = Field(
        default_factory=lambda: ["http://localhost:5173", "http://localhost:3000"],
        description="Allowed CORS origins (no wildcards with credentials)",
    )

    # Rate limiting (designed to prevent abuse, not interfere with normal usage)
    rate_limit_enabled: bool = Field(default=True, description="Enable rate limiting")
    rate_limit_auth: str = Field(
        default="20/minute",
        description="Rate limit for auth endpoints (unauthenticated)",
    )
    rate_limit_api_anonymous: str = Field(
        default="60/minute", description="Rate limit for anonymous API requests"
    )
    rate_limit_api_authenticated: str = Field(
        default="600/minute", description="Rate limit for authenticated API requests"
    )
    # Note: Admin users bypass rate limiting entirely

    # Suggestion voting settings
    vote_cooldown_days: int = Field(
        default=7,
        ge=1,
        le=90,
        description="Days before same user can vote for same stock again",
    )

    # Auto-approve settings (all conditions must be met)
    auto_approve_enabled: bool = Field(
        default=False, description="Enable auto-approval of suggestions"
    )
    auto_approve_votes: int = Field(
        default=50, ge=5, description="Minimum votes required for auto-approval"
    )
    auto_approve_unique_voters: int = Field(
        default=10, ge=3, description="Minimum unique voters (by fingerprint) required"
    )
    auto_approve_min_age_hours: int = Field(
        default=48, ge=1, description="Minimum hours since suggestion created"
    )

    # Valkey (Redis-compatible)
    valkey_url: str = Field(
        default="redis://valkey:6379/0", description="Valkey connection URL"
    )
    valkey_max_connections: int = Field(default=10, ge=1, le=100)
    cache_default_ttl: int = Field(
        default=300, ge=1, description="Default cache TTL in seconds"
    )
    session_ttl: int = Field(
        default=60 * 60 * 24 * 7, ge=60, description="Session TTL in seconds"
    )

    # Stock data
    default_symbols: List[str] = Field(
        default_factory=lambda: [
            "AAPL",
            "MSFT",
            "GOOG",
            "AMZN",
            "META",
            "TSLA",
            "NVDA",
            "NFLX",
            "AMD",
        ]
    )
    default_min_dip_pct: float = Field(default=0.10, gt=0, lt=1)
    default_min_days: int = Field(default=2, ge=0)
    history_days: int = Field(default=400, ge=30)
    update_window_days: int = Field(default=5, ge=1)
    chart_days: int = Field(default=180, ge=7, le=365)

    # Scheduler
    scheduler_enabled: bool = Field(
        default=True, description="Enable background job scheduler"
    )
    scheduler_timezone: str = Field(default="UTC", description="Scheduler timezone")

    # External API timeouts
    external_api_timeout: int = Field(
        default=30, ge=5, le=120, description="External API timeout in seconds"
    )
    external_api_retries: int = Field(
        default=3, ge=0, le=5, description="External API retry count"
    )

    # OpenAI API settings
    openai_api_key: str = Field(
        default="", description="OpenAI API key for AI-powered analysis"
    )

    # Logo.dev API settings
    logo_dev_public_key: str = Field(
        default="", description="Logo.dev API public key"
    )
    logo_dev_secret_key: str = Field(
        default="", description="Logo.dev API secret key"
    )
    logo_cache_days: int = Field(
        default=90, ge=1, le=365, description="Days to cache logos before refetching"
    )

    # Logging
    log_level: str = Field(
        default="INFO", description="Log level: DEBUG, INFO, WARNING, ERROR"
    )
    log_format: str = Field(default="json", description="Log format: json or text")

    @field_validator("cors_origins", mode="before")
    @classmethod
    def parse_cors_origins(cls, v):
        if isinstance(v, str):
            return [origin.strip() for origin in v.split(",") if origin.strip()]
        return v

    @field_validator("default_symbols", mode="before")
    @classmethod
    def parse_symbols(cls, v):
        if isinstance(v, str):
            return [s.strip().upper() for s in v.split(",") if s.strip()]
        return v

    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        valid = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        upper = v.upper()
        if upper not in valid:
            raise ValueError(f"log_level must be one of {valid}")
        return upper

    @property
    def is_production(self) -> bool:
        return self.environment == "production"

    @property
    def is_development(self) -> bool:
        return self.environment == "development"


@lru_cache
def get_settings() -> Settings:
    """Cached settings factory."""
    return Settings()


settings = get_settings()
