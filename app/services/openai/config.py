"""
OpenAI client configuration.

Centralized configuration for all OpenAI-related settings with environment
variable support and runtime overrides from database settings.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import timedelta
from enum import Enum
from functools import lru_cache
from typing import Any

from pydantic import Field
from pydantic_settings import BaseSettings


class TaskType(str, Enum):
    """Supported AI generation tasks."""
    BIO = "bio"              # Dating-app style stock bio
    RATING = "rating"        # Buy/hold/sell rating with reasoning
    SUMMARY = "summary"      # Company description summary
    AGENT = "agent"          # Persona agent analysis (Buffett, Lynch, etc.)
    PORTFOLIO = "portfolio"  # Portfolio advisor analysis


@dataclass(frozen=True)
class ModelLimits:
    """Token limits for a specific model."""
    context_window: int
    max_output: int
    reserved_overhead: int = 500  # Safety margin for system tokens


# Model token limits - updated for 2025/2026 models
MODEL_LIMITS: dict[str, ModelLimits] = {
    # GPT-5 family (o-series reasoning models)
    "gpt-5": ModelLimits(context_window=200_000, max_output=100_000),
    "gpt-5-mini": ModelLimits(context_window=200_000, max_output=100_000),
    "o3": ModelLimits(context_window=200_000, max_output=100_000),
    "o3-mini": ModelLimits(context_window=200_000, max_output=100_000),
    "o1": ModelLimits(context_window=200_000, max_output=100_000),
    "o1-mini": ModelLimits(context_window=128_000, max_output=65_536),
    "o1-preview": ModelLimits(context_window=128_000, max_output=32_768),
    # GPT-4 family
    "gpt-4o": ModelLimits(context_window=128_000, max_output=16_384),
    "gpt-4o-mini": ModelLimits(context_window=128_000, max_output=16_384),
    "gpt-4-turbo": ModelLimits(context_window=128_000, max_output=4_096),
    "gpt-4": ModelLimits(context_window=8_192, max_output=4_096),
    # GPT-3.5 family
    "gpt-3.5-turbo": ModelLimits(context_window=16_385, max_output=4_096),
}

# Default fallback for unknown models
DEFAULT_MODEL_LIMITS = ModelLimits(context_window=128_000, max_output=4_096)


@dataclass(frozen=True)
class TaskConfig:
    """Configuration for a specific task type."""
    min_chars: int = 0           # Minimum output characters (0 = no limit)
    max_chars: int = 0           # Maximum output characters (0 = no limit)
    reasoning_overhead: int = 100  # GPT-5 reasoning token overhead
    reasoning_max_chars: int = 0   # Max chars for reasoning field
    max_emojis: int = 0          # Max emoji count (0 = no limit)
    default_max_tokens: int = 300  # Default output tokens for this task


# Per-task configuration
TASK_CONFIGS: dict[TaskType, TaskConfig] = {
    TaskType.BIO: TaskConfig(
        min_chars=150,
        max_chars=300,
        reasoning_overhead=100,
        max_emojis=2,
        default_max_tokens=200,
    ),
    TaskType.RATING: TaskConfig(
        min_chars=0,
        max_chars=0,
        reasoning_overhead=300,
        reasoning_max_chars=400,
        default_max_tokens=400,
    ),
    TaskType.SUMMARY: TaskConfig(
        min_chars=280,
        max_chars=420,
        reasoning_overhead=150,
        default_max_tokens=250,
    ),
    TaskType.AGENT: TaskConfig(
        min_chars=0,
        max_chars=0,
        reasoning_overhead=300,
        reasoning_max_chars=500,
        default_max_tokens=500,
    ),
    TaskType.PORTFOLIO: TaskConfig(
        min_chars=500,
        max_chars=1000,
        reasoning_overhead=200,
        default_max_tokens=800,
    ),
}


class OpenAISettings(BaseSettings):
    """OpenAI client configuration from environment variables."""
    
    api_key: str = Field(default="", alias="OPENAI_API_KEY")
    default_model: str = Field(default="gpt-5-mini", alias="OPENAI_MODEL")
    
    # Retry configuration
    max_retries: int = Field(default=3, alias="OPENAI_MAX_RETRIES")
    retry_delay: float = Field(default=1.0, alias="OPENAI_RETRY_DELAY")
    retry_max_delay: float = Field(default=30.0, alias="OPENAI_RETRY_MAX_DELAY")
    
    # Circuit breaker configuration
    circuit_breaker_threshold: int = Field(default=5, alias="OPENAI_CB_THRESHOLD")
    circuit_breaker_timeout: int = Field(default=60, alias="OPENAI_CB_TIMEOUT")
    
    # Connection configuration
    client_ttl_hours: int = Field(default=1, alias="OPENAI_CLIENT_TTL_HOURS")
    max_connections: int = Field(default=100, alias="OPENAI_MAX_CONNECTIONS")
    
    # Token estimation
    chars_per_token: int = Field(default=4, alias="OPENAI_CHARS_PER_TOKEN")
    max_description_chars: int = Field(default=50_000, alias="OPENAI_MAX_DESC_CHARS")
    
    # Telemetry
    record_usage: bool = Field(default=True, alias="OPENAI_RECORD_USAGE")
    
    model_config = {"env_file": ".env", "extra": "ignore"}
    
    @property
    def client_ttl(self) -> timedelta:
        """Get client TTL as timedelta."""
        return timedelta(hours=self.client_ttl_hours)


@lru_cache(maxsize=1)
def get_settings() -> OpenAISettings:
    """Get cached OpenAI settings instance."""
    return OpenAISettings()


def get_model_limits(model: str) -> ModelLimits:
    """Get token limits for a model, with fallback for unknown models."""
    # Check exact match first
    if model in MODEL_LIMITS:
        return MODEL_LIMITS[model]
    
    # Check prefix matches (e.g., "gpt-4o-2024-01-01" matches "gpt-4o")
    for prefix, limits in MODEL_LIMITS.items():
        if model.startswith(prefix):
            return limits
    
    return DEFAULT_MODEL_LIMITS


def get_task_config(task: TaskType) -> TaskConfig:
    """Get configuration for a task type."""
    return TASK_CONFIGS.get(task, TaskConfig())


def is_reasoning_model(model: str) -> bool:
    """Check if model is a GPT-5/o-series reasoning model."""
    return any(model.startswith(prefix) for prefix in ("gpt-5", "o1", "o3"))
