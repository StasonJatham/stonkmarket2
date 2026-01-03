"""
OpenAI Client Package - Modular, type-safe OpenAI integration.

Usage:
    from app.services.openai import (
        # Core generation
        generate,
        generate_bio,
        rate_dip,
        summarize_company,
        
        # Batch processing
        submit_batch,
        check_batch,
        collect_batch,
        retry_failed_items,
        
        # Configuration
        TaskType,
        get_settings,
        
        # Client access
        get_client,
        openai_client,
        
        # Output schemas
        BioOutput,
        RatingOutput,
        SummaryOutput,
        AgentOutput,
        PortfolioOutput,
    )
"""

from app.services.openai.batch import (
    cancel_batch,
    check_batch,
    collect_batch,
    retry_failed_items,
    submit_batch,
)
from app.services.openai.client import (
    check_api_key,
    get_available_models,
    get_client,
    get_client_manager,
    openai_client,
)
from app.services.openai.config import (
    OpenAISettings,
    TaskType,
    get_model_limits,
    get_settings,
    get_task_config,
    is_reasoning_model,
)
from app.services.openai.contexts import (
    AgentContext,
    BioContext,
    PortfolioContext,
    RatingContext,
    SummaryContext,
    context_to_dict,
)
from app.services.openai.generate import (
    UsageMetrics,
    build_prompt,
    calculate_safe_output_tokens,
    count_tokens,
    estimate_tokens,
    generate,
    generate_bio,
    rate_dip,
    summarize_company,
)
from app.services.openai.prompts import INSTRUCTIONS, get_instructions
from app.services.openai.schemas import (
    AgentOutput,
    BioOutput,
    HealthRating,
    InsightType,
    PortfolioOutput,
    RatingOutput,
    RatingValue,
    RiskSeverity,
    SummaryOutput,
    TASK_SCHEMAS,
    get_json_schema,
    get_output_model,
)
from app.services.openai.validation import (
    extract_text_field,
    repair_output,
    truncate_at_sentence,
    validate_output,
)

# Backward compatibility: DEFAULT_MODEL
DEFAULT_MODEL = get_settings().default_model

__all__ = [
    # Core generation
    "generate",
    "generate_bio",
    "rate_dip",
    "summarize_company",
    # Batch processing
    "submit_batch",
    "check_batch",
    "collect_batch",
    "retry_failed_items",
    "cancel_batch",
    # Client
    "get_client",
    "get_client_manager",
    "openai_client",
    "check_api_key",
    "get_available_models",
    # Config
    "TaskType",
    "OpenAISettings",
    "get_settings",
    "get_model_limits",
    "get_task_config",
    "is_reasoning_model",
    "DEFAULT_MODEL",
    # Contexts
    "BioContext",
    "RatingContext",
    "SummaryContext",
    "AgentContext",
    "PortfolioContext",
    "context_to_dict",
    # Output schemas
    "BioOutput",
    "RatingOutput",
    "SummaryOutput",
    "AgentOutput",
    "PortfolioOutput",
    "RatingValue",
    "InsightType",
    "HealthRating",
    "RiskSeverity",
    "TASK_SCHEMAS",
    "get_json_schema",
    "get_output_model",
    # Prompts
    "INSTRUCTIONS",
    "get_instructions",
    # Validation
    "validate_output",
    "repair_output",
    "truncate_at_sentence",
    "extract_text_field",
    # Generation helpers
    "UsageMetrics",
    "build_prompt",
    "count_tokens",
    "estimate_tokens",
    "calculate_safe_output_tokens",
]
