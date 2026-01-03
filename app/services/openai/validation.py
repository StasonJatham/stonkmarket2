"""
Output validation and repair for AI-generated content.

Provides validation of AI outputs against expected formats and
automatic repair attempts for minor issues.
"""

from __future__ import annotations

import re
from typing import Any

from pydantic import ValidationError

from app.core.logging import get_logger
from app.services.openai.config import TaskType, get_task_config
from app.services.openai.schemas import get_output_model

logger = get_logger("openai.validation")


# Emoji detection pattern
EMOJI_PATTERN = re.compile(
    "[\U0001F600-\U0001F64F"  # emoticons
    "\U0001F300-\U0001F5FF"  # symbols & pictographs
    "\U0001F680-\U0001F6FF"  # transport & map
    "\U0001F1E0-\U0001F1FF"  # flags
    "\U00002702-\U000027B0"  # dingbats
    "\U0001F900-\U0001F9FF"  # supplemental symbols
    "\U0001FA00-\U0001FA6F"  # chess symbols
    "\U0001FA70-\U0001FAFF"  # symbols extended
    "\U00002600-\U000026FF"  # misc symbols
    "]+",
    flags=re.UNICODE,
)


def count_emojis(text: str) -> int:
    """Count the number of emojis in text."""
    return len(EMOJI_PATTERN.findall(text))


def validate_output(task: TaskType, output: dict[str, Any]) -> tuple[bool, list[str]]:
    """
    Validate AI output against task-specific rules.
    
    Returns:
        Tuple of (is_valid, list of validation errors)
    """
    errors: list[str] = []
    config = get_task_config(task)
    
    # First validate against Pydantic model
    try:
        model = get_output_model(task)
        model.model_validate(output)
    except ValidationError as e:
        errors.extend([err["msg"] for err in e.errors()])
        return False, errors
    
    # Task-specific validation
    if task == TaskType.BIO:
        bio = output.get("bio", "")
        
        # Check character limits
        if config.min_chars and len(bio) < config.min_chars:
            errors.append(f"Bio too short: {len(bio)} chars (min {config.min_chars})")
        if config.max_chars and len(bio) > config.max_chars:
            errors.append(f"Bio too long: {len(bio)} chars (max {config.max_chars})")
        
        # Check emoji count
        if config.max_emojis:
            emoji_count = count_emojis(bio)
            if emoji_count > config.max_emojis:
                errors.append(f"Too many emojis: {emoji_count} (max {config.max_emojis})")
    
    elif task == TaskType.SUMMARY:
        summary = output.get("summary", "")
        
        # Check character limits
        if config.min_chars and len(summary) < config.min_chars:
            errors.append(f"Summary too short: {len(summary)} chars (min {config.min_chars})")
        if config.max_chars and len(summary) > config.max_chars:
            errors.append(f"Summary too long: {len(summary)} chars (max {config.max_chars})")
    
    elif task == TaskType.RATING:
        reasoning = output.get("reasoning", "")
        
        # Check reasoning length
        if config.reasoning_max_chars and len(reasoning) > config.reasoning_max_chars:
            errors.append(
                f"Reasoning too long: {len(reasoning)} chars (max {config.reasoning_max_chars})"
            )
    
    elif task == TaskType.PORTFOLIO:
        headline = output.get("headline", "")
        if len(headline) > 120:
            errors.append(f"Headline too long: {len(headline)} chars (max 120)")
        
        insights = output.get("insights", [])
        if len(insights) < 2:
            errors.append(f"Too few insights: {len(insights)} (min 2)")
        if len(insights) > 4:
            errors.append(f"Too many insights: {len(insights)} (max 4)")
    
    return len(errors) == 0, errors


def repair_output(task: TaskType, output: dict[str, Any]) -> dict[str, Any]:
    """
    Attempt to repair minor issues in AI output.
    
    Handles:
    - Truncating text that's too long
    - Removing excess emojis
    - Trimming whitespace
    """
    config = get_task_config(task)
    result = output.copy()
    
    if task == TaskType.BIO:
        bio = result.get("bio", "")
        
        # Trim whitespace
        bio = bio.strip()
        
        # Truncate if too long
        if config.max_chars and len(bio) > config.max_chars:
            bio = truncate_at_sentence(bio, config.max_chars)
        
        # Remove excess emojis
        if config.max_emojis:
            emoji_count = count_emojis(bio)
            if emoji_count > config.max_emojis:
                bio = remove_excess_emojis(bio, config.max_emojis)
        
        result["bio"] = bio
    
    elif task == TaskType.SUMMARY:
        summary = result.get("summary", "")
        summary = summary.strip()
        
        if config.max_chars and len(summary) > config.max_chars:
            summary = truncate_at_sentence(summary, config.max_chars)
        
        result["summary"] = summary
    
    elif task == TaskType.RATING:
        reasoning = result.get("reasoning", "")
        reasoning = reasoning.strip()
        
        if config.reasoning_max_chars and len(reasoning) > config.reasoning_max_chars:
            reasoning = truncate_at_sentence(reasoning, config.reasoning_max_chars)
        
        result["reasoning"] = reasoning
    
    elif task == TaskType.PORTFOLIO:
        headline = result.get("headline", "")
        if len(headline) > 120:
            result["headline"] = headline[:117] + "..."
        
        # Limit insights and actions
        if len(result.get("insights", [])) > 4:
            result["insights"] = result["insights"][:4]
        if len(result.get("actions", [])) > 3:
            result["actions"] = result["actions"][:3]
        if len(result.get("risks", [])) > 3:
            result["risks"] = result["risks"][:3]
    
    return result


def truncate_at_sentence(text: str, max_chars: int) -> str:
    """Truncate text at a sentence boundary, respecting max_chars."""
    if len(text) <= max_chars:
        return text
    
    # Find the last sentence boundary before max_chars
    truncated = text[:max_chars]
    
    # Look for sentence endings
    for sep in [". ", "! ", "? "]:
        last_sep = truncated.rfind(sep)
        if last_sep > max_chars // 2:  # Only if we keep at least half
            return truncated[:last_sep + 1].strip()
    
    # Fall back to word boundary
    last_space = truncated.rfind(" ")
    if last_space > max_chars // 2:
        return truncated[:last_space].strip() + "..."
    
    # Hard truncate
    return truncated[:max_chars - 3].strip() + "..."


def remove_excess_emojis(text: str, max_emojis: int) -> str:
    """Remove emojis beyond the maximum allowed."""
    emojis_found = EMOJI_PATTERN.findall(text)
    
    if len(emojis_found) <= max_emojis:
        return text
    
    # Keep first N emojis, remove the rest
    emojis_to_remove = emojis_found[max_emojis:]
    
    result = text
    for emoji in emojis_to_remove:
        result = result.replace(emoji, "", 1)
    
    return result


def extract_text_field(task: TaskType, output: dict[str, Any]) -> str | None:
    """Extract the primary text field from a structured output."""
    if task == TaskType.BIO:
        return output.get("bio")
    elif task == TaskType.SUMMARY:
        return output.get("summary")
    elif task in (TaskType.RATING, TaskType.AGENT):
        return output.get("reasoning")
    elif task == TaskType.PORTFOLIO:
        return output.get("headline")
    return None
