"""Text cleaner utility for removing AI-generated text artifacts.

LLMs like ChatGPT tend to use certain characters that are uncommon in human text.
This module provides utilities to detect and replace these artifacts.
"""

from __future__ import annotations

import re
from typing import Optional


# Common AI artifacts: Unicode characters that LLMs prefer over ASCII equivalents
AI_REPLACEMENTS = {
    # Em dashes and en dashes → regular hyphen
    "\u2014": "-",      # — (em dash)
    "\u2013": "-",      # – (en dash)
    "\u2012": "-",      # ‒ (figure dash)
    "\u2015": "-",      # ― (horizontal bar)
    
    # Smart quotes → regular quotes
    "\u201c": '"',      # " (left double quotation)
    "\u201d": '"',      # " (right double quotation)
    "\u201e": '"',      # „ (double low-9 quotation)
    "\u201f": '"',      # ‟ (double high-reversed-9 quotation)
    "\u2018": "'",      # ' (left single quotation)
    "\u2019": "'",      # ' (right single quotation / apostrophe)
    "\u201a": "'",      # ‚ (single low-9 quotation)
    "\u201b": "'",      # ‛ (single high-reversed-9 quotation)
    "\u2032": "'",      # ′ (prime)
    "\u2033": '"',      # ″ (double prime)
    
    # Ellipsis
    "\u2026": "...",    # … (horizontal ellipsis)
    
    # Spaces
    "\u00a0": " ",      # Non-breaking space
    "\u2003": " ",      # Em space
    "\u2002": " ",      # En space
    "\u2009": " ",      # Thin space
    "\u200a": " ",      # Hair space
    "\u202f": " ",      # Narrow no-break space
    "\u205f": " ",      # Medium mathematical space
    
    # Bullets and symbols
    "\u2022": "-",      # • (bullet)
    "\u2023": "-",      # ‣ (triangular bullet)
    "\u2043": "-",      # ⁃ (hyphen bullet)
    "\u25e6": "-",      # ◦ (white bullet)
    
    # Other common replacements
    "\u2212": "-",      # − (minus sign)
    "\u2010": "-",      # ‐ (hyphen)
    "\u2011": "-",      # ‑ (non-breaking hyphen)
    "\u00ad": "",       # Soft hyphen (invisible)
    "\u200b": "",       # Zero-width space
    "\u200c": "",       # Zero-width non-joiner
    "\u200d": "",       # Zero-width joiner
    "\ufeff": "",       # BOM / zero-width no-break space
    
    # Fractions (common in AI text)
    "\u00bd": "1/2",    # ½
    "\u00bc": "1/4",    # ¼
    "\u00be": "3/4",    # ¾
    "\u2153": "1/3",    # ⅓
    "\u2154": "2/3",    # ⅔
    
    # Other symbols
    "\u00d7": "x",      # × (multiplication sign)
    "\u00f7": "/",      # ÷ (division sign)
    "\u2192": "->",     # → (right arrow)
    "\u2190": "<-",     # ← (left arrow)
    "\u2194": "<->",    # ↔ (left-right arrow)
    "\u2713": "",       # ✓ (check mark)
    "\u2714": "",       # ✔ (heavy check mark)
    "\u2717": "",       # ✗ (ballot x)
    "\u2718": "",       # ✘ (heavy ballot x)
}

# Build translation table for fast replacement
_TRANSLATION_TABLE = str.maketrans(AI_REPLACEMENTS)


def clean_ai_text(text: Optional[str]) -> Optional[str]:
    """
    Remove common AI-generated text artifacts.
    
    This replaces Unicode characters commonly used by LLMs with their
    plain ASCII equivalents.
    
    Args:
        text: Input text that may contain AI artifacts
        
    Returns:
        Cleaned text with ASCII equivalents, or None if input was None
    """
    if text is None:
        return None
    
    if not text:
        return text
    
    # Apply character replacements
    cleaned = text.translate(_TRANSLATION_TABLE)
    
    # Normalize multiple spaces
    cleaned = re.sub(r' +', ' ', cleaned)
    
    # Normalize multiple dashes (common in AI text: "high--performance")
    cleaned = re.sub(r'-{2,}', '-', cleaned)
    
    # Strip leading/trailing whitespace
    cleaned = cleaned.strip()
    
    return cleaned


def detect_ai_artifacts(text: str) -> list[tuple[str, int, str]]:
    """
    Detect AI-generated text artifacts in a string.
    
    Useful for debugging and understanding what characters are present.
    
    Args:
        text: Input text to analyze
        
    Returns:
        List of tuples: (character, position, unicode_name)
    """
    artifacts = []
    for i, char in enumerate(text):
        if char in AI_REPLACEMENTS:
            try:
                import unicodedata
                name = unicodedata.name(char, f"U+{ord(char):04X}")
            except ValueError:
                name = f"U+{ord(char):04X}"
            artifacts.append((char, i, name))
    return artifacts


def has_ai_artifacts(text: str) -> bool:
    """Check if text contains any common AI artifacts."""
    return any(char in AI_REPLACEMENTS for char in text)


def truncate_summary(
    text: str,
    max_chars: int = 500,
    target_chars: int = 400,
) -> str:
    """
    Truncate AI summary to fit within database limits.
    
    Tries to truncate at sentence boundaries for cleaner output.
    Falls back to word boundaries if no sentence break found.
    
    Args:
        text: The summary text to truncate
        max_chars: Absolute maximum (DB column limit)
        target_chars: Preferred length if truncation needed
        
    Returns:
        Truncated text, guaranteed <= max_chars
    """
    if not text or len(text) <= max_chars:
        return text
    
    # If slightly over target but within max, accept it
    if len(text) <= max_chars:
        return text
    
    # Try to find a good sentence break point
    # Look for sentence endings (. ! ?) near target
    search_start = max(0, target_chars - 50)
    search_end = min(len(text), target_chars + 50)
    search_region = text[search_start:search_end]
    
    # Find last sentence break in search region
    last_break = -1
    for i, char in enumerate(search_region):
        if char in '.!?' and i + search_start < target_chars + 30:
            last_break = i
    
    if last_break > 0:
        # Found a sentence break - use it
        truncated = text[:search_start + last_break + 1].strip()
        if len(truncated) <= max_chars:
            return truncated
    
    # No good sentence break - truncate at word boundary
    truncated = text[:target_chars]
    
    # Find last space to avoid cutting mid-word
    last_space = truncated.rfind(' ')
    if last_space > target_chars - 50:  # Don't go too far back
        truncated = truncated[:last_space]
    
    # Add ellipsis if we truncated
    truncated = truncated.rstrip('.,;:!? ') + '...'
    
    # Final safety check
    if len(truncated) > max_chars:
        truncated = truncated[:max_chars - 3] + '...'
    
    return truncated
