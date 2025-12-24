# OpenAI Prompt Examples

This directory contains examples of the prompts sent to OpenAI for each task type.
Each file shows the complete prompt structure with sample stock data.

## Task Types

| Task       | File                | Purpose                                    |
| ---------- | ------------------- | ------------------------------------------ |
| `bio`      | [01_bio.md](01_bio.md)           | Dating-app style stock bio                 |
| `rating`   | [03_rating.md](03_rating.md)     | Buy/hold/sell rating with JSON output      |
| `summary`  | [04_summary.md](04_summary.md)   | Simple business description                |

## Prompt Structure

Each OpenAI API call consists of:

1. **System Instructions** - Task-specific rules and personality
2. **User Prompt** - Stock data formatted as key-value pairs

```text
┌─────────────────────────────────────────┐
│  SYSTEM INSTRUCTIONS                    │
│  (from INSTRUCTIONS dict)               │
│  - Rules for the AI                     │
│  - Output format requirements           │
│  - Personality/tone guidance            │
├─────────────────────────────────────────┤
│  USER PROMPT                            │
│  (from _build_prompt function)          │
│  - Stock: AAPL                          │
│  - Company: Apple Inc.                  │
│  - Sector: Technology                   │
│  - Current Price: $178.50               │
│  - Dip: -10.6% from high                │
│  - ... more context data                │
└─────────────────────────────────────────┘
```

## Token Limits

See [openai_client.py](../app/services/openai_client.py) for the `MODEL_LIMITS` configuration.

| Model        | Context Window | Max Output | Input $/1M | Output $/1M |
| ------------ | -------------- | ---------- | ---------- | ----------- |
| gpt-5-mini   | 1,047,576      | 65,536     | $0.10      | $0.40       |
| gpt-5        | 1,047,576      | 65,536     | $2.00      | $8.00       |
| gpt-4o-mini  | 128,000        | 16,384     | $0.15      | $0.60       |
| gpt-4o       | 128,000        | 16,384     | $2.50      | $10.00      |

## Dynamic Token Calculation

The `_calculate_safe_output_tokens()` function ensures we never exceed limits:

```python
# Estimates input tokens from text length (~4 chars = 1 token)
input_estimate = estimate_tokens(instructions) + estimate_tokens(prompt)

# Calculates available output space
available = context_window - input_estimate - 100  # 100 token buffer

# Applies GPT-5 reasoning multiplier (4x for internal "thinking" tokens)
safe_output = min(desired * 4, available, max_output)
```
