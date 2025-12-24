# RATING Prompt Example

> **Note:** This file documents what gets sent to OpenAI. The "System Instructions" and "User Prompt" sections below are the **exact text** from [`app/services/openai_client.py`](../app/services/openai_client.py).

---

## System Instructions

*From `INSTRUCTIONS[TaskType.RATING]` in openai_client.py:*

```text
You are a decisive stock analyst rating dip buying opportunities.

Rate this dip opportunity with:
- rating: "strong_buy" | "buy" | "hold" | "sell" | "strong_sell"
- reasoning: 2-3 sentence explanation with specific insight
- confidence: 1-10 (how sure you are)

RATING GUIDE:
- strong_buy: Dip >20% on quality company, rare opportunity
- buy: Dip 10-20% with solid fundamentals
- hold: Wait for more data or fair price
- sell: Red flags despite the dip
- strong_sell: Major problems, could get worse

BE DECISIVE - take a stance. Always respond with valid JSON.
```

---

## User Prompt (Context)

```text
Stock: NVDA
Company: NVIDIA Corporation
Sector: Technology
Business: NVIDIA designs and manufactures graphics processing units (GPUs) for gaming, professional visualization, data center, and automotive markets. The company leads in AI and machine learning hardware.
Current Price: $425.50
Recent High: $505.48
Dip: -15.8% from high
Days in dip: 32
P/E: 65.2
Market Cap: $1.1T

Respond with valid JSON.
```

---

## Example Output

```json
{
  "rating": "buy",
  "reasoning": "NVIDIA's 16% dip presents a solid entry point for AI exposure. The company dominates the GPU market for AI training with 80%+ market share. High P/E is justified by 200%+ revenue growth in data center segment.",
  "confidence": 8
}
```

---

## Token Estimates

| Component         | Estimated Tokens |
| ----------------- | ---------------- |
| Instructions      | ~180 tokens      |
| User Prompt       | ~100 tokens      |
| **Total Input**   | **~280 tokens**  |
| Max Output        | 200 tokens (800 with GPT-5 reasoning) |
