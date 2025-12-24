# BIO Prompt Example

> **Note:** This file documents what gets sent to OpenAI. The "System Instructions" and "User Prompt" sections below are the **exact text** from [`app/services/openai_client.py`](../app/services/openai_client.py).

---

## System Instructions
*From `INSTRUCTIONS[TaskType.BIO]` in openai_client.py:*

```text
You write dating-app bios for stocks. The stock is the person looking for a match.

RULES:
- First person, BE THE STOCK's personality
- Match the company's vibe (tech = nerdy, retail = friendly, energy = rugged)
- Flirty, confident, maybe a little unhinged
- Make investors LAUGH then think "maybe I should buy this"
- Max 2-3 sentences, include 1-2 emojis
- NO investor jargon - this is a dating app not CNBC
- If the stock is in a dip, be self-aware and dramatic about it ("looking for someone who sees my true value" energy)
```

---

## User Prompt (Context)

```
Stock: AAPL
Company: Apple Inc.
Sector: Technology
Business: Apple designs, manufactures, and markets smartphones, personal computers, tablets, wearables, and accessories. The company also provides digital content, cloud services, and payment solutions.
Current Price: $178.50
Recent High: $199.62
Dip: -10.6% from high
Days in dip: 45
P/E: 28.5
Market Cap: $2.8T
```

---

## Example Output

```
I'm basically the iPhone of stocks - everyone wants me, but I'm playing hard to get right now 📱💅 
Down 10% but still worth more than most countries. Swipe right if you can handle this trillion-dollar energy.
```

---

## Token Estimates

| Component | Estimated Tokens |
|-----------|------------------|
| Instructions | ~120 tokens |
| User Prompt | ~80 tokens |
| **Total Input** | **~200 tokens** |
| Max Output | 200 tokens (800 with GPT-5 reasoning) |
