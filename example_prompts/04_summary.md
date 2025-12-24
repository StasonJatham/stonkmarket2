# SUMMARY Prompt Example

> **Note:** This file documents what gets sent to OpenAI. The "System Instructions" and "User Prompt" sections below are the **exact text** from [`app/services/openai_client.py`](../app/services/openai_client.py).

---

## System Instructions

*From `INSTRUCTIONS[TaskType.SUMMARY]` in openai_client.py:*

```text
You explain complex businesses in simple terms.

RULES:
- Maximum 400 characters
- Simple language anyone can understand
- Focus on what the company actually does
- No jargon or complex terms
- Two or three sentences max
```

---

## User Prompt (Context)

```text
Stock: MSFT
Company: Microsoft Corporation
Sector: Technology
Business: Microsoft develops and licenses software, services, devices, and solutions. The company offers operating systems, productivity applications, cloud computing services, and gaming products.

Full Description:
Microsoft Corporation develops and supports software, services, devices and solutions worldwide. The Productivity and Business Processes segment offers office, exchange, SharePoint, Microsoft Teams, office 365 Security and Compliance, Microsoft viva, and Microsoft 365 copilot; and LinkedIn solutions. This segment also provides Dynamics 365, a set of intelligent, cloud-based applications across ERP, CRM, power apps, and power automate; and on-premises ERP and CRM applications. The Intelligent Cloud segment offers Azure and other cloud services; SQL and Windows Servers, Visual Studio, System Center, and related Client Access Licenses, as well as nuance and GitHub; and Enterprise Services including Enterprise Support Services, Industry Solutions, and Microsoft Consulting Services. The More Personal Computing segment offers Windows, including windows OEM licensing and other non-volume licensing of the Windows operating system; Windows Commercial comprising volume licensing of the Windows operating system, Windows cloud services, and other Windows commercial offerings; patent licensing; and Windows Internet of Things; and devices, including Microsoft Surface, HoloLens, and PC accessories.
```

---

## Example Output

```text
Microsoft makes the software that runs most office computers and businesses. They sell Windows, Office apps like Word and Excel, and run a massive cloud service called Azure that powers websites and apps worldwide. They also own Xbox, LinkedIn, and GitHub.
```

---

## Token Estimates

| Component         | Estimated Tokens |
| ----------------- | ---------------- |
| Instructions      | ~70 tokens       |
| User Prompt       | ~350 tokens      |
| **Total Input**   | **~420 tokens**  |
| Max Output        | 200 tokens (800 with GPT-5 reasoning) |

---

## Notes

The SUMMARY task is unique because it includes the full company description from Yahoo Finance, which can be quite long (up to 2000+ characters). The dynamic token calculation ensures we don't exceed the context window even with lengthy descriptions.
