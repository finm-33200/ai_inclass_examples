# 03 - Multi-Tool Agent

This example demonstrates an agent with **multiple tools** that it selects based on the question. The LLM decides which tools are needed and in what order.

## Why Multiple Tools?

Real-world agents need diverse capabilities:
- Market data (prices, volumes)
- Fundamental analysis (P/E, market cap)
- News and sentiment
- Calculations

## What You'll Learn

1. **Tool selection**: How LLMs choose from multiple options
2. **Tool descriptions matter**: Better descriptions → better choices
3. **Multi-tool queries**: One question, multiple tool calls
4. **Routing logic**: The LLM figures out dependencies

## Available Tools

| Tool | Purpose | When Used |
|------|---------|-----------|
| `get_stock_price` | Current price + change | "What's the price of X?" |
| `get_company_info` | Fundamentals (P/E, sector) | "Tell me about X" |
| `get_historical_prices` | Price history | "How has X performed?" |
| `get_news` | Recent headlines | "What's happening with X?" |
| `calculate_return` | % return calculation | "If I bought at X..." |

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run

```bash
python finance_agent.py
```

## Example: Single Tool

```
Question: What's the current price of NVDA?

[Step 1] Tools selected:
  → get_stock_price({"ticker": "NVDA"})
    Result: {"ticker": "NVDA", "price": 875.30, "change": 15.20...}

[Analysis]
NVIDIA (NVDA) is currently trading at $875.30, up $15.20 (+1.74%) today.
```

## Example: Multiple Tools

```
Question: Compare AAPL and JPM - which has a better P/E ratio?

[Step 1] Tools selected:
  → get_company_info({"ticker": "AAPL"})
    Result: {"ticker": "AAPL", "pe_ratio": 28.5, "sector": "Technology"...}
  → get_company_info({"ticker": "JPM"})
    Result: {"ticker": "JPM", "pe_ratio": 11.2, "sector": "Financial"...}

[Analysis]
JPM has a better (lower) P/E ratio of 11.2 compared to AAPL's 28.5.
JPM is in Financial sector, while AAPL is in Technology.
```

## Example: Complex Analysis

```
Question: Analyze MSFT: price, fundamentals, news, and 7-day trend.

[Step 1] Tools selected:
  → get_stock_price({"ticker": "MSFT"})
  → get_company_info({"ticker": "MSFT"})
  → get_news({"ticker": "MSFT"})
  → get_historical_prices({"ticker": "MSFT", "days": 7})

[Analysis]
Microsoft (MSFT) comprehensive analysis:
- Current: $378.90 (-$2.10 / -0.55%)
- Fundamentals: P/E 35.2, Market Cap $2.9T, Technology sector
- 7-day trend: Started at $372, now $378 (+1.6%)
- News: Azure growth 29% YoY, Copilot adoption accelerating...
```

## Key Code Pattern

```python
# Tool descriptions guide selection
{
    "name": "get_stock_price",
    "description": "Get current price. Use for market data.",  # ← Key!
}

# The LLM reads descriptions and chooses
response = client.chat.completions.create(
    model=model,
    messages=messages,
    tools=TOOLS,  # All 5 tools available
)
# LLM returns which tools to call based on question
```

## Tool Selection Tips

Good descriptions:
- "Get current price and daily change for a stock. Use for market data."

Bad descriptions:
- "Get stock price" (too vague)
- "Returns JSON with price" (implementation detail, not purpose)

## Finance Application

Multi-tool agents power:
- Research assistants (gather diverse data)
- Trading analysis (price + news + fundamentals)
- Portfolio reports (combine multiple data sources)
- Due diligence workflows

## Try It

1. **Add a new tool**: Create `compare_stocks(ticker1, ticker2)` that calls other tools
2. **Improve descriptions**: Make tool descriptions more specific, see if selection improves
3. **Test edge cases**: Ask about an unknown ticker - how does error handling work?
4. **Complex queries**: "Which tech stock has the best P/E and most positive news?"
