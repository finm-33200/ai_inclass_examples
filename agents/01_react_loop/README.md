# 01 - ReAct Loop: The Simplest Agent

This example builds an agent from scratch using just the OpenAI SDK. No frameworks, no abstractions—just the raw mechanics of how agents work.

## Why Start Here?

Every agent framework (LangGraph, AutoGen, CrewAI) implements the same core pattern: **ReAct** (Reason + Act). By building it ourselves, we understand what these frameworks actually do.

## What You'll Learn

1. **The Agent Loop**: `Think → Act → Observe → Repeat`
2. **How tools work**: LLM requests a tool, we execute it, feed result back
3. **Why agents are different from function calling**: The loop continues until the LLM decides it has enough information
4. **Message history**: How conversation state accumulates

## The Pattern

```
User asks question
    ↓
┌─────────────────────────┐
│  LLM thinks about it    │  ← REASON
└─────────────────────────┘
    ↓
Does LLM want to use a tool?
    ↓ Yes                     ↓ No
┌──────────────┐         ┌──────────────┐
│ Execute tool │  ← ACT  │ Return answer│
└──────────────┘         └──────────────┘
    ↓
┌──────────────────────────┐
│ Add result to messages   │  ← OBSERVE
└──────────────────────────┘
    ↓
    └──────→ Back to "LLM thinks"
```

## Setup

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Ensure OPENROUTER_API_KEY is in your .env at repo root
```

## Run

```bash
python react_agent.py
```

## Example Output

```
Using model: openai/gpt-4o-mini

User: What's the price of AAPL? If I buy 100 shares, how much would that cost?

============================================================

[Step 1] Agent decides to use tools:
  Action: get_stock_price({'ticker': 'AAPL'})
  Observation: {"ticker": "AAPL", "price": 178.5, "currency": "USD"}

[Step 2] Agent decides to use tools:
  Action: calculate({'expression': '178.5 * 100'})
  Observation: {"result": 17850.0}

[Final Answer]
The current price of AAPL (Apple Inc.) is $178.50. If you buy 100 shares, it would cost you $17,850.00.
```

## Key Code Sections

### Tool Definitions
```python
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_stock_price",
            "description": "Get the current price of a stock...",
            ...
        }
    }
]
```

### The Loop
```python
for iteration in range(max_iterations):
    response = client.chat.completions.create(model=model, messages=messages, tools=TOOLS)

    if message.tool_calls:
        # Execute tools, add results to messages
        ...
    else:
        # Done - return final answer
        return message.content
```

## Try It

1. **Change the model**: Uncomment different models in `main()` to compare quality vs cost
2. **Add a tool**: Create a `get_company_news(ticker)` function
3. **Ask complex questions**: "Compare AAPL and MSFT prices. Which is more expensive per share?"
4. **Break it**: Remove `max_iterations` safeguard—what happens with an infinite loop?

## Models to Try

| Model | Cost | Quality |
|-------|------|---------|
| `meta-llama/llama-3.1-8b-instruct` | $0.02/1M | Good for simple tasks |
| `openai/gpt-4o-mini` | $0.15/1M | Reliable, handles complex reasoning |
| `google/gemini-2.0-flash-exp:free` | Free | Good for testing |
| `anthropic/claude-3-haiku` | $0.25/1M | Fast, good quality |

## Finance Application

This agent pattern is the foundation for:
- Portfolio analysis bots
- Market data Q&A systems
- Automated research assistants
- Trading signal generators

The key insight: the LLM decides which tools to use and when it has enough information. We just provide the tools and execute them.
