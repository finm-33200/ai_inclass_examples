# 05 - LangGraph ReAct Agent

This example rebuilds the ReAct agent from example 01 using **LangGraph's prebuilt components**. Compare the code to see what LangGraph handles for you.

## Why Use LangGraph for Agents?

Example 01 showed the raw mechanics. LangGraph provides:
- **Automatic tool handling** - No manual dispatch
- **Streaming** - See agent "thinking" in real-time
- **Checkpointing** - Save/restore agent state
- **Error recovery** - Built-in retry logic

## What You'll Learn

1. **@tool decorator** - Define tools with type hints
2. **create_react_agent** - One-liner agent creation
3. **Streaming** - Watch the agent work step-by-step
4. **Comparison** - Manual vs framework trade-offs

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run

```bash
python langgraph_agent.py
```

## Example Output

```
LangGraph ReAct Agent
==================================================

Question: What's the price of AAPL?
----------------------------------------
  [Tool Call] get_stock_price({'ticker': 'AAPL'})
  [Tool Result] AAPL (Apple Inc.) is trading at $178.5...

Answer: The current price of AAPL (Apple Inc.) is $178.50.

Question: If I buy 100 shares of NVDA, how much would that cost?
----------------------------------------
  [Tool Call] get_stock_price({'ticker': 'NVDA'})
  [Tool Result] NVDA (NVIDIA Corp.) is trading at $875.3...
  [Tool Call] calculate({'expression': '100 * 875.3'})
  [Tool Result] Result: 87530.0...

Answer: If you buy 100 shares of NVDA at $875.30, it would cost $87,530.00.
```

## Code Comparison

### Example 01 (Manual)

```python
# Define tools as JSON schema
TOOLS = [{"type": "function", "function": {"name": "get_stock_price", ...}}]

# Manual dispatch
def execute_tool(name, args):
    if name == "get_stock_price":
        return get_stock_price(args["ticker"])
    ...

# Manual loop
while True:
    response = client.chat.completions.create(model, messages, tools)
    if response.tool_calls:
        for tc in response.tool_calls:
            result = execute_tool(tc.name, tc.args)
            messages.append(tool_result)
    else:
        break
```

### Example 05 (LangGraph)

```python
# Define tools with decorator
@tool
def get_stock_price(ticker: str) -> str:
    """Get stock price."""  # Docstring becomes description!
    ...

# One-liner agent
agent = create_react_agent(llm, [get_stock_price, calculate])

# Stream results
for event in agent.stream({"messages": [("user", question)]}):
    ...
```

## Tool Definition with @tool

```python
from langchain_core.tools import tool
from typing import Annotated

@tool
def get_stock_price(
    ticker: Annotated[str, "Stock ticker symbol (e.g., AAPL)"]
) -> str:
    """Get the current price of a stock. Returns price in USD."""
    # The docstring becomes the tool description
    # Annotated types provide parameter descriptions
    ...
```

## When to Use Each Approach

**Use Manual (Example 01) when:**
- Learning how agents work
- Need maximum control
- Simple, one-off agents
- Minimal dependencies

**Use LangGraph (Example 05) when:**
- Production applications
- Need streaming, checkpointing
- Complex multi-agent systems
- Want LangSmith observability

## Finance Application

The same agent pattern powers:
- Trading assistants with real-time data
- Research copilots with multiple data sources
- Portfolio analyzers with calculation tools

## Try It

1. **Add a tool**: Create `get_company_info(ticker)` and add to tools list
2. **Enable tracing**: Set `LANGCHAIN_API_KEY` and `LANGCHAIN_TRACING_V2=true`
3. **Compare models**: Change the model string to compare quality
4. **Custom prompt**: Modify `state_modifier` to change agent personality
