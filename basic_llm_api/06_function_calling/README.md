# 06 - Function Calling (Tool Use)

Let the LLM decide when to call external functions/tools to get information.

## Why Function Calling?

LLMs have knowledge cutoffs and can't access real-time data. Function calling allows:
- **Real-time data**: Get current stock prices, weather, etc.
- **External actions**: Send emails, update databases
- **Computation**: Perform calculations the LLM shouldn't do
- **Agentic behavior**: LLM decides which tools to use and when

## What You'll Learn

- Defining tools for the LLM
- Handling tool calls in your code
- The tool use loop pattern
- Finance-specific tools (stock prices, FRED data)

## Setup

```bash
cd 06_function_calling
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Examples

### Simple Tools
```bash
python simple_tools.py
```

### Finance Tools
```bash
python finance_tools.py
```

## Key Concepts

### Defining Tools

Tools are defined as JSON schemas describing what they do and their parameters:

```python
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_stock_price",
            "description": "Get the current price of a stock",
            "parameters": {
                "type": "object",
                "properties": {
                    "ticker": {
                        "type": "string",
                        "description": "Stock ticker symbol (e.g., AAPL)"
                    }
                },
                "required": ["ticker"]
            }
        }
    }
]
```

### The Tool Use Loop

1. Send message with tools defined
2. LLM responds with tool calls (or final answer)
3. Execute the tool calls
4. Send results back to LLM
5. Repeat until LLM gives final answer

```python
while True:
    response = client.chat.completions.create(...)
    if response.choices[0].finish_reason == "tool_calls":
        # Execute tools and continue
    else:
        # Final answer
        break
```

### Tool Call Response

When the LLM wants to use a tool:
```python
tool_calls = response.choices[0].message.tool_calls
for call in tool_calls:
    name = call.function.name
    args = json.loads(call.function.arguments)
    result = execute_tool(name, args)
```

## Finance Application

The `finance_tools.py` example includes:
- `get_stock_price`: Simulated current stock price
- `calculate_return`: Compute return between two prices
- `get_fred_data`: Fetch economic data from FRED

## Try It

1. Ask a question that requires multiple tool calls
2. Add a new tool (e.g., currency conversion)
3. What happens if you ask something the tools can't answer?
4. Try: "Compare the YTD return of AAPL vs MSFT"
