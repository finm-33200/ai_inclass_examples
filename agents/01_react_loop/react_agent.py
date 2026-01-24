"""
01_react_loop/react_agent.py

The simplest possible agent: a ReAct (Reason + Act) loop from scratch.
No frameworks, just the raw mechanics of how agents work.

This is the foundation for understanding all agent frameworks.
"""

import json
from pathlib import Path

from decouple import Config, RepositoryEnv
from openai import OpenAI


# =============================================================================
# TOOLS - Define what the agent can do
# =============================================================================

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_stock_price",
            "description": "Get the current price of a stock. Returns price in USD.",
            "parameters": {
                "type": "object",
                "properties": {
                    "ticker": {
                        "type": "string",
                        "description": "Stock ticker symbol (e.g., AAPL, MSFT, GOOGL)",
                    }
                },
                "required": ["ticker"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": "Perform a mathematical calculation. Use for any math operations.",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "Math expression to evaluate (e.g., '100 * 1.05', '250 / 4')",
                    }
                },
                "required": ["expression"],
            },
        },
    },
]

# Simulated stock data (in production, use yfinance or similar)
STOCK_DATA = {
    "AAPL": 178.50,
    "MSFT": 378.90,
    "GOOGL": 141.20,
    "AMZN": 178.25,
    "NVDA": 875.30,
    "TSLA": 248.50,
    "META": 505.75,
}


def get_stock_price(ticker: str) -> str:
    """Get stock price (simulated)."""
    ticker = ticker.upper()
    if ticker in STOCK_DATA:
        return json.dumps({"ticker": ticker, "price": STOCK_DATA[ticker], "currency": "USD"})
    return json.dumps({"error": f"Unknown ticker: {ticker}"})


def calculate(expression: str) -> str:
    """Safely evaluate a math expression."""
    allowed = set("0123456789+-*/.(). ")
    if not all(c in allowed for c in expression):
        return json.dumps({"error": "Invalid characters in expression"})
    try:
        result = eval(expression)
        return json.dumps({"result": result})
    except Exception as e:
        return json.dumps({"error": str(e)})


def execute_tool(name: str, arguments: dict) -> str:
    """Dispatch tool calls to the appropriate function."""
    if name == "get_stock_price":
        return get_stock_price(arguments["ticker"])
    elif name == "calculate":
        return calculate(arguments["expression"])
    return json.dumps({"error": f"Unknown tool: {name}"})


# =============================================================================
# THE REACT LOOP - The core of how agents work
# =============================================================================


def run_agent(client: OpenAI, model: str, user_message: str, max_iterations: int = 10):
    """
    The ReAct loop: Think -> Act -> Observe -> Repeat

    This is the fundamental pattern behind all agent frameworks.
    LangGraph, AutoGen, CrewAI - they all implement this loop.
    """
    messages = [
        {
            "role": "system",
            "content": """You are a helpful financial assistant. Use the available tools
to answer questions accurately. Always show your reasoning.""",
        },
        {"role": "user", "content": user_message},
    ]

    print(f"User: {user_message}\n")
    print("=" * 60)

    for iteration in range(max_iterations):
        # THINK: Ask the LLM what to do next
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=TOOLS,
        )

        message = response.choices[0].message

        # Check if the agent wants to ACT (use tools)
        if message.tool_calls:
            # Add assistant message to history
            messages.append(message)

            print(f"\n[Step {iteration + 1}] Agent decides to use tools:")

            for tool_call in message.tool_calls:
                name = tool_call.function.name
                args = json.loads(tool_call.function.arguments)

                print(f"  Action: {name}({args})")

                # ACT: Execute the tool
                result = execute_tool(name, args)

                print(f"  Observation: {result}")

                # OBSERVE: Add result to conversation
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": result,
                    }
                )
        else:
            # No more tools needed - agent has final answer
            print(f"\n[Final Answer]")
            print(message.content)
            return message.content

    print("\n[Warning] Max iterations reached")
    return None


# =============================================================================
# MAIN
# =============================================================================


def main():
    # Load API key
    repo_root = Path(__file__).resolve().parents[2]
    config = Config(RepositoryEnv(repo_root / ".env"))

    # Use OpenRouter with a cheap model
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=config("OPENROUTER_API_KEY"),
    )

    # Try different models - uncomment to compare:
    # model = "meta-llama/llama-3.1-8b-instruct"  # Very cheap ($0.02/1M) - often flawed!
    model = "openai/gpt-4o-mini"              # Cheap, reliable ($0.15/1M)
    # model = "google/gemini-2.0-flash-exp:free" # Free tier
    # model = "anthropic/claude-3-haiku"        # Cheap Claude ($0.25/1M)

    print(f"Using model: {model}\n")

    # Test the agent with a finance question
    question = "What's the price of AAPL? If I buy 100 shares, how much would that cost?"

    run_agent(client, model, question)


if __name__ == "__main__":
    main()
