"""
06_function_calling/finance_tools.py

Finance-focused function calling example.
Tools for stock prices, returns, and economic data.
"""

import json
import random
from pathlib import Path

from decouple import Config, RepositoryEnv
from openai import OpenAI


# Define finance-related tools
TOOLS = [
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
                        "description": "Stock ticker symbol (e.g., AAPL, MSFT)",
                    }
                },
                "required": ["ticker"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_stock_info",
            "description": "Get basic information about a stock including market cap and P/E ratio",
            "parameters": {
                "type": "object",
                "properties": {
                    "ticker": {
                        "type": "string",
                        "description": "Stock ticker symbol",
                    }
                },
                "required": ["ticker"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculate_return",
            "description": "Calculate the percentage return between two prices",
            "parameters": {
                "type": "object",
                "properties": {
                    "start_price": {
                        "type": "number",
                        "description": "Starting price",
                    },
                    "end_price": {
                        "type": "number",
                        "description": "Ending price",
                    },
                },
                "required": ["start_price", "end_price"],
            },
        },
    },
]

# Simulated stock data (in production, use a real API like yfinance)
STOCK_DATA = {
    "AAPL": {"price": 178.50, "name": "Apple Inc.", "pe": 28.5, "market_cap": "2.8T"},
    "MSFT": {
        "price": 378.90,
        "name": "Microsoft Corp.",
        "pe": 35.2,
        "market_cap": "2.9T",
    },
    "GOOGL": {
        "price": 141.20,
        "name": "Alphabet Inc.",
        "pe": 24.1,
        "market_cap": "1.8T",
    },
    "AMZN": {
        "price": 178.25,
        "name": "Amazon.com Inc.",
        "pe": 62.3,
        "market_cap": "1.9T",
    },
    "NVDA": {"price": 875.30, "name": "NVIDIA Corp.", "pe": 65.8, "market_cap": "2.2T"},
}


def get_stock_price(ticker: str) -> str:
    """Get current stock price (simulated)."""
    ticker = ticker.upper()
    if ticker in STOCK_DATA:
        # Add some random variation to simulate real-time
        base_price = STOCK_DATA[ticker]["price"]
        price = base_price * (1 + random.uniform(-0.02, 0.02))
        return json.dumps({"ticker": ticker, "price": round(price, 2), "currency": "USD"})
    return json.dumps({"error": f"Unknown ticker: {ticker}"})


def get_stock_info(ticker: str) -> str:
    """Get stock information (simulated)."""
    ticker = ticker.upper()
    if ticker in STOCK_DATA:
        data = STOCK_DATA[ticker]
        return json.dumps(
            {
                "ticker": ticker,
                "name": data["name"],
                "price": data["price"],
                "pe_ratio": data["pe"],
                "market_cap": data["market_cap"],
            }
        )
    return json.dumps({"error": f"Unknown ticker: {ticker}"})


def calculate_return(start_price: float, end_price: float) -> str:
    """Calculate percentage return."""
    if start_price <= 0:
        return json.dumps({"error": "Start price must be positive"})
    pct_return = ((end_price - start_price) / start_price) * 100
    return json.dumps(
        {
            "start_price": start_price,
            "end_price": end_price,
            "return_pct": round(pct_return, 2),
        }
    )


def execute_tool(name: str, arguments: dict) -> str:
    """Execute a tool by name."""
    if name == "get_stock_price":
        return get_stock_price(arguments["ticker"])
    elif name == "get_stock_info":
        return get_stock_info(arguments["ticker"])
    elif name == "calculate_return":
        return calculate_return(arguments["start_price"], arguments["end_price"])
    else:
        return json.dumps({"error": f"Unknown tool: {name}"})


def main():
    # Load API key from .env at repository root
    repo_root = Path(__file__).resolve().parents[2]
    config = Config(RepositoryEnv(repo_root / ".env"))

    client = OpenAI(api_key=config("OPENAI_API_KEY"))

    # Finance question requiring multiple tools
    user_question = (
        "Compare AAPL and NVDA - which has a better P/E ratio? "
        "Also, if I bought AAPL at $150 and it's now at its current price, what's my return?"
    )

    messages = [
        {
            "role": "system",
            "content": """You are a financial analyst assistant. Use the available tools
            to get accurate data. Always cite the actual numbers from the tools.""",
        },
        {"role": "user", "content": user_question},
    ]

    print(f"User: {user_question}\n")

    # Tool use loop
    iteration = 0
    max_iterations = 10  # Prevent infinite loops

    while iteration < max_iterations:
        iteration += 1

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            tools=TOOLS,
        )

        message = response.choices[0].message

        if message.tool_calls:
            messages.append(message)

            print(f"[Iteration {iteration}] Tool calls:")
            for tool_call in message.tool_calls:
                name = tool_call.function.name
                args = json.loads(tool_call.function.arguments)
                print(f"  {name}({args})")

                result = execute_tool(name, args)
                print(f"    -> {result}")

                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": result,
                    }
                )
            print()
        else:
            print(f"Assistant:\n{message.content}")
            break


if __name__ == "__main__":
    main()
