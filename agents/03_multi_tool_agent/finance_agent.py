"""
03_multi_tool_agent/finance_agent.py

Agent with multiple finance tools demonstrating tool selection and routing.
The LLM decides which tools to use based on the question.

Key concept: Tool descriptions guide the LLM's choices.
"""

import json
import random
from datetime import datetime, timedelta
from pathlib import Path

from decouple import Config, RepositoryEnv
from openai import OpenAI


# =============================================================================
# MULTIPLE TOOLS - The LLM chooses which to use
# =============================================================================

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_stock_price",
            "description": "Get the current price and daily change for a stock. Use this for current market data.",
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
            "name": "get_company_info",
            "description": "Get fundamental information about a company including sector, market cap, P/E ratio. Use for company research.",
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
            "name": "get_historical_prices",
            "description": "Get historical price data for a stock over a time period. Use for trend analysis.",
            "parameters": {
                "type": "object",
                "properties": {
                    "ticker": {
                        "type": "string",
                        "description": "Stock ticker symbol",
                    },
                    "days": {
                        "type": "integer",
                        "description": "Number of days of history (max 30)",
                    },
                },
                "required": ["ticker", "days"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_news",
            "description": "Get recent news headlines for a stock. Use for sentiment and event analysis.",
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
            "description": "Calculate percentage return between two prices. Use for performance calculations.",
            "parameters": {
                "type": "object",
                "properties": {
                    "start_price": {"type": "number", "description": "Starting price"},
                    "end_price": {"type": "number", "description": "Ending price"},
                },
                "required": ["start_price", "end_price"],
            },
        },
    },
]


# =============================================================================
# SIMULATED DATA - In production, use real APIs
# =============================================================================

STOCK_DATA = {
    "AAPL": {"price": 178.50, "change": 1.25, "name": "Apple Inc.", "sector": "Technology", "pe": 28.5, "market_cap": "2.8T"},
    "MSFT": {"price": 378.90, "change": -2.10, "name": "Microsoft Corp.", "sector": "Technology", "pe": 35.2, "market_cap": "2.9T"},
    "GOOGL": {"price": 141.20, "change": 0.85, "name": "Alphabet Inc.", "sector": "Technology", "pe": 24.1, "market_cap": "1.8T"},
    "AMZN": {"price": 178.25, "change": -1.50, "name": "Amazon.com Inc.", "sector": "Consumer Cyclical", "pe": 62.3, "market_cap": "1.9T"},
    "NVDA": {"price": 875.30, "change": 15.20, "name": "NVIDIA Corp.", "sector": "Technology", "pe": 65.8, "market_cap": "2.2T"},
    "JPM": {"price": 195.40, "change": 0.30, "name": "JPMorgan Chase", "sector": "Financial", "pe": 11.2, "market_cap": "560B"},
    "V": {"price": 275.80, "change": 1.10, "name": "Visa Inc.", "sector": "Financial", "pe": 29.8, "market_cap": "550B"},
}

NEWS_DATA = {
    "AAPL": [
        "Apple announces record iPhone sales in Asia",
        "Apple Vision Pro sees strong developer interest",
        "Warren Buffett increases Apple stake",
    ],
    "MSFT": [
        "Microsoft Azure revenue grows 29% YoY",
        "Microsoft Copilot adoption accelerates",
        "Microsoft announces $10B stock buyback",
    ],
    "NVDA": [
        "NVIDIA H100 demand exceeds supply",
        "NVIDIA partners with major cloud providers",
        "Analysts raise NVIDIA price targets",
    ],
}


# =============================================================================
# TOOL IMPLEMENTATIONS
# =============================================================================


def get_stock_price(ticker: str) -> str:
    ticker = ticker.upper()
    if ticker in STOCK_DATA:
        data = STOCK_DATA[ticker]
        # Add slight randomization for realism
        price = data["price"] * (1 + random.uniform(-0.01, 0.01))
        return json.dumps({
            "ticker": ticker,
            "price": round(price, 2),
            "change": data["change"],
            "change_pct": round(data["change"] / data["price"] * 100, 2),
        })
    return json.dumps({"error": f"Unknown ticker: {ticker}"})


def get_company_info(ticker: str) -> str:
    ticker = ticker.upper()
    if ticker in STOCK_DATA:
        data = STOCK_DATA[ticker]
        return json.dumps({
            "ticker": ticker,
            "name": data["name"],
            "sector": data["sector"],
            "pe_ratio": data["pe"],
            "market_cap": data["market_cap"],
        })
    return json.dumps({"error": f"Unknown ticker: {ticker}"})


def get_historical_prices(ticker: str, days: int) -> str:
    ticker = ticker.upper()
    if ticker not in STOCK_DATA:
        return json.dumps({"error": f"Unknown ticker: {ticker}"})

    days = min(days, 30)
    base_price = STOCK_DATA[ticker]["price"]

    # Generate simulated historical data
    prices = []
    for i in range(days, 0, -1):
        date = (datetime.now() - timedelta(days=i)).strftime("%Y-%m-%d")
        # Random walk with slight upward bias
        variation = random.uniform(-0.03, 0.035)
        price = base_price * (1 + variation * (days - i) / days)
        prices.append({"date": date, "price": round(price, 2)})

    return json.dumps({
        "ticker": ticker,
        "days": days,
        "prices": prices,
        "start_price": prices[0]["price"],
        "end_price": prices[-1]["price"],
    })


def get_news(ticker: str) -> str:
    ticker = ticker.upper()
    news = NEWS_DATA.get(ticker, [f"No recent news for {ticker}"])
    return json.dumps({"ticker": ticker, "headlines": news})


def calculate_return(start_price: float, end_price: float) -> str:
    if start_price <= 0:
        return json.dumps({"error": "Start price must be positive"})
    return_pct = ((end_price - start_price) / start_price) * 100
    return json.dumps({
        "start_price": start_price,
        "end_price": end_price,
        "return_pct": round(return_pct, 2),
        "return_absolute": round(end_price - start_price, 2),
    })


def execute_tool(name: str, args: dict) -> str:
    """Route tool calls to implementations."""
    tools = {
        "get_stock_price": lambda a: get_stock_price(a["ticker"]),
        "get_company_info": lambda a: get_company_info(a["ticker"]),
        "get_historical_prices": lambda a: get_historical_prices(a["ticker"], a["days"]),
        "get_news": lambda a: get_news(a["ticker"]),
        "calculate_return": lambda a: calculate_return(a["start_price"], a["end_price"]),
    }
    if name in tools:
        return tools[name](args)
    return json.dumps({"error": f"Unknown tool: {name}"})


# =============================================================================
# AGENT
# =============================================================================


def run_agent(client: OpenAI, model: str, question: str, max_iterations: int = 10) -> str:
    """Run the multi-tool agent."""
    messages = [
        {
            "role": "system",
            "content": """You are a financial analyst assistant with access to market data tools.
Use the appropriate tools to answer questions thoroughly. You can use multiple tools
to gather comprehensive information. Always cite specific data from the tools.""",
        },
        {"role": "user", "content": question},
    ]

    print(f"Question: {question}\n")
    print("-" * 60)

    for iteration in range(max_iterations):
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=TOOLS,
        )

        message = response.choices[0].message

        if message.tool_calls:
            messages.append(message)

            print(f"\n[Step {iteration + 1}] Tools selected:")
            for tool_call in message.tool_calls:
                name = tool_call.function.name
                args = json.loads(tool_call.function.arguments)
                print(f"  → {name}({json.dumps(args)})")

                result = execute_tool(name, args)
                # Show truncated result
                result_preview = result[:100] + "..." if len(result) > 100 else result
                print(f"    Result: {result_preview}")

                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": result,
                })
        else:
            print(f"\n[Analysis]\n{message.content}")
            return message.content

    return "[Max iterations reached]"


# =============================================================================
# MAIN - Demonstrate multi-tool selection
# =============================================================================


def main():
    repo_root = Path(__file__).resolve().parents[2]
    config = Config(RepositoryEnv(repo_root / ".env"))

    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=config("OPENROUTER_API_KEY"),
    )

    model = "openai/gpt-4o-mini"
    print(f"Multi-Tool Finance Agent (using {model})\n")
    print("=" * 60)

    # Questions that require different tool combinations
    questions = [
        # Single tool
        "What's the current price of NVDA?",

        # Multiple tools for comparison
        "Compare AAPL and JPM - which has a better P/E ratio and what are their sectors?",

        # Complex analysis requiring multiple tools
        "Analyze MSFT: get the current price, company fundamentals, recent news, and 7-day price trend.",
    ]

    for i, question in enumerate(questions, 1):
        print(f"\n{'='*60}")
        print(f"EXAMPLE {i}")
        print("=" * 60)
        run_agent(client, model, question)
        print()


if __name__ == "__main__":
    main()
