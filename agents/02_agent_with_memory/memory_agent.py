"""
02_agent_with_memory/memory_agent.py

Agent with conversation memory - maintains context across multiple turns.
This is how chatbots remember what you talked about earlier.

Key concept: The message list IS the memory. Each turn adds to it.
"""

import json
from pathlib import Path

from decouple import Config, RepositoryEnv
from openai import OpenAI


# =============================================================================
# TOOLS - Same as before, but now used across a conversation
# =============================================================================

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_stock_price",
            "description": "Get the current price of a stock",
            "parameters": {
                "type": "object",
                "properties": {
                    "ticker": {"type": "string", "description": "Stock ticker symbol"}
                },
                "required": ["ticker"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_portfolio_value",
            "description": "Get the current value of the user's portfolio",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": "Perform a mathematical calculation",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {"type": "string", "description": "Math expression"}
                },
                "required": ["expression"],
            },
        },
    },
]

# Simulated data
STOCK_DATA = {
    "AAPL": 178.50,
    "MSFT": 378.90,
    "GOOGL": 141.20,
    "NVDA": 875.30,
}

# Simulated user portfolio
PORTFOLIO = {
    "AAPL": 50,   # 50 shares
    "MSFT": 20,   # 20 shares
    "NVDA": 10,   # 10 shares
}


def get_stock_price(ticker: str) -> str:
    ticker = ticker.upper()
    if ticker in STOCK_DATA:
        return json.dumps({"ticker": ticker, "price": STOCK_DATA[ticker]})
    return json.dumps({"error": f"Unknown ticker: {ticker}"})


def get_portfolio_value() -> str:
    holdings = []
    total = 0
    for ticker, shares in PORTFOLIO.items():
        price = STOCK_DATA.get(ticker, 0)
        value = shares * price
        total += value
        holdings.append({"ticker": ticker, "shares": shares, "price": price, "value": value})
    return json.dumps({"holdings": holdings, "total_value": total})


def calculate(expression: str) -> str:
    allowed = set("0123456789+-*/.(). ")
    if not all(c in allowed for c in expression):
        return json.dumps({"error": "Invalid expression"})
    try:
        return json.dumps({"result": eval(expression)})
    except Exception as e:
        return json.dumps({"error": str(e)})


def execute_tool(name: str, args: dict) -> str:
    if name == "get_stock_price":
        return get_stock_price(args["ticker"])
    elif name == "get_portfolio_value":
        return get_portfolio_value()
    elif name == "calculate":
        return calculate(args["expression"])
    return json.dumps({"error": f"Unknown tool: {name}"})


# =============================================================================
# AGENT WITH MEMORY
# =============================================================================


class PortfolioAgent:
    """
    Agent that maintains conversation memory.

    The key insight: memory is just the messages list.
    Each turn, we append new messages. The LLM sees everything.
    """

    def __init__(self, client: OpenAI, model: str):
        self.client = client
        self.model = model
        # This list IS the memory
        self.messages = [
            {
                "role": "system",
                "content": """You are a portfolio assistant. You help users understand their
investments. You have access to their portfolio and can look up stock prices.
Be conversational and remember what the user has asked about previously.""",
            }
        ]

    def chat(self, user_message: str, max_iterations: int = 10) -> str:
        """Process a user message and return the response."""
        # Add user message to memory
        self.messages.append({"role": "user", "content": user_message})

        for _ in range(max_iterations):
            response = self.client.chat.completions.create(
                model=self.model,
                messages=self.messages,
                tools=TOOLS,
            )

            message = response.choices[0].message

            if message.tool_calls:
                self.messages.append(message)

                for tool_call in message.tool_calls:
                    name = tool_call.function.name
                    args = json.loads(tool_call.function.arguments)
                    result = execute_tool(name, args)

                    print(f"  [Tool] {name}({args}) -> {result}")

                    self.messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "content": result,
                        }
                    )
            else:
                # Add assistant response to memory
                self.messages.append({"role": "assistant", "content": message.content})
                return message.content

        return "[Max iterations reached]"

    def get_memory_size(self) -> int:
        """Return the number of messages in memory."""
        return len(self.messages)

    def summarize_memory(self) -> str:
        """Show what's in memory (for debugging)."""
        summary = []
        for msg in self.messages:
            role = msg["role"] if isinstance(msg, dict) else msg.role
            if role == "system":
                summary.append("system: [instructions]")
            elif role == "user":
                content = msg["content"] if isinstance(msg, dict) else msg.content
                summary.append(f"user: {content[:50]}...")
            elif role == "assistant":
                content = msg.get("content") if isinstance(msg, dict) else getattr(msg, "content", None)
                if content:
                    summary.append(f"assistant: {content[:50]}...")
                else:
                    summary.append("assistant: [tool calls]")
            elif role == "tool":
                summary.append("tool: [result]")
        return "\n".join(summary)


# =============================================================================
# INTERACTIVE REPL
# =============================================================================


def main():
    repo_root = Path(__file__).resolve().parents[2]
    config = Config(RepositoryEnv(repo_root / ".env"))

    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=config("OPENROUTER_API_KEY"),
    )

    model = "openai/gpt-4o-mini"
    print(f"Portfolio Agent (using {model})")
    print("Type 'quit' to exit, 'memory' to see conversation history")
    print("-" * 50)

    agent = PortfolioAgent(client, model)

    # Demo conversation
    demo_questions = [
        "What's my portfolio worth?",
        "How much AAPL do I have?",
        "What if AAPL goes up 10%? How much would my AAPL position be worth?",
    ]

    for question in demo_questions:
        print(f"\nYou: {question}")
        response = agent.chat(question)
        print(f"\nAgent: {response}")
        print(f"  [Memory: {agent.get_memory_size()} messages]")

    print("\n" + "=" * 50)
    print("Notice how the agent remembered:")
    print("- Your portfolio holdings from the first question")
    print("- The AAPL position when calculating the 10% gain")
    print("=" * 50)

    # Interactive mode
    print("\nNow try your own questions (or 'quit' to exit):\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if not user_input:
            continue
        if user_input.lower() == "quit":
            break
        if user_input.lower() == "memory":
            print("\n--- Memory Contents ---")
            print(agent.summarize_memory())
            print("-" * 25)
            continue

        response = agent.chat(user_input)
        print(f"\nAgent: {response}")
        print(f"  [Memory: {agent.get_memory_size()} messages]\n")


if __name__ == "__main__":
    main()
