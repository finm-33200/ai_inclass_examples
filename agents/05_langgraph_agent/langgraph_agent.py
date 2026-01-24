"""
05_langgraph_agent/langgraph_agent.py

ReAct agent built with LangGraph's prebuilt components.
Compares with the manual implementation from example 01.
"""

from pathlib import Path
from typing import Annotated

from decouple import Config, RepositoryEnv
from langchain_core.tools import tool
from langchain_core.messages import SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent


# =============================================================================
# SETUP
# =============================================================================

repo_root = Path(__file__).resolve().parents[2]
config = Config(RepositoryEnv(repo_root / ".env"))

# Use ChatOpenAI with OpenRouter
llm = ChatOpenAI(
    model="openai/gpt-4o-mini",
    base_url="https://openrouter.ai/api/v1",
    api_key=config("OPENROUTER_API_KEY"),
)


# =============================================================================
# TOOLS - Using LangChain's @tool decorator
# =============================================================================

STOCK_DATA = {
    "AAPL": {"price": 178.50, "name": "Apple Inc."},
    "MSFT": {"price": 378.90, "name": "Microsoft Corp."},
    "GOOGL": {"price": 141.20, "name": "Alphabet Inc."},
    "NVDA": {"price": 875.30, "name": "NVIDIA Corp."},
}


@tool
def get_stock_price(ticker: Annotated[str, "Stock ticker symbol (e.g., AAPL)"]) -> str:
    """Get the current price of a stock. Returns price in USD."""
    ticker = ticker.upper()
    if ticker in STOCK_DATA:
        data = STOCK_DATA[ticker]
        return f"{ticker} ({data['name']}) is trading at ${data['price']}"
    return f"Unknown ticker: {ticker}"


@tool
def calculate(expression: Annotated[str, "Math expression (e.g., '100 * 178.5')"]) -> str:
    """Perform a mathematical calculation."""
    allowed = set("0123456789+-*/.(). ")
    if not all(c in allowed for c in expression):
        return "Error: Invalid characters"
    try:
        result = eval(expression)
        return f"Result: {result}"
    except Exception as e:
        return f"Error: {e}"


# List of tools for the agent
tools = [get_stock_price, calculate]


# =============================================================================
# CREATE AGENT
# =============================================================================

# System message for the agent
SYSTEM_MESSAGE = SystemMessage(
    content="You are a helpful financial assistant. Use tools to answer questions accurately."
)


def create_agent():
    """
    Create a ReAct agent using LangGraph's prebuilt function.

    This is equivalent to the manual loop in example 01, but with:
    - Built-in state management
    - Automatic tool handling
    - Streaming support
    - Checkpointing support
    """
    return create_react_agent(
        llm,
        tools,
        prompt=SYSTEM_MESSAGE,
    )


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("LangGraph ReAct Agent")
    print("=" * 50)

    agent = create_agent()

    questions = [
        "What's the price of AAPL?",
        "If I buy 100 shares of NVDA, how much would that cost?",
    ]

    for question in questions:
        print(f"\nQuestion: {question}")
        print("-" * 40)

        # Stream the agent's response
        for event in agent.stream({"messages": [("user", question)]}):
            # Event contains updates to state
            for key, value in event.items():
                if key == "agent":
                    # Agent made a decision
                    msg = value["messages"][-1]
                    if hasattr(msg, "tool_calls") and msg.tool_calls:
                        for tc in msg.tool_calls:
                            print(f"  [Tool Call] {tc['name']}({tc['args']})")
                    elif hasattr(msg, "content") and msg.content:
                        print(f"\nAnswer: {msg.content}")
                elif key == "tools":
                    # Tool returned a result
                    msg = value["messages"][-1]
                    content = msg.content if len(msg.content) < 80 else msg.content[:80] + "..."
                    print(f"  [Tool Result] {content}")

        print()


if __name__ == "__main__":
    main()
