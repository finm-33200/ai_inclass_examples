"""
04_langgraph_intro/conditional_routing.py

LangGraph with conditional edges - routing based on state.
This is key for building agents that make decisions.
"""

from typing import TypedDict, Literal
from pathlib import Path

from decouple import Config, RepositoryEnv
from langgraph.graph import StateGraph, END
from openai import OpenAI


# =============================================================================
# STATE
# =============================================================================

class AlertState(TypedDict):
    """State for a stock alert system."""
    ticker: str
    price: float
    threshold: float
    alert_type: str  # "above" or "below"
    should_alert: bool
    message: str


# =============================================================================
# SETUP
# =============================================================================

repo_root = Path(__file__).resolve().parents[2]
config = Config(RepositoryEnv(repo_root / ".env"))

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=config("OPENROUTER_API_KEY"),
)

MODEL = "openai/gpt-4o-mini"


# =============================================================================
# NODES
# =============================================================================

def check_condition(state: AlertState) -> dict:
    """Check if alert condition is met."""
    print(f"  [Node] check_condition: {state['ticker']} at ${state['price']}")

    price = state["price"]
    threshold = state["threshold"]
    alert_type = state["alert_type"]

    if alert_type == "above":
        should_alert = price > threshold
    else:
        should_alert = price < threshold

    print(f"    → Condition '{alert_type} {threshold}': {should_alert}")
    return {"should_alert": should_alert}


def send_alert(state: AlertState) -> dict:
    """Generate and 'send' an alert message."""
    print("  [Node] send_alert - TRIGGERED!")

    direction = "above" if state["alert_type"] == "above" else "below"
    message = f"🚨 ALERT: {state['ticker']} is ${state['price']}, {direction} your threshold of ${state['threshold']}"

    return {"message": message}


def no_alert(state: AlertState) -> dict:
    """No alert needed."""
    print("  [Node] no_alert - Condition not met")
    return {"message": f"✓ {state['ticker']} at ${state['price']} - no alert needed"}


# =============================================================================
# CONDITIONAL ROUTING
# =============================================================================

def route_on_alert(state: AlertState) -> Literal["send_alert", "no_alert"]:
    """
    Routing function - returns the name of the next node.
    This is how LangGraph does conditional branching.
    """
    if state["should_alert"]:
        return "send_alert"
    return "no_alert"


# =============================================================================
# BUILD GRAPH WITH CONDITIONAL EDGE
# =============================================================================

def build_graph():
    """
    Build a graph with conditional routing.

    Graph structure:
        check_condition
              ↓
        [conditional]
         ↙        ↘
    send_alert   no_alert
         ↘        ↙
            END
    """
    workflow = StateGraph(AlertState)

    # Add nodes
    workflow.add_node("check_condition", check_condition)
    workflow.add_node("send_alert", send_alert)
    workflow.add_node("no_alert", no_alert)

    # Entry point
    workflow.set_entry_point("check_condition")

    # Conditional edge - this is the key concept!
    workflow.add_conditional_edges(
        "check_condition",      # From this node
        route_on_alert,         # Use this function to decide
        {                       # Map return values to node names
            "send_alert": "send_alert",
            "no_alert": "no_alert",
        }
    )

    # Both branches end
    workflow.add_edge("send_alert", END)
    workflow.add_edge("no_alert", END)

    return workflow.compile()


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("LangGraph Conditional Routing: Stock Alert System")
    print("=" * 50)

    graph = build_graph()

    # Test cases
    test_cases = [
        {"ticker": "AAPL", "price": 185.00, "threshold": 180.00, "alert_type": "above"},
        {"ticker": "MSFT", "price": 375.00, "threshold": 380.00, "alert_type": "above"},
        {"ticker": "NVDA", "price": 850.00, "threshold": 900.00, "alert_type": "below"},
    ]

    for i, case in enumerate(test_cases, 1):
        print(f"\n--- Test {i} ---")

        initial_state: AlertState = {
            **case,
            "should_alert": False,
            "message": "",
        }

        final_state = graph.invoke(initial_state)
        print(f"  Result: {final_state['message']}")


if __name__ == "__main__":
    main()
