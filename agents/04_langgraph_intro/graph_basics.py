"""
04_langgraph_intro/graph_basics.py

Introduction to LangGraph: building workflows as graphs.
This example shows StateGraph, nodes, edges - without agents yet.

Key concept: Workflows as directed graphs with state.
"""

from typing import TypedDict, Literal
from pathlib import Path

from decouple import Config, RepositoryEnv
from langgraph.graph import StateGraph, END
from openai import OpenAI


# =============================================================================
# STATE - The data that flows through the graph
# =============================================================================

class AnalysisState(TypedDict):
    """
    State is a TypedDict that flows through the graph.
    Each node can read and update this state.
    """
    ticker: str
    earnings_text: str
    sentiment: str        # "positive", "negative", "neutral"
    key_points: list[str]
    recommendation: str
    final_report: str


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


def call_llm(prompt: str) -> str:
    """Helper to call the LLM."""
    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content


# =============================================================================
# NODES - Functions that transform state
# =============================================================================

def analyze_sentiment(state: AnalysisState) -> dict:
    """
    Node 1: Analyze sentiment of earnings text.
    Returns updates to merge into state.
    """
    print("  [Node] analyze_sentiment")

    prompt = f"""Analyze the sentiment of this earnings summary.
Respond with exactly one word: positive, negative, or neutral.

Text: {state['earnings_text']}

Sentiment:"""

    sentiment = call_llm(prompt).strip().lower()
    # Ensure valid sentiment
    if sentiment not in ["positive", "negative", "neutral"]:
        sentiment = "neutral"

    return {"sentiment": sentiment}


def extract_key_points(state: AnalysisState) -> dict:
    """
    Node 2: Extract key points from earnings.
    """
    print("  [Node] extract_key_points")

    prompt = f"""Extract 3 key points from this earnings summary.
Return as a simple numbered list.

Text: {state['earnings_text']}

Key points:"""

    response = call_llm(prompt)
    # Simple parsing - split by newlines and clean
    points = [p.strip() for p in response.split("\n") if p.strip()][:3]

    return {"key_points": points}


def make_recommendation(state: AnalysisState) -> dict:
    """
    Node 3: Make a recommendation based on sentiment.
    This demonstrates conditional logic within a node.
    """
    print(f"  [Node] make_recommendation (sentiment: {state['sentiment']})")

    sentiment = state["sentiment"]

    if sentiment == "positive":
        rec = "CONSIDER BUYING - Positive earnings indicate strong performance"
    elif sentiment == "negative":
        rec = "CONSIDER SELLING - Negative earnings suggest challenges"
    else:
        rec = "HOLD - Neutral earnings, wait for more data"

    return {"recommendation": rec}


def generate_report(state: AnalysisState) -> dict:
    """
    Node 4: Generate final report combining all analysis.
    """
    print("  [Node] generate_report")

    report = f"""
EARNINGS ANALYSIS REPORT: {state['ticker']}
{'='*50}

SENTIMENT: {state['sentiment'].upper()}

KEY POINTS:
{chr(10).join(f'  • {p}' for p in state['key_points'])}

RECOMMENDATION: {state['recommendation']}
"""
    return {"final_report": report}


# =============================================================================
# BUILD THE GRAPH
# =============================================================================

def build_graph():
    """
    Build a LangGraph workflow for earnings analysis.

    Graph structure:
        analyze_sentiment
              ↓
        extract_key_points
              ↓
        make_recommendation
              ↓
        generate_report
              ↓
             END
    """
    # Create graph with state schema
    workflow = StateGraph(AnalysisState)

    # Add nodes
    workflow.add_node("analyze_sentiment", analyze_sentiment)
    workflow.add_node("extract_key_points", extract_key_points)
    workflow.add_node("make_recommendation", make_recommendation)
    workflow.add_node("generate_report", generate_report)

    # Add edges (define flow)
    workflow.set_entry_point("analyze_sentiment")
    workflow.add_edge("analyze_sentiment", "extract_key_points")
    workflow.add_edge("extract_key_points", "make_recommendation")
    workflow.add_edge("make_recommendation", "generate_report")
    workflow.add_edge("generate_report", END)

    # Compile to runnable graph
    return workflow.compile()


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("LangGraph Basics: Earnings Analysis Workflow")
    print("=" * 50)
    print(f"Using model: {MODEL}\n")

    # Sample earnings text
    earnings_text = """
    Apple reported Q4 2024 earnings that beat analyst expectations.
    Revenue grew 8% year-over-year to $89.5 billion, driven by strong
    iPhone and Services performance. The company announced a $110 billion
    stock buyback program. However, China sales declined 2% amid increased
    competition. Management provided optimistic guidance for Q1 2025.
    """

    # Initial state
    initial_state: AnalysisState = {
        "ticker": "AAPL",
        "earnings_text": earnings_text.strip(),
        "sentiment": "",
        "key_points": [],
        "recommendation": "",
        "final_report": "",
    }

    # Build and run the graph
    print("Building graph...")
    graph = build_graph()

    print("\nRunning workflow:")
    print("-" * 50)

    # Invoke the graph
    final_state = graph.invoke(initial_state)

    # Show result
    print("\n" + "=" * 50)
    print(final_state["final_report"])


if __name__ == "__main__":
    main()
