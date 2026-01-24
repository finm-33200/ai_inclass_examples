# 04 - LangGraph Introduction

This example introduces **LangGraph** - a framework for building workflows as directed graphs. We start with the fundamentals before building agents.

## Why LangGraph?

After building agents from scratch (examples 01-03), you understand the mechanics. LangGraph provides:
- **State management** - Built-in handling of data flow
- **Conditional routing** - Declarative branching logic
- **Observability** - LangSmith integration for debugging
- **Persistence** - Checkpointing and recovery

## What You'll Learn

1. **StateGraph** - Define workflow state with TypedDict
2. **Nodes** - Functions that transform state
3. **Edges** - Connections between nodes
4. **Conditional edges** - Routing based on state values

## The Files

| File | Concept |
|------|---------|
| `graph_basics.py` | Linear workflow with nodes and edges |
| `conditional_routing.py` | Branching based on conditions |

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run

```bash
# Basic workflow
python graph_basics.py

# Conditional routing
python conditional_routing.py
```

## Core Concepts

### 1. State (TypedDict)

```python
class AnalysisState(TypedDict):
    ticker: str
    sentiment: str
    recommendation: str
```

State is a dictionary that flows through the graph. Each node can read and update it.

### 2. Nodes (Functions)

```python
def analyze_sentiment(state: AnalysisState) -> dict:
    # Read from state
    text = state["earnings_text"]

    # Do work
    sentiment = analyze(text)

    # Return updates (merged into state)
    return {"sentiment": sentiment}
```

Nodes are just functions: take state, return updates.

### 3. Edges (Flow)

```python
workflow.add_edge("analyze_sentiment", "make_recommendation")
```

Edges define the order of execution.

### 4. Conditional Edges (Branching)

```python
def route_on_alert(state) -> str:
    if state["should_alert"]:
        return "send_alert"
    return "no_alert"

workflow.add_conditional_edges(
    "check_condition",
    route_on_alert,
    {"send_alert": "send_alert", "no_alert": "no_alert"}
)
```

Routing functions decide which node runs next.

## Example Output: graph_basics.py

```
Building graph...

Running workflow:
--------------------------------------------------
  [Node] analyze_sentiment
  [Node] extract_key_points
  [Node] make_recommendation (sentiment: positive)
  [Node] generate_report

==================================================

EARNINGS ANALYSIS REPORT: AAPL
==================================================

SENTIMENT: POSITIVE

KEY POINTS:
  • Revenue grew 8% YoY to $89.5B
  • $110B stock buyback announced
  • Strong iPhone and Services performance

RECOMMENDATION: CONSIDER BUYING - Positive earnings indicate strong performance
```

## Example Output: conditional_routing.py

```
--- Test 1 ---
  [Node] check_condition: AAPL at $185.0
    → Condition 'above 180.0': True
  [Node] send_alert - TRIGGERED!
  Result: 🚨 ALERT: AAPL is $185.0, above your threshold of $180.0

--- Test 2 ---
  [Node] check_condition: MSFT at $375.0
    → Condition 'above 380.0': False
  [Node] no_alert - Condition not met
  Result: ✓ MSFT at $375.0 - no alert needed
```

## Graph Visualization

### graph_basics.py
```
┌─────────────────────┐
│  analyze_sentiment  │
└──────────┬──────────┘
           ↓
┌─────────────────────┐
│ extract_key_points  │
└──────────┬──────────┘
           ↓
┌─────────────────────┐
│ make_recommendation │
└──────────┬──────────┘
           ↓
┌─────────────────────┐
│   generate_report   │
└──────────┬──────────┘
           ↓
          END
```

### conditional_routing.py
```
┌─────────────────────┐
│   check_condition   │
└──────────┬──────────┘
           ↓
    [route_on_alert]
      ↙         ↘
┌──────────┐  ┌──────────┐
│send_alert│  │ no_alert │
└────┬─────┘  └────┬─────┘
     ↘          ↙
         END
```

## Why This Matters for Agents

In the next example, we'll use these concepts to build agents:
- **State** holds conversation history
- **Nodes** are LLM calls and tool executions
- **Conditional edges** decide: "use tool" or "return answer"

The ReAct loop from example 01 becomes a graph!

## Try It

1. **Add a node**: Insert a "format_output" node before END
2. **Multiple conditions**: Route based on sentiment (positive/negative/neutral)
3. **Parallel nodes**: Use `add_node` with same target to run nodes in parallel
4. **Error handling**: Add a node that handles errors and routes to recovery
