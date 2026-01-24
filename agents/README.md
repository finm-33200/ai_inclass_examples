# Agents

This section covers building AI agents from first principles through production-ready frameworks. We start with raw mechanics (pure SDK) and progress to LangGraph for complex workflows.

## Why This Progression?

**Examples 01-03** use the OpenAI SDK directly with OpenRouter. You'll see exactly how agent loops work: the think-act-observe cycle, message management, and tool dispatch. No magic, no abstractions.

**Examples 04-08** introduce LangGraph, which handles state management, tool execution, and observability. Once you understand the fundamentals, frameworks make you more productive.

**Example 08** ("Flawed Code") is a minimal Claude Code clone. It's intentionally simple (~200 lines) and uses cheap models—so it makes mistakes. That's the point: understanding both capabilities and limitations.

## Model Philosophy

All examples use **cheap models via OpenRouter** (Llama 3.1 8B, Gemini Flash, etc.). This:
- Keeps costs near zero for experimentation
- Shows agents work even with small models
- Demonstrates quality/cost tradeoffs
- Explains why it's called "Flawed Code"

## Directory Overview

| # | Directory | What You Learn | Framework |
|---|-----------|----------------|-----------|
| 01 | `01_react_loop/` | ReAct pattern from scratch | Pure SDK |
| 02 | `02_agent_with_memory/` | Conversation context, memory | Pure SDK |
| 03 | `03_multi_tool_agent/` | Tool selection, routing | Pure SDK |
| 04 | `04_langgraph_intro/` | StateGraph, nodes, edges | LangGraph |
| 05 | `05_langgraph_agent/` | LangGraph ReAct agent | LangGraph |
| 06 | `06_code_execution/` | Sandboxed Python execution | LangGraph |
| 07 | `07_file_operations/` | File system tools | LangGraph |
| 08 | `08_flawed_code/` | Minimal Claude Code clone | LangGraph |

## Prerequisites

- Python 3.10+
- `OPENROUTER_API_KEY` in your `.env` file
- Optional: `LANGCHAIN_API_KEY` for LangSmith tracing (examples 04+)

## Quick Start

```bash
# From repo root
cd agents/01_react_loop
pip install -r requirements.txt
python react_agent.py
```

## Finance Applications

Each example includes finance-relevant hooks:
- Stock price lookups and market data
- Portfolio tracking with conversation memory
- Multi-source financial data aggregation
- Automated data analysis pipelines
- Code generation for quant workflows
