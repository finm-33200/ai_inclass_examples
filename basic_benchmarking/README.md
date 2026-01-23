# Basic Benchmarking

This topic covers how to benchmark and evaluate LLM outputs, progressing from manual approaches to evaluation frameworks.

## Why Benchmarking Matters

In finance applications, LLM quality directly impacts decision-making. You need to:
- **Measure performance**: latency, cost, throughput
- **Evaluate quality**: correctness, completeness, compliance
- **Compare models**: which model gives the best cost/quality tradeoff?

## Progression

| Example | What You Learn |
|---------|----------------|
| [01_latency_and_cost](01_latency_and_cost/) | Manual benchmarking: timing, token counting, cost calculation |
| [02_promptfoo_intro](02_promptfoo_intro/) | Declarative evaluation with YAML configs |
| [03_llm_as_judge](03_llm_as_judge/) | Automated quality scoring using LLM-as-judge |
| 04_promptfoo_finance_evals | Finance-specific assertions *(future)* |
| 05_langsmith_tracing | Observability for agents *(future)* |
| 06_langsmith_evals | LangSmith evaluation workflows *(future)* |

## Framework Comparison

| Tool | Approach | Best For |
|------|----------|----------|
| **Manual Python** | Roll your own | Learning fundamentals |
| **promptfoo** | YAML configs, CLI | Quick A/B tests, CI/CD |
| **LangSmith** | Tracing + evals | Agent debugging, production monitoring |

## Prerequisites

- Python 3.10+
- OpenAI API key (set in repo root `.env`)
- Node.js (for promptfoo examples)
