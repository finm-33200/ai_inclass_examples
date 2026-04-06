# 06_chronogpt_exploration - Temporally-Consistent LLMs

Explore ChronoGPT, a collection of language models trained to maintain temporal consistency - each model only "knows" information available up to its training cutoff date.

## What You'll Learn

- Loading ChronoGPT models from HuggingFace
- Understanding temporal consistency in LLMs
- Comparing responses across different time periods

## Key Concepts

- **Temporal Consistency**: Models only reference knowledge available at their cutoff date
- **Chronological Training**: Separate models trained on data up to specific dates
- **Historical Analysis**: Ask questions about events using period-appropriate knowledge

## The ChronoGPT Paper

ChronoGPT addresses a fundamental issue: standard LLMs blend knowledge from different time periods, making them unreliable for historical analysis or backtesting.

**Paper**: [ChronoGPT: Chronologically Consistent Language Models](https://arxiv.org/abs/2502.01029)

**Key insight**: For finance applications like backtesting trading strategies, you need models that don't "look ahead" at future information.

## Available Models

| Model | Cutoff Date | Use Case |
|-------|-------------|----------|
| `chrono-gpt-instruct-v1-20001231` | Dec 2000 | Dot-com era analysis |
| `chrono-gpt-instruct-v1-20081231` | Dec 2008 | Financial crisis period |
| `chrono-gpt-instruct-v1-20201231` | Dec 2020 | Pre-ChatGPT era |
| `chrono-gpt-instruct-v1-20241231` | Dec 2024 | Most recent knowledge |

## Prerequisites

- Python 3.10+
- ~4GB RAM
- First run downloads model weights

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Running the Examples

```bash
# Basic ChronoGPT loading
python 01_load_chronogpt.py

# Compare 2000 vs 2024 responses
python 02_temporal_comparison.py

# Finance-specific knowledge tests
python 03_finance_questions.py
```

## Try It

1. Ask about events that happened between model cutoff dates
2. Compare investment advice across time periods
3. Test knowledge of companies that didn't exist in 2000

## Example Questions to Explore

- "What is the current state of AI research?" (compare 2000 vs 2024)
- "What are the best tech stocks to invest in?" (see how advice differs)
- "Describe the cryptocurrency market" (2000 model won't know about Bitcoin)
