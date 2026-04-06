# 07_chronogpt_benchmark - Research-Grade ChronoGPT Evaluation

Full benchmark evaluation of ChronoGPT models. This demonstrates research-grade model evaluation on custom architectures.

## What You'll Learn

- Benchmarking custom model architectures
- Comparing models across time periods
- Analyzing temporal consistency in evaluations

## Key Concepts

- **Custom Evaluation**: Since ChronoGPT uses a non-standard architecture, we implement custom evaluation
- **Temporal Benchmarking**: Compare the same benchmark across different model time periods
- **Baseline Comparison**: Compare ChronoGPT against standard models

## Prerequisites

- Python 3.10+
- ~8GB RAM (loading multiple models)
- ChronoGPT models downloaded

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Running the Examples

```bash
# Run benchmark suite on ChronoGPT
chmod +x 01_run_benchmark_suite.sh
./01_run_benchmark_suite.sh

# Analyze and visualize results
python 02_analyze_results.py

# Compare temporal versions
python 03_temporal_benchmark.py
```

## Benchmark Tasks

| Task | Description | What it Tests |
|------|-------------|---------------|
| GSM8K | Grade school math | Reasoning, arithmetic |
| HellaSwag | Common sense | World knowledge |
| TruthfulQA | Factuality | Resistance to misconceptions |

## Understanding Results

ChronoGPT models are optimized for temporal consistency, not benchmark performance. Expect:
- Lower scores than frontier models (GPT-4, Claude)
- Interesting differences between time periods
- Strong performance on period-appropriate questions

## Research Applications

1. **Backtesting Studies**: Use period-appropriate models for historical simulations
2. **Temporal Analysis**: How did LLM knowledge evolve over time?
3. **Counterfactual Research**: What advice would an AI have given in 2000?

## Notes

- Benchmarks run locally - no API costs
- Results are deterministic (temperature=0)
- First run downloads model weights (~4GB per model)
