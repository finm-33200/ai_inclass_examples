# 04 - Multi-Benchmark Suite

Run all API-compatible benchmarks together and create a comprehensive model comparison.

## What You'll Learn

- Running multiple benchmarks in a single evaluation
- Aggregating results across different capability dimensions
- Creating a "model scorecard" visualization

## Why Run Multiple Benchmarks?

Single benchmarks can be misleading. A model might excel at math (GSM8K) but struggle with factual accuracy (TruthfulQA). Running a suite gives you:

1. **Balanced assessment** - see strengths and weaknesses
2. **Task-specific insights** - which model for which use case
3. **Confidence** - consistent patterns across benchmarks

## Benchmarks Included

| Benchmark | Tests | Finance Relevance |
|-----------|-------|-------------------|
| **GSM8K** | Math reasoning | Calculations, quantitative analysis |
| **MMLU (CoT)** | Knowledge (3 subjects) | Economics, macroeconomics, math |
| **TruthfulQA (gen)** | Factual accuracy | Avoiding hallucinations |

**Note**: HellaSwag and TruthfulQA MC require logprobs which API models don't support.
For those benchmarks, use local HuggingFace models.

## How to Run

### Step 1: Run full suite

```bash
# Run all benchmarks with 50 samples each (default)
./01_run_suite.sh

# Run with more samples
./01_run_suite.sh 100
```

This takes longer but runs everything in one go.

### Step 2: Aggregate results

```bash
python 02_aggregate_results.py
```

Creates a bar chart and summary table.

## Cost Estimate

| Samples | Est. Total Cost (3 models) |
|---------|---------------------------|
| 50 each | ~$0.10-0.20 |
| 100 each | ~$0.20-0.40 |

## Output

The aggregation script creates:

1. **Bar chart** (`benchmark_comparison.png`) - Visual comparison across dimensions
2. **Summary table** - Average scores and rankings
3. **Per-benchmark breakdown** - Detailed results

## Choosing a Model

Based on your use case:

| Use Case | Prioritize |
|----------|-----------|
| Financial calculations | GSM8K |
| Research/analysis | MMLU + TruthfulQA |
| General assistant | Balanced (all) |

## Try It

1. Run the full suite with 50 samples
2. Look at the chart - which model has the most balanced performance?
3. Which model would you choose for a financial chatbot? Why?
4. What additional benchmarks might be useful for finance?

## Resources

- [lm-eval GitHub](https://github.com/EleutherAI/lm-evaluation-harness)
- [Open LLM Leaderboard](https://huggingface.co/spaces/open-llm-leaderboard/open_llm_leaderboard) (uses similar benchmarks)
