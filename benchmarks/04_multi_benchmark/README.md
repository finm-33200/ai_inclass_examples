# 04 - Multi-Benchmark Suite (Quick Demo)

Run multiple benchmarks together - a quick demo showing combined evaluation.

## What You'll Learn

- Running multiple benchmarks in a single lm-eval call
- Understanding different benchmark types (math reasoning vs knowledge)

## Quick Demo Configuration

This example is configured for fast in-class demonstration:

| Setting | Value |
|---------|-------|
| Benchmarks | GSM8K, TruthfulQA, MMLU (econometrics) |
| Samples | 5 each |
| Models | 3 (Llama 3.1, Gemma 2, Qwen 2.5) |
| **Runtime** | ~20-30 seconds |

## Benchmarks Included

| Benchmark | Tests | Finance Relevance |
|-----------|-------|-------------------|
| **GSM8K** | Math reasoning | Calculations, quantitative analysis |
| **TruthfulQA** | Factual accuracy | Avoiding hallucinations |
| **MMLU (CoT)** | Knowledge | Economics, statistics |

**Note**: Additional benchmarks available in `05_parallel_models/`.

## How to Run

### Step 1: Run benchmarks

```bash
# Run quick demo (default 5 samples)
./01_run_suite.sh

# Run with more samples
./01_run_suite.sh 20
```

### Step 2: Aggregate results

```bash
python 02_aggregate_results.py
```

## Cost Estimate

This quick demo costs ~$0.002 (negligible).

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

1. Run the quick demo to verify setup
2. Modify the script to add more benchmarks (e.g., `truthfulqa_gen`)
3. For parallel model execution, see `05_parallel_models/`

## Resources

- [lm-eval GitHub](https://github.com/EleutherAI/lm-evaluation-harness)
- [Open LLM Leaderboard](https://huggingface.co/spaces/open-llm-leaderboard/open_llm_leaderboard) (uses similar benchmarks)
