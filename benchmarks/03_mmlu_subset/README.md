# 03 - MMLU Subset Benchmark (Quick Demo)

Test model knowledge using MMLU (Massive Multitask Language Understanding) - a single-subject quick demo.

## What You'll Learn

- Running MMLU benchmarks via lm-eval
- Understanding multiple-choice benchmarking with API models
- Basic benchmark output format

## Quick Demo Configuration

This example is configured for fast in-class demonstration:

| Setting | Value |
|---------|-------|
| Subject | `econometrics` |
| Samples | 5 |
| Models | 3 (Llama 3.1, Gemma 2, Qwen 2.5) |
| **Runtime** | ~10-15 seconds |

## Why MMLU?

MMLU is a comprehensive knowledge benchmark covering 57 subjects from STEM to humanities. The `econometrics` subject tests statistical methods relevant to finance.

## API Model Limitation

**Important**: Standard MMLU tasks (`mmlu_*`) require logprobs which API models don't support.

We use the **Chain-of-Thought variant** (`mmlu_cot_llama_*`) which is generation-based:

- Model generates reasoning then outputs final answer
- Uses the Llama-style CoT prompt format
- Works with any API model that supports chat completions
- Results may differ slightly from published benchmarks using logprob scoring

This is fine for learning and comparison, but for research-grade benchmarks, use local HuggingFace models with the standard `mmlu_*` tasks.

## How to Run

### Step 1: Run benchmark

```bash
# Run quick demo (default 5 samples)
./01_run_mmlu.sh

# Run with more samples
./01_run_mmlu.sh 20
```

### Step 2: Analyze results

```bash
python 02_analyze_results.py
```

## Cost Estimate

This quick demo costs ~$0.001 (negligible).

## Understanding MMLU Format

Each MMLU question is multiple choice:

```
Question: The longest railway line in the world, the Trans-Siberian, connects
         St. Petersburg with which Russian city?
A) Irkutsk
B) Moscow
C) Vladivostok
D) Murmansk

Answer: C
```

## Output

Results saved to `results/` with per-subject breakdowns:
- Aggregate accuracy across all subjects
- Per-subject accuracy scores
- Comparison across models

## Try It

1. Run the quick demo to verify setup
2. Modify the script to add more subjects (e.g., `mmlu_cot_llama_high_school_macroeconomics`)
3. For parallel model execution, see `05_parallel_models/`

## Resources

- [MMLU Paper](https://arxiv.org/abs/2009.03300)
- [MMLU GitHub](https://github.com/hendrycks/test)
- [lm-eval MMLU tasks](https://github.com/EleutherAI/lm-evaluation-harness/tree/main/lm_eval/tasks/mmlu)
