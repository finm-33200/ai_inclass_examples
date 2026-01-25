# 03 - MMLU Subset Benchmark

Test model knowledge across finance-relevant subjects using MMLU (Massive Multitask Language Understanding).

## What You'll Learn

- Running MMLU on specific subject subsets
- Understanding multiple-choice benchmarking with API models
- Analyzing per-subject performance

## Why MMLU?

MMLU is a comprehensive knowledge benchmark covering 57 subjects from STEM to humanities. For finance applications, relevant subjects include:

- Economics (micro/macro)
- Econometrics
- Mathematics
- Computer Science
- Business Ethics

## Subjects We Test

| Subject | # Questions | Relevance |
|---------|-------------|-----------|
| `econometrics` | ~114 | Statistical methods for finance |
| `high_school_macroeconomics` | ~390 | Economic fundamentals |
| `high_school_microeconomics` | ~238 | Market behavior |
| `college_mathematics` | ~100 | Quantitative foundations |
| `college_computer_science` | ~100 | Tech skills for quant work |
| `professional_accounting` | ~282 | Financial statements, auditing |

## API Model Limitation

**Important**: Standard MMLU tasks (`mmlu_*`) require logprobs which API models don't support.

We use the **Chain-of-Thought variant** (`mmlu_cot_llama_*`) which is generation-based:

- Model generates reasoning then outputs final answer
- Uses the Llama-style CoT prompt format
- Works with any API model that supports chat completions
- Results may differ slightly from published benchmarks using logprob scoring

This is fine for learning and comparison, but for research-grade benchmarks, use local HuggingFace models with the standard `mmlu_*` tasks.

## How to Run

### Step 1: Run benchmarks

```bash
# Run all subjects (default 50 samples each)
./01_run_mmlu.sh

# Run with more samples
./01_run_mmlu.sh 100
```

### Step 2: Analyze results

```bash
python 02_analyze_results.py
```

## Cost Estimate

| Samples/Subject | Subjects | Est. Cost (3 models) |
|-----------------|----------|---------------------|
| 50 | 6 | ~$0.02-0.04 |
| 100 | 6 | ~$0.04-0.08 |

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

1. Run with 50 samples to verify setup
2. Which subjects do models perform best on?
3. Is there a model that's particularly good at economics?
4. Try adding `business_ethics` to the subject list

## Resources

- [MMLU Paper](https://arxiv.org/abs/2009.03300)
- [MMLU GitHub](https://github.com/hendrycks/test)
- [lm-eval MMLU tasks](https://github.com/EleutherAI/lm-evaluation-harness/tree/main/lm_eval/tasks/mmlu)
