# 01 - lm-evaluation-harness Introduction

Get started with EleutherAI's lm-evaluation-harness - the industry standard for LLM benchmarking.

## What You'll Learn

- Installing lm-eval
- Running your first benchmark via command line
- Using OpenRouter as the API backend
- Understanding output format

## Why This Framework?

lm-evaluation-harness is the **de facto standard** for LLM benchmarks. When you see benchmark numbers in papers, leaderboards, or model cards, they're almost always from this tool. Learning it means:

1. You can reproduce published results
2. Your benchmark numbers are directly comparable to others
3. You're using what hiring managers and reviewers expect

## Setup

```bash
# Install lm-eval
pip install lm-eval

# Verify installation
lm_eval --help
```

## Environment

Set your OpenRouter API key in the repo root `.env` file:

```bash
OPENROUTER_API_KEY=your-key-here
```

## How to Run

The simplest way - run 10 GSM8K examples:

```bash
# Load environment variables
set -a && source ../../.env && set +a

# Run the benchmark script
./01_first_benchmark.sh
```

Or run the command directly:

```bash
# Set API key
export OPENAI_API_KEY=$OPENROUTER_API_KEY

# Run benchmark
lm_eval --model openai-chat-completions \
    --model_args model=meta-llama/llama-3.1-8b-instruct,base_url=https://openrouter.ai/api/v1/chat/completions,num_concurrent=50,timeout=120,max_retries=3,tokenizer_backend=None \
    --tasks gsm8k \
    --limit 10 \
    --apply_chat_template \
    --output_path results/
```

## Command Breakdown

| Argument | Purpose |
|----------|---------|
| `--model openai-chat-completions` | Use OpenAI-compatible chat API |
| `--model_args model=...,base_url=...` | Model name and API endpoint |
| `num_concurrent=50` | Run 50 API requests in parallel (fast!) |
| `timeout=120` | 2-minute timeout per request |
| `max_retries=3` | Retry failed requests (handles rate limits) |
| `--tasks gsm8k` | Which benchmark to run |
| `--limit 10` | Only run 10 examples (for testing) |
| `--output_path results/` | Where to save JSON results |

## Understanding Output

lm-eval prints a results table:

```
|Tasks|Version|     Filter     |n-shot|  Metric   |   |Value|   |Stderr|
|-----|------:|----------------|-----:|-----------|---|----:|---|-----:|
|gsm8k|      3|flexible-extract|     5|exact_match|↑  |  0.5|±  |0.1667|
|     |       |strict-match    |     5|exact_match|↑  |  0.3|±  |0.1528|
```

- **exact_match**: Whether the extracted answer exactly matches the ground truth
- **Stderr**: Standard error (confidence interval)
- **n-shot**: Number of few-shot examples in prompt
- **↑**: Arrow indicates higher is better

### GSM8K Filters

GSM8K uses two scoring methods:

| Filter | Description |
|--------|-------------|
| **flexible-extract** | Parses the response to find the final numerical answer (more lenient) |
| **strict-match** | Requires the answer in a specific format like `#### 42` (stricter) |

The flexible filter typically gives higher scores since it's more forgiving of format variations.

## Cost

Running 10 examples costs approximately **$0.001-0.002** with cheap models.

## Try It

1. Run the script with `--limit 10` to verify setup
2. Try a different model: `google/gemma-2-9b-it`
3. Try a different task: `hellaswag`
4. Remove `--limit` to run the full benchmark (costs more)

## Next Steps

Once this works, move to [02_gsm8k](../02_gsm8k/) to compare multiple models.
