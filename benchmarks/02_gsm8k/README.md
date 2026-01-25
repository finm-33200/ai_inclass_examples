# 02 - GSM8K Benchmark

Compare multiple models on grade-school math problems using OpenAI's GSM8K benchmark.

## What You'll Learn

- Running benchmarks across multiple models
- Parsing lm-eval JSON output
- Visualizing model comparisons with matplotlib

## Why GSM8K?

GSM8K (Grade School Math 8K) is a benchmark released by OpenAI containing ~8,000 grade-school math word problems. It's valuable because:

1. **Verifiable answers**: Each problem has a single correct numeric answer
2. **Reasoning required**: Models must show step-by-step work
3. **Relevant pattern**: Same structure as quant finance interview questions (problem → reasoning → number)

Example problem:
> Janet's ducks lay 16 eggs per day. She eats three for breakfast and bakes muffins with four. She sells the rest at $2 each. How much does she make per day?

## Models Compared

All ~$0.05-0.10 per 1M tokens via OpenRouter:

| Model | Size | Notes |
|-------|------|-------|
| `meta-llama/llama-3.1-8b-instruct` | 8B | Meta's instruction-tuned Llama |
| `google/gemma-2-9b-it` | 9B | Google's Gemma 2 |
| `qwen/qwen-2.5-7b-instruct` | 7B | Alibaba's Qwen 2.5 |

## How to Run

### Step 1: Run benchmarks

```bash
# Quick demo (~20 seconds, 5 samples per model)
./01_run_gsm8k.sh

# Fuller run (100 samples per model)
./01_run_gsm8k.sh 100

# Full benchmark (all 1319 samples)
./01_run_gsm8k.sh all
```

### Step 2: Analyze results

```bash
python 02_compare_models.py
```

This generates a bar chart comparing model accuracy.

## Cost Estimate

| Samples | Est. Cost (3 models) |
|---------|---------------------|
| 5 (default) | ~$0.002 |
| 100 | ~$0.03-0.05 |
| Full (1319) | ~$0.30-0.50 |

## Output

Results are saved to `results/` as JSON files:
- `results/llama-3.1-8b/`
- `results/gemma-2-9b/`
- `results/qwen-2.5-7b/`

Each contains detailed per-example results and aggregate metrics.

## Understanding Results

lm-eval prints a results table like this:

```
|Tasks|Version|     Filter     |n-shot|  Metric   |   |Value|   |Stderr|
|-----|------:|----------------|-----:|-----------|---|----:|---|-----:|
|gsm8k|      3|flexible-extract|     5|exact_match|↑  | 0.77|±  |0.0423|
|     |       |strict-match    |     5|exact_match|↑  | 0.36|±  |0.0482|
```

### Column Definitions

| Column | Meaning |
|--------|---------|
| **Tasks** | Benchmark name (gsm8k) |
| **Version** | Task version number (for reproducibility) |
| **Filter** | Answer extraction method (see below) |
| **n-shot** | Number of few-shot examples shown to the model before each question |
| **Metric** | What's being measured (`exact_match` = correct answer) |
| **↑** | Arrow indicates higher values are better |
| **Value** | The score (0.77 = 77% correct) |
| **Stderr** | Standard error - confidence interval for the score |

### GSM8K Filters

GSM8K uses two different methods to extract and score answers:

| Filter | Description | When to Use |
|--------|-------------|-------------|
| **flexible-extract** | Parses the response to find any final numerical answer | Primary metric - more realistic of real-world usage |
| **strict-match** | Requires exact format: `#### 42` at the end | Tests if model follows GSM8K's specific format |

**Why the big difference?** In the example above, Qwen scores 77% with flexible extraction but only 36% with strict matching. This means the model often gets the right answer but doesn't format it as `#### answer`. For practical purposes, **flexible-extract is the more meaningful metric**.

### Interpreting Standard Error

The Stderr column shows uncertainty in your measurement:
- `0.77 ± 0.0423` means the true accuracy is likely between ~73% and ~81%
- Smaller samples → larger standard error
- Running 100 samples gives Stderr ~0.04-0.05
- Running full benchmark (1319 samples) gives Stderr ~0.01

## Try It

1. Run the quick demo (default) to verify setup
2. Compare results across models - which does best?
3. Run with 100 samples for more reliable results
4. Try adding a 4th model (e.g., `mistralai/mistral-7b-instruct`)
5. Examine the JSON output to see per-example results

## Connection to Finance

This benchmark pattern (problem → reasoning → verifiable answer) is the same pattern used in:
- Quant finance interview questions
- Risk calculations
- Portfolio math problems

The homework will have students create similar question generators for finance domains.

## Resources

- [GSM8K GitHub](https://github.com/openai/grade-school-math) (OpenAI)
- [GSM8K Paper](https://arxiv.org/abs/2110.14168)
- [lm-eval GSM8K task](https://github.com/EleutherAI/lm-evaluation-harness/tree/main/lm_eval/tasks/gsm8k)
