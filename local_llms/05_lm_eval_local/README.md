# 05_lm_eval_local - Benchmark Local Models

Use the lm-evaluation-harness to benchmark local HuggingFace models. This bridges to the `benchmarks/` directory examples but runs entirely on your machine.

## What You'll Learn

- Running lm-eval with local HuggingFace models
- Comparing local model performance to API benchmarks
- Understanding benchmark metrics and limitations

## Key Concepts

- **lm-evaluation-harness**: Standard framework for evaluating language models
- **GSM8K**: Grade school math benchmark (tests reasoning)
- **HellaSwag**: Common sense reasoning benchmark
- **Batch size**: Number of examples processed at once (lower = less memory)

## Prerequisites

- Python 3.10+
- ~4GB RAM
- TinyLlama model (downloaded automatically)

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Running the Examples

```bash
# Run GSM8K benchmark (small subset)
chmod +x 01_tinyllama_gsm8k.sh
./01_tinyllama_gsm8k.sh

# Compare results to API models
python 02_compare_local_vs_api.py
```

## Understanding the Output

The benchmark produces metrics like:
- **acc**: Accuracy (% of correct answers)
- **acc_norm**: Normalized accuracy
- **exact_match**: For GSM8K, whether the numerical answer matches exactly

## Benchmark Tasks

| Task | Type | Measures |
|------|------|----------|
| gsm8k | Math | Multi-step arithmetic reasoning |
| hellaswag | Reasoning | Common sense completion |
| truthfulqa | Factuality | Resistance to common misconceptions |
| mmlu | Knowledge | Multi-domain academic knowledge |

## Notes on Local Benchmarking

- **Speed**: Local CPU inference is slower than GPU/API
- **Memory**: Some benchmarks require significant context
- **Reproducibility**: Same model + same prompts = same results (unlike APIs with variation)

## Try It

1. Run with `--limit 5` for quick tests, increase for more accurate results
2. Try different models: change the `--model_args pretrained=` path
3. Compare TinyLlama to Phi-2 on math tasks
