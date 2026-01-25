# Benchmarks

Standardized LLM benchmarking using EleutherAI's `lm-evaluation-harness` - the industry-standard framework.

## Why lm-evaluation-harness?

> **Use EleutherAI's lm-evaluation-harness as the backbone**, and optionally layer Promptfoo or LangSmith *around* it—not instead of it.

- **De facto standard** for benchmarking - produces numbers that reviewers, hiring managers, and readers immediately trust
- **Native HuggingFace support** - critical for local models (like ChronoGPT)
- **API model support** - works with OpenRouter via OpenAI-compatible endpoint
- **Extensible** - can add custom benchmarks

### When to Use What

| Tool | Use For |
|------|---------|
| **lm-evaluation-harness** | Standard benchmarks (HellaSwag, MMLU, GSM8K, etc.) |
| **Promptfoo** | Custom prompts, A/B tests, CI/CD, regression tests |
| **LangSmith** | Agent tracing, production monitoring, experiment tracking |

## Two Modes of Operation

| Use Case | Model Type | Example |
|----------|-----------|---------|
| **In-class (these examples)** | API models via OpenRouter | Fast, cheap, no GPU needed |
| **Research** | Local HuggingFace models | ChronoGPT, custom fine-tuned models |

## Progression

| Example | What You Learn |
|---------|----------------|
| [01_lm_eval_intro](01_lm_eval_intro/) | First benchmark in one command |
| [02_gsm8k](02_gsm8k/) | Math reasoning benchmark (OpenAI's grade-school math) |
| [03_mmlu_subset](03_mmlu_subset/) | Knowledge testing across finance-relevant subjects |
| [04_multi_benchmark](04_multi_benchmark/) | Run multiple benchmarks together |

## Cost Estimates (100 samples × 3 cheap models)

| Benchmark | Est. Cost |
|-----------|-----------|
| GSM8K | ~$0.02-0.05 |
| MMLU (6 subjects) | ~$0.02-0.04 |
| HellaSwag | ~$0.01-0.02 |
| TruthfulQA | ~$0.01-0.03 |
| **Total** | **~$0.15-0.40** |

## API Model Limitations

When using API models (vs local HuggingFace models):

- **No logprobs**: Most APIs don't return token probabilities
- **MCQ scoring**: MMLU uses generation-based scoring (model outputs "A", "B", "C", or "D") instead of logprob comparison
- **Results may differ**: Scores may differ slightly from published benchmarks that use logprob scoring
- **For research-grade benchmarks**: Use local HF models with native integration

## Prerequisites

- Python 3.10+
- OpenRouter API key (set `OPENROUTER_API_KEY` in repo root `.env`)

## Installation

```bash
pip install lm-eval
```

## Models Used

These examples use cheap models via OpenRouter (~$0.05-0.10 per 1M tokens):

- `meta-llama/llama-3.1-8b-instruct`
- `google/gemma-2-9b-it`
- `qwen/qwen-2.5-7b-instruct`

## Resources

- [lm-evaluation-harness GitHub](https://github.com/EleutherAI/lm-evaluation-harness)
- [Available Tasks](https://github.com/EleutherAI/lm-evaluation-harness/tree/main/lm_eval/tasks)
- [OpenRouter Models](https://openrouter.ai/models)
- [GSM8K Paper](https://arxiv.org/abs/2110.14168) (OpenAI)
