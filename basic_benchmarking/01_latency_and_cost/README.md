# 01 - Latency and Cost Benchmarking

Learn how to manually benchmark LLM API calls by measuring latency, token usage, and cost.

## What You'll Learn

- Timing API calls with `time.time()`
- Extracting token usage from OpenAI API responses
- Calculating cost from per-million-token pricing
- Storing results as JSON for reproducibility

## Key Concepts

### Latency Measurement

```python
start = time.time()
response = client.chat.completions.create(...)
latency = time.time() - start
```

### Token Extraction

The OpenAI API returns token counts in the response:

```python
response.usage.prompt_tokens    # Input tokens
response.usage.completion_tokens  # Output tokens
response.usage.total_tokens     # Sum
```

### Cost Calculation

OpenAI prices are per million tokens (different for input vs output):

```python
cost = (input_tokens * input_price + output_tokens * output_price) / 1_000_000
```

## How to Run

```bash
# Install dependencies
pip install -r requirements.txt

# Run benchmark (makes API calls)
python benchmark.py

# Analyze results
python analyze.py
```

## Files

| File | Purpose |
|------|---------|
| `benchmark.py` | Runs prompts against models, measures metrics |
| `analyze.py` | Loads results, computes summaries |
| `results/` | JSON files from each benchmark run |

## Try It

1. Add a third model to the `MODELS` list (e.g., `gpt-3.5-turbo`)
2. Add a new finance prompt to test
3. Run multiple times and compare variance in latency
4. Modify `analyze.py` to find the fastest model per prompt
