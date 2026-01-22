# 10 - Model Comparison

Systematically compare LLM models on quality, cost, and latency.

## Why Compare Models?

Different models have different:
- **Quality**: Accuracy, reasoning ability, instruction following
- **Cost**: Price per token varies 10-100x across models
- **Latency**: Response times range from <1s to 30+ seconds
- **Capabilities**: Not all models support all features

Choosing the right model for your task can save significant cost.

## What You'll Learn

- Running prompts across multiple models
- Measuring latency and token usage
- Comparing output quality
- Making cost/quality tradeoffs

## Setup

```bash
cd 10_model_comparison
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Examples

### Run Benchmark
```bash
python benchmark.py
```

### Analyze Results
```bash
python evaluate.py
```

## Key Concepts

### Models to Compare

Via OpenRouter, you can compare:
- OpenAI: `gpt-4o`, `gpt-4o-mini`
- Anthropic: `claude-3-5-sonnet`
- Google: `gemini-2.0-flash-exp:free`
- Open-source: `llama-3.1-70b-instruct`, `mixtral-8x7b`

### Benchmark Design

A good benchmark includes:
1. **Multiple prompts**: Test different task types
2. **Consistent evaluation**: Same criteria for all models
3. **Multiple runs**: Account for variance
4. **Relevant tasks**: Match your actual use case

### Cost Calculation

```python
cost = (input_tokens * input_price + output_tokens * output_price) / 1_000_000
```

Prices are per million tokens. A simple prompt might use 100 tokens, an earnings analysis might use 5,000+.

## Finance Tasks

The benchmark includes finance-specific prompts:
- Financial concept explanation
- Stock comparison
- Risk analysis
- Data extraction

## Results

Results are saved to `results/` directory as JSON for further analysis.

## Try It

1. Add your own benchmark prompts
2. Add more models to compare
3. Create a cost-optimized model routing strategy
4. Compare open-source vs proprietary models on your specific task
