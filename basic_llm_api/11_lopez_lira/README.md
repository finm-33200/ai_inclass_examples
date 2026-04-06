# 11_lopez_lira: Sentiment Classification for Return Prediction

This example replicates the core methodology from Lopez-Lira (2023), "Can ChatGPT Forecast Stock Price Movements?" The paper demonstrated that GPT-4's sentiment classifications of financial news headlines correlate with next-day stock returns.

## Key Insight

Lopez-Lira's headline finding: **model scale matters.** GPT-4 sentiment scores predict returns, but smaller models (BERT, GPT-3.5) showed no predictive power. This explains the shift from fine-tuned small models to prompting frontier models.

## Files

| File | Description |
|------|-------------|
| `01_classify_headlines.py` | Basic sentiment classification with GPT-4o-mini |
| `02_structured_sentiment.py` | Type-safe classification using Pydantic schemas |
| `03_batch_classification.py` | Process multiple headlines with error handling |
| `data/sample_headlines.csv` | 25 sample financial headlines for testing |

## Setup

```bash
# From repository root
cd basic_llm_api/11_lopez_lira

# Ensure .env file exists at repo root with OPENAI_API_KEY
# Run any example
python 01_classify_headlines.py
```

## Classification Schema

Following Lopez-Lira (2023), headlines are classified as:

- **good news** - Positive for the company's stock price
- **bad news** - Negative for the company's stock price
- **uncertain** - Ambiguous or mixed implications

## Example Output

```
Apple Reports Record Q4 Earnings... → good news (95%)
Fed Signals More Rate Hikes...     → bad news (88%)
Tesla Announces New Factory...     → uncertain (62%)
```

## Try It

1. Run the basic classifier and observe outputs
2. Modify the system prompt and compare results
3. Add your own headlines to `data/sample_headlines.csv`
4. Compare GPT-4o-mini vs other models via OpenRouter

## Paper Reference

> Lopez-Lira, A. & Tang, Y. (2023). "Can ChatGPT Forecast Stock Price Movements? Return Predictability and Large Language Models." [arXiv:2304.07619](https://arxiv.org/abs/2304.07619)
