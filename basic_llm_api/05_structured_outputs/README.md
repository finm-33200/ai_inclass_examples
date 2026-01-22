# 05 - Structured Outputs

Get reliable, typed JSON data from LLMs instead of free-form text.

## Why Structured Outputs?

LLMs naturally produce unstructured text, but many applications need structured data:
- Extracting entities from documents
- Building pipelines that consume LLM output
- Ensuring consistent response formats

Structured outputs guarantee the LLM returns valid JSON matching your schema.

## What You'll Learn

- Using JSON mode for basic structured responses
- Using Pydantic for type-safe schemas
- OpenAI's structured output feature
- Extracting financial data from text

## Setup

```bash
cd 05_structured_outputs
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Examples

### Basic JSON Mode
```bash
python json_mode.py
```

### Pydantic Schema (Recommended)
```bash
python pydantic_schema.py
```

## Key Concepts

### JSON Mode

Request JSON output with `response_format`:

```python
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=messages,
    response_format={"type": "json_object"},  # Request JSON
)
```

**Important**: You must also instruct the model in the prompt to output JSON.

### Pydantic Schemas

Define a typed schema that OpenAI will enforce:

```python
from pydantic import BaseModel

class CompanyAnalysis(BaseModel):
    ticker: str
    sentiment: Literal["bullish", "bearish", "neutral"]
    confidence: float
    key_points: list[str]
```

### OpenAI Structured Outputs

OpenAI's `response_format` with Pydantic guarantees valid output:

```python
response = client.beta.chat.completions.parse(
    model="gpt-4o-mini",
    messages=messages,
    response_format=CompanyAnalysis,
)
analysis = response.choices[0].message.parsed
```

## Finance Application

The `pydantic_schema.py` example extracts:
- Ticker symbol
- Sentiment (bullish/bearish/neutral)
- Key financial metrics
- Risk factors

This pattern is useful for processing earnings calls, news articles, and analyst reports.

## Try It

1. Modify the Pydantic schema to extract additional fields
2. Try with different financial news snippets
3. What happens if the text doesn't contain a clear ticker?
4. Experiment with nested schemas (e.g., a list of companies)
