# 05 - Structured Outputs

Get reliable, typed JSON data from LLMs instead of free-form text.

All examples use **OpenRouter** for model access.

## Why Structured Outputs?

LLMs naturally produce unstructured text, but many applications need structured data:
- Extracting entities from documents
- Building pipelines that consume LLM output
- Ensuring consistent response formats

Structured outputs guarantee the LLM returns valid JSON matching your schema.

## What You'll Learn

1. **JSON Mode** - Basic approach, request JSON output
2. **JSON Schema** - Explicit schema enforcement
3. **Pydantic Schema** - Type-safe Python objects
4. **Model Comparison** - See which models handle schemas reliably

## Setup

```bash
cd 05_structured_outputs
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Ensure `OPENROUTER_API_KEY` is set in your `.env` file at the repository root.

## Examples

### 01 - Basic JSON Mode
```bash
python 01_json_mode.py
```
Simplest approach. Request JSON output with `response_format={"type": "json_object"}`.
You must instruct the model in the prompt to output JSON.

### 02 - Explicit JSON Schema
```bash
python 02_json_schema.py
```
Define a JSON schema that the model MUST follow. More reliable than basic JSON mode.

### 03 - Pydantic Schema (Recommended)
```bash
python 03_pydantic_schema.py
```
Best of both worlds: Pydantic type definitions + API schema enforcement.

### 04 - Model Comparison
```bash
python 04_model_comparison.py
```
Compare GPT-4o-mini vs Gemini. See which models handle structured outputs reliably.

## Key Concepts

### JSON Mode (Simplest)

```python
response = client.chat.completions.create(
    model="openai/gpt-4o-mini",
    messages=messages,
    response_format={"type": "json_object"},
)
data = json.loads(response.choices[0].message.content)
```

### Explicit JSON Schema (More Reliable)

```python
response = client.chat.completions.create(
    model="openai/gpt-4o-mini",
    messages=messages,
    response_format={
        "type": "json_schema",
        "json_schema": {
            "name": "company_analysis",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    "ticker": {"type": "string"},
                    "sentiment": {"type": "string", "enum": ["bullish", "bearish", "neutral"]},
                },
                "required": ["ticker", "sentiment"],
            },
        },
    },
)
```

### Pydantic Schema (Recommended)

```python
from pydantic import BaseModel, Field

class CompanyAnalysis(BaseModel):
    ticker: str
    sentiment: Literal["bullish", "bearish", "neutral"]
    confidence: float = Field(ge=0.0, le=1.0)

def pydantic_to_strict_schema(model):
    """Convert Pydantic model to OpenAI-compatible strict schema."""
    schema = model.model_json_schema()
    schema["additionalProperties"] = False
    schema["required"] = list(schema["properties"].keys())
    if "$defs" in schema:
        del schema["$defs"]
    return schema

response = client.chat.completions.create(
    model="openai/gpt-4o-mini",
    messages=messages,
    response_format={
        "type": "json_schema",
        "json_schema": {
            "name": "CompanyAnalysis",
            "strict": True,
            "schema": pydantic_to_strict_schema(CompanyAnalysis),
        },
    },
)
analysis = CompanyAnalysis.model_validate_json(response.choices[0].message.content)
```

**Note:** OpenAI's strict mode requires `additionalProperties: false` and all fields in `required`. The `pydantic_to_strict_schema` helper transforms Pydantic's schema to meet these requirements.

## Model Compatibility

Not all models handle structured outputs equally well:

| Model | JSON Mode | JSON Schema | Notes |
|-------|-----------|-------------|-------|
| `openai/gpt-4o-mini` | ✅ | ✅ | Reliable |
| `google/gemini-2.0-flash-exp:free` | ✅ | ⚠️ | May have schema issues |

Run `04_model_comparison.py` to see the differences firsthand.

## Finance Application

The examples extract:
- Ticker symbol
- Sentiment (bullish/bearish/neutral)
- Key financial metrics
- Risk factors

This pattern is useful for processing earnings calls, news articles, and analyst reports.

## Try It

1. Modify the Pydantic schema to extract additional fields
2. Try with different financial news snippets
3. What happens if the text doesn't contain a clear ticker?
4. Run the model comparison and observe the differences
