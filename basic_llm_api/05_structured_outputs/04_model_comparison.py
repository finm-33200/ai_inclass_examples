"""
05_structured_outputs/04_model_comparison.py

Compare structured output reliability across different models.
Not all models handle JSON schema enforcement equally well.

This script tests the same structured output request against:
- openai/gpt-4o-mini (reliable)
- google/gemini-2.0-flash-exp:free (may have issues)
"""

import json
from pathlib import Path
from typing import Literal

from decouple import Config, RepositoryEnv
from openai import OpenAI
from pydantic import BaseModel, Field, ValidationError


class CompanyAnalysis(BaseModel):
    """Structured analysis of a company from financial text."""

    ticker: str = Field(description="Stock ticker symbol")
    company_name: str = Field(description="Full company name")
    revenue_billions: float | None = Field(
        description="Reported revenue in billions USD, or null if not mentioned"
    )
    sentiment: Literal["bullish", "bearish", "neutral"] = Field(
        description="Overall sentiment based on the text"
    )
    key_points: list[str] = Field(description="2-4 key takeaways from the text")


def pydantic_to_strict_schema(model: type[BaseModel]) -> dict:
    """
    Convert a Pydantic model to a strict JSON schema compatible with OpenAI/OpenRouter.
    """
    schema = model.model_json_schema()
    schema["additionalProperties"] = False
    if "properties" in schema:
        schema["required"] = list(schema["properties"].keys())
    if "$defs" in schema:
        del schema["$defs"]
    return schema


def test_model(client: OpenAI, model: str, messages: list, schema: dict) -> dict:
    """Test a model's structured output capability and return results."""
    result = {
        "model": model,
        "success": False,
        "error": None,
        "data": None,
        "raw_response": None,
    }

    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "CompanyAnalysis",
                    "strict": True,
                    "schema": schema,
                },
            },
        )

        raw_content = response.choices[0].message.content
        result["raw_response"] = raw_content

        # Try to parse as JSON
        data = json.loads(raw_content)

        # Validate with Pydantic
        analysis = CompanyAnalysis.model_validate(data)
        result["data"] = analysis.model_dump()
        result["success"] = True

    except json.JSONDecodeError as e:
        result["error"] = f"JSON parse error: {e}"
    except ValidationError as e:
        result["error"] = f"Pydantic validation error: {e}"
    except Exception as e:
        result["error"] = f"API error: {e}"

    return result


def main():
    # Load API key from .env at repository root
    repo_root = Path(__file__).resolve().parents[2]
    config = Config(RepositoryEnv(repo_root / ".env"))

    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=config("OPENROUTER_API_KEY"),
    )

    # Models to compare
    models = [
        "openai/gpt-4o-mini",
        "google/gemini-2.0-flash-exp:free",
    ]

    # Sample financial text
    financial_text = """
    Apple Inc. (AAPL) reported Q4 2024 earnings that beat analyst expectations.
    Revenue came in at $94.9 billion, up 6% year-over-year. CEO Tim Cook
    noted concerns about China market conditions but remained optimistic.
    """

    messages = [
        {
            "role": "system",
            "content": "You are a financial analyst. Extract structured data from the provided text.",
        },
        {
            "role": "user",
            "content": f"Analyze this text:\n\n{financial_text}",
        },
    ]

    schema = pydantic_to_strict_schema(CompanyAnalysis)

    print("=" * 60)
    print("STRUCTURED OUTPUT MODEL COMPARISON")
    print("=" * 60)
    print("\nInput text:")
    print(financial_text.strip())
    print("\n")

    # Test each model
    for model in models:
        print("-" * 60)
        print(f"Testing: {model}")
        print("-" * 60)

        result = test_model(client, model, messages, schema)

        if result["success"]:
            print("Status: SUCCESS")
            print(f"Data: {json.dumps(result['data'], indent=2)}")
        else:
            print("Status: FAILED")
            print(f"Error: {result['error']}")
            if result["raw_response"]:
                print(f"Raw response: {result['raw_response'][:500]}...")

        print()

    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print("""
Key takeaways:
- GPT-4o-mini reliably follows JSON schema constraints
- Some models (especially free tiers) may have compatibility issues
- Always validate responses with Pydantic for type safety
- Consider fallback strategies for production use
""")


if __name__ == "__main__":
    main()
