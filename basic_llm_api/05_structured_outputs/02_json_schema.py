"""
05_structured_outputs/02_json_schema.py

Explicit JSON schema - the model MUST return data matching your schema.
More reliable than basic JSON mode because the schema is enforced.

Uses OpenRouter's json_schema response format.
"""

import json
from pathlib import Path

from decouple import Config, RepositoryEnv
from openai import OpenAI


def main():
    # Load API key from .env at repository root
    repo_root = Path(__file__).resolve().parents[2]
    config = Config(RepositoryEnv(repo_root / ".env"))

    # Create client with OpenRouter base URL
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=config("OPENROUTER_API_KEY"),
    )

    # Sample financial text to analyze
    financial_text = """
    Apple Inc. (AAPL) reported Q4 2024 earnings that beat analyst expectations.
    Revenue came in at $94.9 billion, up 6% year-over-year. The company's
    Services segment showed strong growth at 12% YoY. iPhone sales were
    slightly below expectations but Mac sales exceeded forecasts. CEO Tim Cook
    noted concerns about China market conditions but remained optimistic about
    AI product integration.
    """

    # Define the JSON schema explicitly
    # This schema is enforced by the model - output MUST match
    company_analysis_schema = {
        "type": "object",
        "properties": {
            "ticker": {
                "type": "string",
                "description": "Stock ticker symbol",
            },
            "company_name": {
                "type": "string",
                "description": "Full company name",
            },
            "revenue_billions": {
                "type": ["number", "null"],
                "description": "Reported revenue in billions USD, or null if not mentioned",
            },
            "sentiment": {
                "type": "string",
                "enum": ["bullish", "bearish", "neutral"],
                "description": "Overall sentiment based on the text",
            },
            "key_points": {
                "type": "array",
                "items": {"type": "string"},
                "description": "2-4 key takeaways from the text",
            },
        },
        "required": ["ticker", "company_name", "revenue_billions", "sentiment", "key_points"],
        "additionalProperties": False,
    }

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

    print("Input text:")
    print(financial_text.strip())
    print("\n" + "=" * 60 + "\n")

    # Request structured output with explicit JSON schema
    response = client.chat.completions.create(
        model="openai/gpt-4o-mini",
        messages=messages,
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "company_analysis",
                "strict": True,
                "schema": company_analysis_schema,
            },
        },
    )

    # Parse the JSON response
    json_str = response.choices[0].message.content
    data = json.loads(json_str)

    print("Extracted data (schema-enforced):")
    print(json.dumps(data, indent=2))

    # Access fields programmatically
    print(f"\nCompany: {data['company_name']} ({data['ticker']})")
    print(f"Revenue: ${data['revenue_billions']}B")
    print(f"Sentiment: {data['sentiment']}")

    print("\nKey Points:")
    for i, point in enumerate(data["key_points"], 1):
        print(f"  {i}. {point}")

    # Show model used
    print(f"\n--- Details ---")
    print(f"Model: {response.model}")


if __name__ == "__main__":
    main()
