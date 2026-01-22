"""
05_structured_outputs/json_mode.py

Basic JSON mode - request JSON output from the LLM.
This is simpler but less type-safe than Pydantic schemas.
"""

import json
from pathlib import Path

from decouple import Config, RepositoryEnv
from openai import OpenAI


def main():
    # Load API key from .env at repository root
    repo_root = Path(__file__).resolve().parents[2]
    config = Config(RepositoryEnv(repo_root / ".env"))

    client = OpenAI(api_key=config("OPENAI_API_KEY"))

    # Sample financial text to analyze
    financial_text = """
    Apple Inc. (AAPL) reported Q4 2024 earnings that beat analyst expectations.
    Revenue came in at $94.9 billion, up 6% year-over-year. The company's
    Services segment showed strong growth at 12% YoY. iPhone sales were
    slightly below expectations but Mac sales exceeded forecasts. CEO Tim Cook
    noted concerns about China market conditions but remained optimistic about
    AI product integration.
    """

    messages = [
        {
            "role": "system",
            "content": """You are a financial analyst. Extract key information
            from the provided text and return it as JSON with these fields:
            - ticker: string
            - company_name: string
            - revenue_billions: number or null
            - sentiment: "bullish", "bearish", or "neutral"
            - key_points: array of strings (2-3 main takeaways)

            Return ONLY valid JSON, no other text.""",
        },
        {
            "role": "user",
            "content": f"Analyze this text:\n\n{financial_text}",
        },
    ]

    print("Input text:")
    print(financial_text.strip())
    print("\n" + "=" * 60 + "\n")

    # Request JSON output
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        response_format={"type": "json_object"},  # Enable JSON mode
    )

    # Parse the JSON response
    json_str = response.choices[0].message.content
    data = json.loads(json_str)

    print("Extracted data:")
    print(json.dumps(data, indent=2))

    # Access fields programmatically
    print(f"\nTicker: {data['ticker']}")
    print(f"Sentiment: {data['sentiment']}")


if __name__ == "__main__":
    main()
