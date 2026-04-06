"""
11_lopez_lira/02_structured_sentiment.py

Type-safe sentiment classification using Pydantic schemas.
Returns structured JSON with classification, confidence, and reasoning.

This version uses OpenAI's structured outputs feature for guaranteed
schema compliance.
"""

from pathlib import Path
from typing import Literal

from decouple import Config, RepositoryEnv
from openai import OpenAI
from pydantic import BaseModel, Field


class LopezLiraSentiment(BaseModel):
    """Structured sentiment analysis following Lopez-Lira (2023)."""

    classification: Literal["good news", "bad news", "uncertain"] = Field(
        description="Sentiment classification for stock price impact"
    )
    confidence: float = Field(
        description="Confidence in the classification from 0.0 to 1.0",
        ge=0.0,
        le=1.0,
    )
    reasoning: str = Field(
        description="Brief explanation of why this classification was chosen"
    )
    entities: list[str] = Field(
        description="Company names or tickers mentioned in the headline",
        default_factory=list,
    )


def pydantic_to_strict_schema(model: type[BaseModel]) -> dict:
    """
    Convert a Pydantic model to a strict JSON schema compatible with OpenAI.

    OpenAI's strict mode requires:
    - additionalProperties: false on all objects
    - All properties listed in 'required'
    """
    schema = model.model_json_schema()
    schema["additionalProperties"] = False

    if "properties" in schema:
        schema["required"] = list(schema["properties"].keys())

    if "$defs" in schema:
        del schema["$defs"]

    return schema


def classify_headline(client: OpenAI, headline: str) -> LopezLiraSentiment:
    """Classify a single headline and return structured output."""

    system_prompt = """You are a financial expert. For each headline, determine whether
it represents 'good news', 'bad news', or 'uncertain' for the mentioned company's
stock price.

Consider:
- Market expectations (beating vs missing estimates)
- Typical investor reactions to similar news
- The broader market context implied

Provide your classification with a confidence score and brief reasoning."""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Headline: {headline}"},
        ],
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "LopezLiraSentiment",
                "strict": True,
                "schema": pydantic_to_strict_schema(LopezLiraSentiment),
            },
        },
        temperature=0,
    )

    return LopezLiraSentiment.model_validate_json(
        response.choices[0].message.content
    )


def main():
    # Load API key from .env at repository root
    repo_root = Path(__file__).resolve().parents[2]
    config = Config(RepositoryEnv(repo_root / ".env"))

    client = OpenAI(api_key=config("OPENAI_API_KEY"))

    # Sample headlines to classify
    headlines = [
        "Apple Reports Record Q4 Earnings, Beats Analyst Expectations",
        "Fed Signals More Rate Hikes Ahead, Markets Tumble",
        "Tesla Stock Plunges After Disappointing Delivery Numbers",
        "Microsoft Cloud Revenue Grows 27%, Exceeding Forecasts",
        "Bank of America Warns of Recession Risk in 2024",
    ]

    print("Lopez-Lira Structured Sentiment Classification")
    print("=" * 70)
    print()

    for headline in headlines:
        result = classify_headline(client, headline)

        print(f"Headline: {headline}")
        print(f"  Classification: {result.classification}")
        print(f"  Confidence: {result.confidence:.0%}")
        print(f"  Entities: {', '.join(result.entities) if result.entities else 'None'}")
        print(f"  Reasoning: {result.reasoning}")
        print()

    # Show example JSON output
    print("-" * 70)
    print("Example JSON output:")
    print(result.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
