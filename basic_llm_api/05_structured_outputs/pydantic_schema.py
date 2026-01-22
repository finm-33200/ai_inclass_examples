"""
05_structured_outputs/pydantic_schema.py

Type-safe structured outputs using Pydantic schemas.
OpenAI guarantees the response matches your schema.
"""

from pathlib import Path
from typing import Literal

from decouple import Config, RepositoryEnv
from openai import OpenAI
from pydantic import BaseModel, Field


# Define the schema for company analysis
class CompanyAnalysis(BaseModel):
    """Structured analysis of a company from financial text."""

    ticker: str = Field(description="Stock ticker symbol")
    company_name: str = Field(description="Full company name")
    revenue_billions: float | None = Field(
        description="Reported revenue in billions USD, or null if not mentioned"
    )
    revenue_growth_pct: float | None = Field(
        description="Year-over-year revenue growth percentage, or null if not mentioned"
    )
    sentiment: Literal["bullish", "bearish", "neutral"] = Field(
        description="Overall sentiment based on the text"
    )
    confidence: float = Field(
        description="Confidence in the analysis from 0.0 to 1.0", ge=0.0, le=1.0
    )
    key_points: list[str] = Field(
        description="2-4 key takeaways from the text", min_length=2, max_length=4
    )
    risk_factors: list[str] = Field(
        description="Any mentioned risks or concerns", default_factory=list
    )


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
    AI product integration. The company announced a $110 billion share buyback
    program.
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

    print("Input text:")
    print(financial_text.strip())
    print("\n" + "=" * 60 + "\n")

    # Use structured output with Pydantic schema
    response = client.beta.chat.completions.parse(
        model="gpt-4o-mini",
        messages=messages,
        response_format=CompanyAnalysis,  # Pass the Pydantic class
    )

    # Get the parsed object - already typed!
    analysis: CompanyAnalysis = response.choices[0].message.parsed

    # Access fields with type safety
    print("Extracted Analysis:")
    print(f"  Company: {analysis.company_name} ({analysis.ticker})")
    print(f"  Revenue: ${analysis.revenue_billions}B")
    print(f"  Growth: {analysis.revenue_growth_pct}%")
    print(f"  Sentiment: {analysis.sentiment}")
    print(f"  Confidence: {analysis.confidence:.0%}")

    print("\nKey Points:")
    for i, point in enumerate(analysis.key_points, 1):
        print(f"  {i}. {point}")

    if analysis.risk_factors:
        print("\nRisk Factors:")
        for risk in analysis.risk_factors:
            print(f"  - {risk}")

    # You can also convert back to dict/JSON
    print("\n--- As JSON ---")
    print(analysis.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
