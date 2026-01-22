"""
05_structured_outputs/03_pydantic_schema.py

Type-safe structured outputs using Pydantic schemas.
Combines the benefits of:
- Pydantic's type definitions and validation
- OpenRouter's JSON schema enforcement

The Pydantic model defines the schema, which is passed to the API.
The response is then validated back into a typed Pydantic object.
"""

from pathlib import Path
from typing import Literal

from decouple import Config, RepositoryEnv
from openai import OpenAI
from pydantic import BaseModel, Field


# Define the schema for company analysis using Pydantic
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
    key_points: list[str] = Field(description="2-4 key takeaways from the text")
    risk_factors: list[str] = Field(
        description="Any mentioned risks or concerns", default_factory=list
    )


def pydantic_to_strict_schema(model: type[BaseModel]) -> dict:
    """
    Convert a Pydantic model to a strict JSON schema compatible with OpenAI/OpenRouter.

    OpenAI's strict mode requires:
    - additionalProperties: false on all objects
    - All properties listed in 'required' (no optional fields in the traditional sense)
    - Nullable fields use anyOf with null type
    """
    schema = model.model_json_schema()

    # Add additionalProperties: false to the root
    schema["additionalProperties"] = False

    # Make all properties required (OpenAI strict mode requirement)
    # Pydantic's model_json_schema() already handles nullable types with anyOf
    if "properties" in schema:
        schema["required"] = list(schema["properties"].keys())

    # Remove $defs if present (flatten for compatibility)
    if "$defs" in schema:
        del schema["$defs"]

    return schema


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

    # Convert Pydantic model to a strict-compatible JSON schema
    # OpenAI/OpenRouter strict mode has specific requirements
    schema = pydantic_to_strict_schema(CompanyAnalysis)

    response = client.chat.completions.create(
        model="openai/gpt-4o-mini",
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

    # Validate and parse the response into a typed Pydantic object
    analysis = CompanyAnalysis.model_validate_json(response.choices[0].message.content)

    # Access fields with full type safety
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

    # Show model used
    print(f"\n--- Details ---")
    print(f"Model: {response.model}")


if __name__ == "__main__":
    main()
