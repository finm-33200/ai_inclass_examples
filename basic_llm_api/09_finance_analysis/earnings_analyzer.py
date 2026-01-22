"""
09_finance_analysis/earnings_analyzer.py

Analyze earnings call transcripts using structured outputs.
Extracts key metrics, sentiment, and insights from earnings calls.
"""

from pathlib import Path
from typing import Literal

from decouple import Config, RepositoryEnv
from openai import OpenAI
from pydantic import BaseModel, Field


class FinancialMetrics(BaseModel):
    """Extracted financial metrics from earnings call."""

    revenue_billions: float | None = Field(
        description="Total revenue in billions USD"
    )
    revenue_growth_pct: float | None = Field(
        description="Year-over-year revenue growth percentage"
    )
    gross_margin_pct: float | None = Field(description="Gross margin percentage")
    eps: float | None = Field(description="Earnings per share if mentioned")


class Guidance(BaseModel):
    """Forward-looking guidance from earnings call."""

    revenue_growth_guidance: str | None = Field(
        description="Revenue guidance (e.g., 'low-to-mid single digits')"
    )
    gross_margin_guidance: str | None = Field(
        description="Gross margin guidance range"
    )
    other_guidance: list[str] = Field(
        description="Other forward-looking statements", default_factory=list
    )


class EarningsAnalysis(BaseModel):
    """Complete analysis of an earnings call transcript."""

    ticker: str = Field(description="Stock ticker symbol")
    company_name: str = Field(description="Full company name")
    quarter: str = Field(description="Quarter reported (e.g., 'Q4 FY2024')")

    # Metrics
    metrics: FinancialMetrics = Field(description="Key financial metrics")

    # Guidance
    guidance: Guidance = Field(description="Forward-looking guidance")

    # Qualitative analysis
    overall_sentiment: Literal["positive", "negative", "neutral"] = Field(
        description="Overall tone of management"
    )
    confidence_level: float = Field(
        description="Management's confidence level 0.0-1.0", ge=0.0, le=1.0
    )

    # Key themes and highlights
    key_themes: list[str] = Field(
        description="3-5 major themes from the call", min_length=2, max_length=5
    )
    positive_highlights: list[str] = Field(
        description="Key positive points mentioned", max_length=5
    )
    risks_and_concerns: list[str] = Field(
        description="Risks and concerns mentioned", max_length=5
    )

    # Summary
    executive_summary: str = Field(
        description="2-3 sentence executive summary for investors"
    )


def main():
    # Load API key from .env at repository root
    repo_root = Path(__file__).resolve().parents[2]
    config = Config(RepositoryEnv(repo_root / ".env"))

    client = OpenAI(api_key=config("OPENAI_API_KEY"))

    # Load sample earnings call transcript
    data_dir = Path(__file__).parent / "data"
    transcript_path = data_dir / "sample_earnings.txt"
    transcript = transcript_path.read_text()

    print("=" * 70)
    print("EARNINGS CALL ANALYZER")
    print("=" * 70)
    print(f"\nAnalyzing: {transcript_path.name}")
    print(f"Length: {len(transcript)} characters\n")

    messages = [
        {
            "role": "system",
            "content": """You are a financial analyst specializing in earnings call analysis.
            Extract key information from the provided earnings call transcript.
            Be precise with numbers and provide insightful qualitative analysis.""",
        },
        {
            "role": "user",
            "content": f"Analyze this earnings call transcript:\n\n{transcript}",
        },
    ]

    print("Extracting structured analysis...\n")

    response = client.beta.chat.completions.parse(
        model="gpt-4o-mini",
        messages=messages,
        response_format=EarningsAnalysis,
    )

    analysis: EarningsAnalysis = response.choices[0].message.parsed

    # Display results
    print(f"Company: {analysis.company_name} ({analysis.ticker})")
    print(f"Quarter: {analysis.quarter}")
    print(f"Sentiment: {analysis.overall_sentiment.upper()}")
    print(f"Management Confidence: {analysis.confidence_level:.0%}")

    print("\n--- FINANCIAL METRICS ---")
    m = analysis.metrics
    if m.revenue_billions:
        print(f"Revenue: ${m.revenue_billions}B", end="")
        if m.revenue_growth_pct:
            print(f" ({m.revenue_growth_pct:+.1f}% YoY)")
        else:
            print()
    if m.gross_margin_pct:
        print(f"Gross Margin: {m.gross_margin_pct}%")

    print("\n--- GUIDANCE ---")
    g = analysis.guidance
    if g.revenue_growth_guidance:
        print(f"Revenue: {g.revenue_growth_guidance}")
    if g.gross_margin_guidance:
        print(f"Gross Margin: {g.gross_margin_guidance}")

    print("\n--- KEY THEMES ---")
    for theme in analysis.key_themes:
        print(f"  • {theme}")

    print("\n--- POSITIVE HIGHLIGHTS ---")
    for highlight in analysis.positive_highlights:
        print(f"  ✓ {highlight}")

    print("\n--- RISKS & CONCERNS ---")
    for risk in analysis.risks_and_concerns:
        print(f"  ⚠ {risk}")

    print("\n--- EXECUTIVE SUMMARY ---")
    print(analysis.executive_summary)

    # Save full analysis as JSON
    output_path = data_dir / "analysis_output.json"
    output_path.write_text(analysis.model_dump_json(indent=2))
    print(f"\nFull analysis saved to: {output_path}")


if __name__ == "__main__":
    main()
