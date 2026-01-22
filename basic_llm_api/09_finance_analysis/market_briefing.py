"""
09_finance_analysis/market_briefing.py

Generate a market briefing using LLM summarization.
Demonstrates combining multiple pieces of information into a coherent narrative.
"""

from pathlib import Path
from typing import Literal

from decouple import Config, RepositoryEnv
from openai import OpenAI
from pydantic import BaseModel, Field


class MarketData(BaseModel):
    """Simulated market data for the briefing."""

    spy_change: float
    qqq_change: float
    vix: float
    ten_year_yield: float
    wti_crude: float
    gold: float


class SectorPerformance(BaseModel):
    """Sector-level performance."""

    sector: str
    change_pct: float
    top_mover: str
    top_mover_change: float


class MarketBriefing(BaseModel):
    """Structured market briefing output."""

    headline: str = Field(description="Attention-grabbing headline for the day")
    market_summary: str = Field(description="2-3 sentence market overview")
    key_drivers: list[str] = Field(
        description="3-5 key factors driving today's market", min_length=3, max_length=5
    )
    sector_highlights: list[str] = Field(
        description="Notable sector movements", max_length=4
    )
    risks_to_watch: list[str] = Field(description="Key risks for the coming days")
    outlook: Literal["bullish", "bearish", "neutral"] = Field(
        description="Short-term market outlook"
    )


def get_market_data() -> tuple[MarketData, list[SectorPerformance]]:
    """Return simulated market data. In production, fetch from a data provider."""
    market = MarketData(
        spy_change=0.82,
        qqq_change=1.24,
        vix=14.5,
        ten_year_yield=4.32,
        wti_crude=78.50,
        gold=2045.30,
    )

    sectors = [
        SectorPerformance(
            sector="Technology",
            change_pct=1.45,
            top_mover="NVDA",
            top_mover_change=3.2,
        ),
        SectorPerformance(
            sector="Financials",
            change_pct=0.65,
            top_mover="JPM",
            top_mover_change=1.8,
        ),
        SectorPerformance(
            sector="Energy",
            change_pct=-0.42,
            top_mover="XOM",
            top_mover_change=-0.9,
        ),
        SectorPerformance(
            sector="Healthcare",
            change_pct=0.28,
            top_mover="UNH",
            top_mover_change=1.1,
        ),
        SectorPerformance(
            sector="Consumer Discretionary",
            change_pct=0.95,
            top_mover="AMZN",
            top_mover_change=2.1,
        ),
    ]

    return market, sectors


def get_news_headlines() -> list[str]:
    """Return simulated news headlines. In production, fetch from a news API."""
    return [
        "Fed minutes suggest rate cuts likely in coming months",
        "Tech earnings season begins with strong results from major chipmakers",
        "China manufacturing data shows signs of recovery",
        "Treasury yields stabilize after recent volatility",
        "Oil prices edge lower on demand concerns",
    ]


def main():
    # Load API key from .env at repository root
    repo_root = Path(__file__).resolve().parents[2]
    config = Config(RepositoryEnv(repo_root / ".env"))

    client = OpenAI(api_key=config("OPENAI_API_KEY"))

    # Get market data
    market, sectors = get_market_data()
    headlines = get_news_headlines()

    # Format data for the LLM
    market_context = f"""
TODAY'S MARKET DATA:
- S&P 500 (SPY): {market.spy_change:+.2f}%
- Nasdaq 100 (QQQ): {market.qqq_change:+.2f}%
- VIX: {market.vix:.1f}
- 10-Year Treasury Yield: {market.ten_year_yield:.2f}%
- WTI Crude: ${market.wti_crude:.2f}
- Gold: ${market.gold:.2f}

SECTOR PERFORMANCE:
{chr(10).join(f"- {s.sector}: {s.change_pct:+.2f}% (Top mover: {s.top_mover} {s.top_mover_change:+.1f}%)" for s in sectors)}

TODAY'S HEADLINES:
{chr(10).join(f"- {h}" for h in headlines)}
"""

    print("=" * 70)
    print("MARKET BRIEFING GENERATOR")
    print("=" * 70)
    print("\nInput Data:")
    print(market_context)

    messages = [
        {
            "role": "system",
            "content": """You are a professional market analyst writing a daily briefing.
            Generate insightful, actionable commentary based on the provided market data.
            Your tone should be professional yet accessible to retail investors.""",
        },
        {
            "role": "user",
            "content": f"Generate a market briefing based on this data:\n\n{market_context}",
        },
    ]

    print("\nGenerating briefing...\n")

    response = client.beta.chat.completions.parse(
        model="gpt-4o-mini",
        messages=messages,
        response_format=MarketBriefing,
    )

    briefing: MarketBriefing = response.choices[0].message.parsed

    # Display the briefing
    print("=" * 70)
    print(f"📈 {briefing.headline}")
    print("=" * 70)

    print(f"\n{briefing.market_summary}")

    print("\n📊 KEY DRIVERS")
    for driver in briefing.key_drivers:
        print(f"  • {driver}")

    print("\n🏢 SECTOR HIGHLIGHTS")
    for highlight in briefing.sector_highlights:
        print(f"  • {highlight}")

    print("\n⚠️ RISKS TO WATCH")
    for risk in briefing.risks_to_watch:
        print(f"  • {risk}")

    outlook_emoji = {"bullish": "🟢", "bearish": "🔴", "neutral": "🟡"}
    print(f"\n📍 SHORT-TERM OUTLOOK: {outlook_emoji[briefing.outlook]} {briefing.outlook.upper()}")


if __name__ == "__main__":
    main()
