"""
07_vision_ocr/read_chart.py

Analyze a financial chart and extract insights.
Demonstrates deeper chart analysis capabilities.
"""

import base64
import io
from pathlib import Path

from decouple import Config, RepositoryEnv
from openai import OpenAI
from pydantic import BaseModel, Field

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
from datetime import datetime, timedelta


class ChartAnalysis(BaseModel):
    """Structured analysis of a financial chart."""

    chart_type: str = Field(description="Type of chart (line, bar, candlestick, etc.)")
    title: str = Field(description="Title of the chart")
    time_period: str = Field(description="Time period covered by the chart")
    overall_trend: str = Field(
        description="Overall trend: upward, downward, sideways, or volatile"
    )
    key_observations: list[str] = Field(
        description="3-5 key observations from the chart"
    )
    support_level: float | None = Field(
        description="Approximate support level if visible", default=None
    )
    resistance_level: float | None = Field(
        description="Approximate resistance level if visible", default=None
    )


def generate_candlestick_style_chart() -> bytes:
    """Generate a chart that looks like a financial visualization."""
    np.random.seed(123)

    # Generate dates
    start_date = datetime(2024, 1, 1)
    dates = [start_date + timedelta(days=i) for i in range(0, 250, 1)]
    dates = [d for d in dates if d.weekday() < 5][:180]  # Trading days only

    # Generate price data with trends
    base_price = 150
    returns = np.random.randn(len(dates)) * 0.015
    # Add trend
    trend = np.linspace(0, 0.3, len(dates))
    returns = returns + trend / len(dates)
    prices = base_price * np.cumprod(1 + returns)

    # Add a correction in the middle
    prices[60:90] = prices[60:90] * np.linspace(1, 0.85, 30)
    prices[90:] = prices[90:] * 0.85

    # Moving averages
    ma20 = np.convolve(prices, np.ones(20) / 20, mode="valid")
    ma50 = np.convolve(prices, np.ones(50) / 50, mode="valid")

    # Create figure
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(12, 8), height_ratios=[3, 1], sharex=True
    )
    fig.suptitle("TECH ETF - 2024 YTD Performance", fontsize=14, weight="bold")

    # Price chart
    ax1.plot(dates, prices, "b-", linewidth=1.5, label="Price")
    ax1.plot(dates[19:], ma20, "orange", linewidth=1, label="20-day MA", alpha=0.8)
    ax1.plot(dates[49:], ma50, "red", linewidth=1, label="50-day MA", alpha=0.8)

    # Support and resistance lines
    ax1.axhline(y=140, color="green", linestyle="--", alpha=0.5, label="Support")
    ax1.axhline(y=175, color="red", linestyle="--", alpha=0.5, label="Resistance")

    ax1.set_ylabel("Price ($)")
    ax1.legend(loc="upper left")
    ax1.grid(True, alpha=0.3)

    # Volume chart (simulated)
    volumes = np.random.randint(1000000, 5000000, len(dates))
    volumes[60:90] = volumes[60:90] * 2  # Higher volume during correction
    colors = ["g" if prices[i] > prices[i - 1] else "r" for i in range(1, len(prices))]
    colors = ["g"] + colors

    ax2.bar(dates, volumes, color=colors, alpha=0.7, width=0.8)
    ax2.set_ylabel("Volume")
    ax2.set_xlabel("Date")
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    ax2.xaxis.set_major_locator(mdates.MonthLocator())

    plt.tight_layout()

    # Save to bytes
    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    buf.seek(0)
    return buf.read()


def main():
    # Load API key from .env at repository root
    repo_root = Path(__file__).resolve().parents[2]
    config = Config(RepositoryEnv(repo_root / ".env"))

    client = OpenAI(api_key=config("OPENAI_API_KEY"))

    # Generate chart
    print("Generating sample financial chart...")
    image_bytes = generate_candlestick_style_chart()

    # Save for reference
    sample_path = Path(__file__).parent / "sample_images" / "tech_etf_chart.png"
    sample_path.write_bytes(image_bytes)
    print(f"Saved to: {sample_path}\n")

    # Encode as base64
    b64_image = base64.b64encode(image_bytes).decode("utf-8")

    # Analyze with structured output
    messages = [
        {
            "role": "system",
            "content": """You are a technical analyst. Analyze the provided chart
            and extract key insights about price trends, support/resistance levels,
            and notable patterns.""",
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "Analyze this financial chart in detail.",
                },
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{b64_image}"},
                },
            ],
        },
    ]

    print("Analyzing chart with GPT-4o...\n")

    response = client.beta.chat.completions.parse(
        model="gpt-4o-mini",
        messages=messages,
        response_format=ChartAnalysis,
    )

    analysis: ChartAnalysis = response.choices[0].message.parsed

    print(f"Chart Type: {analysis.chart_type}")
    print(f"Title: {analysis.title}")
    print(f"Time Period: {analysis.time_period}")
    print(f"Overall Trend: {analysis.overall_trend}")

    if analysis.support_level:
        print(f"Support Level: ${analysis.support_level:.2f}")
    if analysis.resistance_level:
        print(f"Resistance Level: ${analysis.resistance_level:.2f}")

    print("\nKey Observations:")
    for i, obs in enumerate(analysis.key_observations, 1):
        print(f"  {i}. {obs}")


if __name__ == "__main__":
    main()
