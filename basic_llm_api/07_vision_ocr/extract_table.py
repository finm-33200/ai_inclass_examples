"""
07_vision_ocr/extract_table.py

Extract structured data from a table image.
Uses vision + structured outputs for reliable data extraction.
"""

import base64
import io
import json
from pathlib import Path
from typing import Literal

from decouple import Config, RepositoryEnv
from openai import OpenAI
from pydantic import BaseModel, Field

import matplotlib.pyplot as plt
import numpy as np


class FinancialRow(BaseModel):
    """A single row from a financial table."""

    company: str
    ticker: str
    revenue_billions: float
    growth_pct: float
    pe_ratio: float


class FinancialTable(BaseModel):
    """Extracted financial table data."""

    title: str = Field(description="Title of the table")
    rows: list[FinancialRow] = Field(description="Data rows from the table")
    source: str | None = Field(description="Source if mentioned", default=None)


def generate_sample_table() -> bytes:
    """Generate a sample financial table as a PNG image."""
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.axis("off")

    # Table data
    headers = ["Company", "Ticker", "Revenue ($B)", "YoY Growth", "P/E Ratio"]
    data = [
        ["Apple Inc.", "AAPL", "383.3", "+2.8%", "28.5"],
        ["Microsoft", "MSFT", "211.9", "+15.6%", "35.2"],
        ["Alphabet", "GOOGL", "307.4", "+8.7%", "24.1"],
        ["Amazon", "AMZN", "574.8", "+11.8%", "62.3"],
        ["NVIDIA", "NVDA", "60.9", "+125.9%", "65.8"],
    ]

    # Create table
    table = ax.table(
        cellText=data,
        colLabels=headers,
        cellLoc="center",
        loc="center",
        colColours=["#4472C4"] * len(headers),
    )

    # Style the table
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 1.8)

    # Style header row
    for j in range(len(headers)):
        table[(0, j)].set_text_props(color="white", weight="bold")

    # Add title
    ax.set_title(
        "Tech Giants - Q4 2024 Financial Summary", fontsize=14, weight="bold", pad=20
    )

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

    # Generate sample table image
    print("Generating sample financial table...")
    image_bytes = generate_sample_table()

    # Save for reference
    sample_path = Path(__file__).parent / "sample_images" / "financial_table.png"
    sample_path.write_bytes(image_bytes)
    print(f"Saved to: {sample_path}\n")

    # Encode as base64
    b64_image = base64.b64encode(image_bytes).decode("utf-8")

    # Extract structured data using vision + Pydantic
    messages = [
        {
            "role": "system",
            "content": "Extract the data from this financial table image into structured format.",
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "Extract all the data from this table.",
                },
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{b64_image}"},
                },
            ],
        },
    ]

    print("Extracting table data with GPT-4o...\n")

    response = client.beta.chat.completions.parse(
        model="gpt-4o-mini",
        messages=messages,
        response_format=FinancialTable,
    )

    table_data: FinancialTable = response.choices[0].message.parsed

    # Display extracted data
    print(f"Table Title: {table_data.title}\n")
    print(f"{'Company':<15} {'Ticker':<8} {'Revenue':<12} {'Growth':<10} {'P/E':<8}")
    print("-" * 55)
    for row in table_data.rows:
        print(
            f"{row.company:<15} {row.ticker:<8} ${row.revenue_billions:<10.1f} {row.growth_pct:>+.1f}%{'':<4} {row.pe_ratio:<8.1f}"
        )

    print("\n--- As JSON ---")
    print(table_data.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
