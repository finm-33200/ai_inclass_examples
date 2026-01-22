"""
Create Chart
============

This script creates visualizations from the processed FRED data:
- GDP over time with year-over-year change
- Unemployment rate with 12-month rolling average

Charts are saved to _output/ as PNG files.
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# Import settings
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from settings import DATA_DIR, OUTPUT_DIR


def create_gdp_chart():
    """Create a chart showing GDP over time."""
    print("Loading processed data...")
    df = pd.read_parquet(DATA_DIR / "processed.parquet")

    print("Creating GDP chart...")

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    # Plot 1: GDP level
    ax1 = axes[0]
    ax1.plot(df.index, df["GDP"], color="steelblue", linewidth=2)
    ax1.set_ylabel("GDP (Billions of $)", fontsize=12)
    ax1.set_title("US Gross Domestic Product", fontsize=14, fontweight="bold")
    ax1.grid(True, alpha=0.3)
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"${x/1000:.0f}T"))

    # Plot 2: Year-over-year change
    ax2 = axes[1]
    colors = ["green" if x >= 0 else "red" for x in df["GDP_YoY_Change"]]
    ax2.bar(df.index, df["GDP_YoY_Change"], color=colors, alpha=0.7, width=50)
    ax2.axhline(y=0, color="black", linestyle="-", linewidth=0.5)
    ax2.set_ylabel("YoY Change (%)", fontsize=12)
    ax2.set_xlabel("Date", fontsize=12)
    ax2.grid(True, alpha=0.3)

    # Format x-axis
    ax2.xaxis.set_major_locator(mdates.YearLocator(5))
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    plt.tight_layout()

    # Save chart
    output_path = OUTPUT_DIR / "gdp_chart.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"Saved GDP chart to {output_path}")


def create_unemployment_chart():
    """Create a chart showing unemployment rate over time."""
    print("Creating unemployment chart...")
    df = pd.read_parquet(DATA_DIR / "processed.parquet")

    fig, ax = plt.subplots(figsize=(12, 6))

    # Plot unemployment rate
    ax.plot(
        df.index,
        df["UNRATE"],
        color="steelblue",
        linewidth=1,
        alpha=0.5,
        label="Monthly Rate",
    )
    ax.plot(
        df.index,
        df["UNRATE_12M_Avg"],
        color="darkred",
        linewidth=2,
        label="12-Month Average",
    )

    ax.set_ylabel("Unemployment Rate (%)", fontsize=12)
    ax.set_xlabel("Date", fontsize=12)
    ax.set_title("US Unemployment Rate", fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right")

    # Format x-axis
    ax.xaxis.set_major_locator(mdates.YearLocator(5))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    plt.tight_layout()

    # Save chart
    output_path = OUTPUT_DIR / "unemployment_chart.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"Saved unemployment chart to {output_path}")


def create_all_charts():
    """Create all charts."""
    create_gdp_chart()
    create_unemployment_chart()
    print("\nAll charts created successfully!")


if __name__ == "__main__":
    from settings import ensure_directories
    ensure_directories()
    create_all_charts()
