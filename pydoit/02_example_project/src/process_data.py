"""
Process Data
============

This script processes the raw FRED data:
- Calculates summary statistics
- Computes year-over-year percent changes
- Saves processed data to _data/processed.parquet
"""

import pandas as pd

# Import settings
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from settings import DATA_DIR


def process_fred_data():
    """
    Process the FRED data and save results.

    Processing steps:
    1. Load raw data
    2. Calculate year-over-year percent changes
    3. Calculate rolling statistics
    4. Save processed data

    Returns:
        pd.DataFrame: The processed data
    """
    print("Loading raw FRED data...")
    df = pd.read_parquet(DATA_DIR / "fred.parquet")

    print("Processing data...")

    # Calculate year-over-year percent change for GDP
    if "GDP" in df.columns:
        df["GDP_YoY_Change"] = df["GDP"].pct_change(periods=4) * 100  # Quarterly data

    # Calculate 12-month rolling average for unemployment
    if "UNRATE" in df.columns:
        df["UNRATE_12M_Avg"] = df["UNRATE"].rolling(window=12, min_periods=1).mean()

    # Drop rows with NaN values for cleaner output
    df_clean = df.dropna()

    # Save processed data
    output_path = DATA_DIR / "processed.parquet"
    df_clean.to_parquet(output_path)

    print(f"Saved processed data to {output_path}")
    print(f"Shape: {df_clean.shape}")
    print(f"Columns: {list(df_clean.columns)}")

    # Also save summary statistics
    summary = df.describe()
    summary_path = DATA_DIR / "summary_stats.csv"
    summary.to_csv(summary_path)
    print(f"Saved summary statistics to {summary_path}")

    return df_clean


def load_processed_data():
    """
    Load the processed data from parquet file.

    Returns:
        pd.DataFrame: The processed data
    """
    path = DATA_DIR / "processed.parquet"
    return pd.read_parquet(path)


if __name__ == "__main__":
    df = process_fred_data()
    print("\nProcessed data preview:")
    print(df.tail(10))
