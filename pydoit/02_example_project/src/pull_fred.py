"""
Pull FRED Data
==============

This script fetches economic data from FRED (Federal Reserve Economic Data)
using the pandas-datareader library.

Data pulled:
- GDP: Gross Domestic Product
- UNRATE: Unemployment Rate

The data is saved to _data/fred.parquet for use by other scripts.
"""

import pandas as pd
from pandas_datareader import data as pdr

# Import settings
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from settings import DATA_DIR, START_DATE, END_DATE, FRED_SERIES


def pull_fred_data():
    """
    Pull data from FRED and save to parquet file.

    Returns:
        pd.DataFrame: The pulled data
    """
    print(f"Pulling FRED data from {START_DATE} to {END_DATE}...")

    # Pull each series
    dfs = []
    for name, series_id in FRED_SERIES.items():
        print(f"  Fetching {name} ({series_id})...")
        df = pdr.DataReader(series_id, "fred", START_DATE, END_DATE)
        df.columns = [name]
        dfs.append(df)

    # Combine all series
    result = pd.concat(dfs, axis=1)

    # Save to parquet
    output_path = DATA_DIR / "fred.parquet"
    result.to_parquet(output_path)
    print(f"\nSaved data to {output_path}")
    print(f"Shape: {result.shape}")
    print(f"Columns: {list(result.columns)}")
    print(f"Date range: {result.index.min()} to {result.index.max()}")

    return result


def load_fred_data():
    """
    Load previously pulled FRED data from parquet file.

    Returns:
        pd.DataFrame: The loaded data
    """
    path = DATA_DIR / "fred.parquet"
    return pd.read_parquet(path)


if __name__ == "__main__":
    # Ensure directories exist
    from settings import ensure_directories
    ensure_directories()

    # Pull the data
    df = pull_fred_data()
    print("\nData preview:")
    print(df.head(10))
