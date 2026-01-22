"""Pull basic economic time series from FRED (Federal Reserve Economic Data).

This module fetches a few key economic indicators from FRED and saves them
to a parquet file for use in analysis notebooks.
"""

from pathlib import Path

import pandas as pd
import pandas_datareader.data as web
from settings import config

DATA_DIR = Path(config("DATA_DIR"))
START_DATE = config("START_DATE")
END_DATE = config("END_DATE")


# Define the series to pull from FRED
series_to_pull = {
    "GDP": "Gross Domestic Product",
    "UNRATE": "Unemployment Rate",
    "CPIAUCSL": "Consumer Price Index for All Urban Consumers",
    "FEDFUNDS": "Federal Funds Effective Rate",
}

series_descriptions = series_to_pull.copy()


def pull_fred(start_date=START_DATE, end_date=END_DATE):
    """
    Pull economic data from FRED.

    Lookup series code, e.g., like this:
    https://fred.stlouisfed.org/series/GDP

    Parameters
    ----------
    start_date : str or datetime
        Start date for the data pull
    end_date : str or datetime
        End date for the data pull

    Returns
    -------
    pd.DataFrame
        DataFrame with FRED time series data indexed by date
    """
    df = web.DataReader(list(series_to_pull.keys()), "fred", start_date, end_date)
    return df


def load_fred(data_dir=DATA_DIR):
    """
    Load previously saved FRED data from parquet file.

    Must first run this module as main to pull and save data.

    Parameters
    ----------
    data_dir : Path
        Directory containing the fred.parquet file

    Returns
    -------
    pd.DataFrame
        DataFrame with FRED time series data
    """
    file_path = Path(data_dir) / "fred.parquet"
    df = pd.read_parquet(file_path)
    return df


def demo():
    """Demonstrate loading FRED data."""
    df = load_fred()
    print(df.head())
    print(f"\nColumns: {list(df.columns)}")
    print(f"Date range: {df.index.min()} to {df.index.max()}")


if __name__ == "__main__":
    today = pd.Timestamp.today().strftime("%Y-%m-%d")
    end_date = today
    df = pull_fred(START_DATE, end_date)
    filedir = Path(DATA_DIR)
    filedir.mkdir(parents=True, exist_ok=True)
    df.to_parquet(filedir / "fred.parquet")
    df.to_csv(filedir / "fred.csv")
    print(f"Saved FRED data to {filedir}")
    print(f"Series: {list(series_to_pull.keys())}")
    print(f"Date range: {df.index.min()} to {df.index.max()}")
