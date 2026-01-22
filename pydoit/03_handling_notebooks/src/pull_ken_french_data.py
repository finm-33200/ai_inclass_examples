""" """

import warnings
import pandas_datareader.data as web

from settings import config

DATA_DIR = config("DATA_DIR")
START_DATE = config("START_DATE")
END_DATE = config("END_DATE")


def pull_ken_french_data(start_date=START_DATE, end_date=END_DATE):
    # Suppress the specific FutureWarning about date_parser
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            category=FutureWarning,
            message="The argument 'date_parser' is deprecated",
        )
        data = web.DataReader(
            "25_Portfolios_OP_INV_5x5_daily", "famafrench", start=start_date, end=end_date
        )
    return data


if __name__ == "__main__":
    data = pull_ken_french_data(start_date=START_DATE, end_date=END_DATE)
    data[0].to_parquet(DATA_DIR / "25_Portfolios_OP_INV_5x5_daily.parquet")
