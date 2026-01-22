from pathlib import Path

import pull_fred
from settings import config

DATA_DIR = Path(config("DATA_DIR"))
OUTPUT_DIR = Path(config("OUTPUT_DIR"))

import seaborn as sns
from matplotlib import pyplot as plt

sns.set()

df = pull_fred.load_fred(data_dir=DATA_DIR)

(
    100
    * df[["CPIAUCSL", "GDP"]]
    .rename(columns={"CPIAUCSL": "Inflation (CPI)", "GDP": "GDP"})
    .dropna()
    .pct_change(4)
).plot()
plt.title("Inflation and GDP, Seasonally Adjusted")
plt.ylabel("Percent change from 12-months prior")
filename = OUTPUT_DIR / "example_plot.png"
plt.savefig(filename)
