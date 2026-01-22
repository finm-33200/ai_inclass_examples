# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.18.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # 01. GDP Over Time
#
# This notebook visualizes US Gross Domestic Product (GDP) over time using
# interactive Plotly charts. The data is pulled from FRED (Federal Reserve
# Economic Data).

# %%
from pathlib import Path

import plotly.express as px
import pull_fred
from settings import config

OUTPUT_DIR = Path(config("OUTPUT_DIR"))

# %%
# Load the FRED data
df = pull_fred.load_fred()
df

# %%
# Create GDP chart
gdp_df = df[["GDP"]].dropna().reset_index()
gdp_df.columns = ["Date", "GDP"]

fig = px.line(
    gdp_df,
    x="Date",
    y="GDP",
    title="US Gross Domestic Product (GDP)",
    labels={"GDP": "GDP (Billions of Dollars)", "Date": "Date"},
)

fig.update_layout(
    hovermode="x unified",
    template="plotly_white",
)

# Display the chart
fig

# %%
# Save the chart as HTML
chart_path = OUTPUT_DIR / "01_gdp_chart.html"
fig.write_html(chart_path)
print(f"Chart saved to: {chart_path}")

# %%
