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
# # 02. Unemployment Rate Over Time
#
# This notebook visualizes the US Unemployment Rate over time using
# interactive Plotly charts. The data is pulled from FRED (Federal Reserve
# Economic Data).
#
# This notebook demonstrates:
# - Dependencies on external data pulls (pull_fred.py)
# - Saving interactive HTML charts with Plotly

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
# Create Unemployment Rate chart
unrate_df = df[["UNRATE"]].dropna().reset_index()
unrate_df.columns = ["Date", "Unemployment Rate"]

fig = px.line(
    unrate_df,
    x="Date",
    y="Unemployment Rate",
    title="US Unemployment Rate",
    labels={"Unemployment Rate": "Unemployment Rate (%)", "Date": "Date"},
)

fig.update_layout(
    hovermode="x unified",
    template="plotly_white",
)

# Display the chart
fig

# %%
# Save the chart as HTML
chart_path = OUTPUT_DIR / "02_unemployment_chart.html"
fig.write_html(chart_path)
print(f"Chart saved to: {chart_path}")

# %%
