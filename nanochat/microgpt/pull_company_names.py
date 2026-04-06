"""
Pull unique company names from Compustat via WRDS and save to company_names.txt.

Uses sqlalchemy + pgpass for non-interactive authentication.
"""

from pathlib import Path

import pandas as pd
from sqlalchemy import create_engine
from decouple import Config, RepositoryEnv

repo_root = Path(__file__).resolve().parents[2]
config = Config(RepositoryEnv(repo_root / ".env"))
WRDS_USERNAME = config("WRDS_USERNAME")

OUTPUT_PATH = Path(__file__).resolve().parent / "company_names.txt"

query = """
    SELECT DISTINCT conm
    FROM comp.company
    WHERE conm IS NOT NULL
    ORDER BY conm
"""

engine = create_engine(f"postgresql://{WRDS_USERNAME}@wrds-pgdata.wharton.upenn.edu:9737/wrds")
df = pd.read_sql(query, engine)
engine.dispose()

names = df["conm"].str.strip().unique()
names = sorted(n for n in names if len(n) > 0)

with open(OUTPUT_PATH, "w") as f:
    for name in names:
        f.write(name + "\n")

print(f"Wrote {len(names)} unique company names to {OUTPUT_PATH}")
