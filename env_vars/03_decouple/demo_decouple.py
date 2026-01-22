"""
Demonstrates python-decouple with defaults and type casting.

Before running:
    cp .env.example .env
    python demo_decouple.py
"""
from decouple import config

# config() reads from .env or environment variables
# default= provides a fallback if not set
# cast= converts the string value to the specified type

debug = config("DEBUG", default=False, cast=bool)
data_dir = config("DATA_DIR", default="./_data")
max_rows = config("MAX_ROWS", default=100, cast=int)

print(f"DEBUG = {debug} (type: {type(debug).__name__})")
print(f"DATA_DIR = {data_dir}")
print(f"MAX_ROWS = {max_rows} (type: {type(max_rows).__name__})")
