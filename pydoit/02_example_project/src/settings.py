"""
Settings Module
===============

Centralized configuration for the project.
All paths and settings are defined here for easy management.
"""

from pathlib import Path

# Base directory is the project root (parent of src/)
BASE_DIR = Path(__file__).absolute().parent.parent

# Directory for auto-generated data (gitignored)
DATA_DIR = BASE_DIR / "_data"

# Directory for output files like charts (gitignored)
OUTPUT_DIR = BASE_DIR / "_output"

# Date range for FRED data
START_DATE = "2000-01-01"
END_DATE = "2024-12-31"

# FRED series to pull
FRED_SERIES = {
    "GDP": "GDP",           # Gross Domestic Product
    "UNRATE": "UNRATE",     # Unemployment Rate
}


def ensure_directories():
    """Create data and output directories if they don't exist."""
    DATA_DIR.mkdir(exist_ok=True)
    OUTPUT_DIR.mkdir(exist_ok=True)
    print(f"Ensured directories exist:")
    print(f"  DATA_DIR: {DATA_DIR}")
    print(f"  OUTPUT_DIR: {OUTPUT_DIR}")


if __name__ == "__main__":
    ensure_directories()
    print(f"\nSettings:")
    print(f"  BASE_DIR: {BASE_DIR}")
    print(f"  START_DATE: {START_DATE}")
    print(f"  END_DATE: {END_DATE}")
