"""
dodo.py - Task Runner Configuration
===================================

This dodo.py demonstrates a typical data science workflow:

1. task_config      - Set up directories
2. task_pull        - Pull data from FRED
3. task_process     - Process the raw data
4. task_chart       - Create visualizations

Run with:
    doit              # Run all tasks
    doit list         # List available tasks
    doit pull         # Run just the pull task (and dependencies)
    doit clean        # Remove all generated files
    doit forget       # Clear task state database
"""

import sys
from pathlib import Path

# Add src to path so we can import our modules
SRC_DIR = Path(__file__).parent / "src"
sys.path.insert(0, str(SRC_DIR))

from settings import DATA_DIR, OUTPUT_DIR


# =============================================================================
# Task: Configuration / Setup
# =============================================================================

def task_config():
    """Create required directories."""

    def create_dirs():
        DATA_DIR.mkdir(exist_ok=True)
        OUTPUT_DIR.mkdir(exist_ok=True)
        print(f"Created directories: {DATA_DIR}, {OUTPUT_DIR}")

    return {
        "actions": [create_dirs],
        "targets": [DATA_DIR, OUTPUT_DIR],
        "uptodate": [True],  # Only run once (dirs already exist check)
    }


# =============================================================================
# Task: Pull FRED Data
# =============================================================================

def task_pull():
    """Pull economic data from FRED."""
    return {
        "actions": [f"python ./src/pull_fred.py"],
        "file_dep": ["./src/settings.py", "./src/pull_fred.py"],
        "targets": [DATA_DIR / "fred.parquet"],
        "clean": True,
        "verbosity": 2,
    }


# =============================================================================
# Task: Process Data
# =============================================================================

def task_process():
    """Process the raw FRED data."""
    return {
        "actions": [f"python ./src/process_data.py"],
        "file_dep": [
            "./src/settings.py",
            "./src/process_data.py",
            DATA_DIR / "fred.parquet",  # Depends on pull task output
        ],
        "targets": [
            DATA_DIR / "processed.parquet",
            DATA_DIR / "summary_stats.csv",
        ],
        "clean": True,
        "verbosity": 2,
    }


# =============================================================================
# Task: Create Charts
# =============================================================================

def task_chart():
    """Create visualizations from processed data."""
    return {
        "actions": [f"python ./src/create_chart.py"],
        "file_dep": [
            "./src/settings.py",
            "./src/create_chart.py",
            DATA_DIR / "processed.parquet",  # Depends on process task output
        ],
        "targets": [
            OUTPUT_DIR / "gdp_chart.png",
            OUTPUT_DIR / "unemployment_chart.png",
        ],
        "clean": True,
        "verbosity": 2,
    }
