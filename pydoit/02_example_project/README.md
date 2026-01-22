# 02_example_project: Python Scripts Workflow

This example demonstrates a realistic data science workflow using PyDoit with Python scripts (no notebooks).

## What This Example Demonstrates

- **Modular project structure**: Separate scripts for each step
- **Settings module**: Centralized configuration
- **External data**: Pulling from FRED API
- **File dependencies**: Each script depends on outputs from previous steps
- **Data processing pipeline**: Pull → Process → Visualize

## Project Structure

```
02_example_project/
├── dodo.py              # Task runner configuration
├── src/
│   ├── settings.py      # Configuration (paths, dates, etc.)
│   ├── pull_fred.py     # Fetch data from FRED
│   ├── process_data.py  # Clean and transform data
│   └── create_chart.py  # Generate visualizations
├── _data/               # Auto-generated data (gitignored)
├── _output/             # Output charts (gitignored)
├── .gitignore
└── README.md
```

## The Workflow

```
task_config ──► task_pull ──► task_process ──► task_chart
     │              │              │               │
     ▼              ▼              ▼               ▼
 directories   fred.parquet   processed.parquet  *.png charts
```

## Requirements

```bash
pip install doit pandas pandas-datareader matplotlib pyarrow
```

## How to Run

```bash
# Run the complete workflow
doit

# List all tasks
doit list

# Run only up to a specific task
doit process

# See what would run (dry-run)
doit list --all --status

# Clean all generated files
doit clean

# Force re-run everything
doit forget && doit
```

## Key Concepts Demonstrated

### 1. Settings Module (`src/settings.py`)

Centralized configuration makes it easy to:
- Change date ranges
- Modify output paths
- Add new data series

### 2. Separate Scripts

Each step is a standalone Python script that can be:
- Run independently for testing
- Modified without affecting other steps
- Reused in other projects

### 3. File Dependencies

dodo.py tracks:
- Which scripts are needed for each task
- Which data files each task needs
- Which files each task produces

### 4. Automatic Re-running

If you modify `pull_fred.py`:
- `task_pull` will re-run
- `task_process` will re-run (depends on fred.parquet)
- `task_chart` will re-run (depends on processed.parquet)

If you only modify `create_chart.py`:
- Only `task_chart` will re-run

## Data Sources

This example pulls from FRED (Federal Reserve Economic Data):
- **GDP**: Gross Domestic Product
- **UNRATE**: Unemployment Rate

## Output

After running `doit`, you'll have:
- `_data/fred.parquet` - Raw FRED data
- `_data/processed.parquet` - Processed data with calculated fields
- `_data/summary_stats.csv` - Summary statistics
- `_output/gdp_chart.png` - GDP visualization
- `_output/unemployment_chart.png` - Unemployment visualization
