# PyDoit Teaching Examples

This directory contains a series of progressively more complex examples demonstrating how to use **PyDoit** (doit) as a task runner for data science workflows.

## What is PyDoit?

PyDoit is a Python-based task automation tool similar to GNU Make. It helps you:

- **Automate workflows**: Define tasks and their dependencies
- **Track changes**: Only re-run tasks when inputs change
- **Document pipelines**: The `dodo.py` file becomes self-documenting
- **Ensure reproducibility**: Run the entire workflow with a single command

## Examples Overview

| Example | Complexity | Key Concepts |
|---------|------------|--------------|
| [01_simplest](01_simplest/) | Beginner | Basic tasks, file dependencies, Python functions |
| [02_example_project](02_example_project/) | Intermediate | Project structure, Python scripts, FRED data |
| [03_handling_notebooks](03_handling_notebooks/) | Advanced | Regular `.ipynb` notebooks, two-stage workflow |
| [04_jupytext_notebooks](04_jupytext_notebooks/) | Advanced | Jupytext `.py` notebooks, Sphinx documentation |

## Learning Path

### 1. Start with [01_simplest](01_simplest/)

The most basic example with 3 tasks that depend on each other:

```python
def task_create_data():
    return {
        'actions': [create_data],
        'targets': ['_output/data.txt'],
    }
```

**Learn:**
- How to define tasks with `task_` functions
- Using `actions`, `file_dep`, and `targets`
- How doit tracks dependencies

### 2. Move to [02_example_project](02_example_project/)

A realistic data science workflow with separate Python scripts:

```
task_config → task_pull → task_process → task_chart
```

**Learn:**
- Project structure with `src/` and settings
- Pulling data from external APIs (FRED)
- Chaining scripts through file dependencies

### 3. Study [03_handling_notebooks](03_handling_notebooks/)

Full clone of `blank_project_simple` showing how to handle regular Jupyter notebooks:

**Learn:**
- Two-stage workflow: convert `.ipynb` to `.py` for change detection
- Execute notebooks only when code changes (not metadata)
- Generate HTML output from notebooks

### 4. Explore [04_jupytext_notebooks](04_jupytext_notebooks/)

Copy of `fred_charts` demonstrating the jupytext workflow with documentation:

**Learn:**
- Store notebooks as `.py` files (version control friendly)
- Jupytext percent format
- Sphinx/Jupyter Book documentation generation

## Quick Start

Install doit:

```bash
pip install doit
```

Navigate to any example directory and run:

```bash
# Run all tasks
doit

# List available tasks
doit list

# Run a specific task
doit <task_name>

# Clean generated files
doit clean

# Force re-run (forget previous state)
doit forget
```

## Key Concepts

### Task Definition

Tasks are Python functions that return a dictionary:

```python
def task_my_task():
    return {
        'actions': [...],      # What to do (functions or shell commands)
        'file_dep': [...],     # Files needed before running
        'targets': [...],      # Files produced by this task
        'clean': True,         # Enable 'doit clean' for this task
    }
```

### Dependency Types

1. **File dependencies** (`file_dep`): Task runs if these files change
2. **Task dependencies** (`task_dep`): Other tasks that must complete first
3. **Targets**: Files produced by the task

### How doit Tracks Changes

doit uses an SQLite database (`.doit.db`) to track:
- MD5 hashes of input files
- Whether targets exist
- When tasks last ran

If nothing has changed, tasks are skipped (up-to-date).

## Comparison: Notebooks Handling

| Aspect | 03_handling_notebooks | 04_jupytext_notebooks |
|--------|----------------------|----------------------|
| Storage | Regular `.ipynb` files | `.py` files (jupytext format) |
| Version Control | Harder (JSON diffs) | Easy (Python diffs) |
| Change Detection | Convert to `.py` first | Direct `.py` tracking |
| Tooling | Standard Jupyter | Requires jupytext |

## Resources

- [PyDoit Documentation](https://pydoit.org/contents.html)
- [Jupytext](https://jupytext.readthedocs.io/)
- [FRED API](https://fred.stlouisfed.org/)
- [Textbook: What is a Task Runner?](../../textbook/docs/Week3/what_is_a_task_runner.html)
