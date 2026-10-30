"""Run or update the project. This file uses the `doit` Python package. It works
like a Makefile, but is Python-based.

Project: FINM 33200 - Generative and Agentic AI for Finance (in-class examples)
"""

import shutil
from os import environ
from pathlib import Path

# Output directories
OUTPUT_DIR = Path("_output")
HTML_OUTPUT_DIR = OUTPUT_DIR / "notebooks"

# Suppress debugger file validation warnings during notebook execution
environ["PYDEVD_DISABLE_FILE_VALIDATION"] = "1"

DOIT_CONFIG = {
    "backend": "sqlite3",
    "dep_file": "./.doit-db.sqlite",
}


# fmt: off
## Helper functions for automatic execution of Jupyter notebooks
def jupyter_execute_notebook(notebook_path):
    return f"jupyter nbconvert --execute --to notebook --ExecutePreprocessor.timeout=180 --ClearMetadataPreprocessor.enabled=True --inplace {notebook_path}"

def jupyter_to_html(notebook_path, output_dir=HTML_OUTPUT_DIR):
    return f"jupyter nbconvert --to html --output-dir={output_dir} {notebook_path}"

def jupyter_clear_output(notebook_path):
    return f"jupyter nbconvert --ClearOutputPreprocessor.enabled=True --ClearMetadataPreprocessor.enabled=True --inplace {notebook_path}"
# fmt: on


def move_file(from_path, to_path):
    """Create a Python action that moves a file into a directory."""

    def _move():
        src = Path(from_path)
        dst = Path(to_path)
        dst.mkdir(parents=True, exist_ok=True)
        shutil.move(str(src), str(dst / src.name))

    return _move


##################################
## Begin rest of PyDoit tasks here
##################################


def task_setup():
    """Create output directories if they don't exist."""

    def _create_dirs():
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        HTML_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    return {
        "actions": [_create_dirs],
        "targets": [OUTPUT_DIR, HTML_OUTPUT_DIR],
        "uptodate": [True],
        "clean": [],
    }


# ---------------------------------------------------------------------------
# Notebook task definitions
# Keys are logical names; values contain the .py source path and dependencies.
# ---------------------------------------------------------------------------
notebook_tasks = {
    "openai_api_overview": {
        "pyfile": "./discussion_notebooks/openai_api_overview.ipynb.py",
        "file_dep": [],
        "targets": [],
    },
    "benchmarking_llms": {
        "pyfile": "./discussion_notebooks/benchmarking_llms.ipynb.py",
        "file_dep": [],
        "targets": [],
    },
}


# fmt: off
def task_run_notebooks():
    """Convert, execute, and export Jupytext notebooks.

    For each notebook the pipeline is:
      1. jupytext: convert .ipynb.py -> .ipynb
      2. nbconvert --execute: run all cells in-place
      3. nbconvert --to html: export to _output/notebooks/
      4. move the .ipynb into _output/
    """
    for name, info in notebook_tasks.items():
        pyfile_path = Path(info["pyfile"])
        # openai_api_overview.ipynb.py -> openai_api_overview.ipynb
        notebook_path = pyfile_path.with_name(pyfile_path.stem)
        yield {
            "name": name,
            "actions": [
                f"jupytext --to notebook --output {notebook_path} {pyfile_path}",
                jupyter_execute_notebook(notebook_path),
                jupyter_to_html(notebook_path),
                move_file(notebook_path, OUTPUT_DIR),
            ],
            "file_dep": [
                pyfile_path,
                *info["file_dep"],
            ],
            "targets": [
                HTML_OUTPUT_DIR / f"{pyfile_path.stem.replace('.ipynb', '')}.html",
                OUTPUT_DIR / pyfile_path.stem,
                *info["targets"],
            ],
            "task_dep": ["setup"],
            "clean": True,
            "verbosity": 2,
        }
# fmt: on


sphinx_targets = [
    "./docs/index.html",
]


def task_build_chartbook_site():
    """Compile Sphinx docs via ChartBook."""
    notebook_scripts = [
        Path(notebook_tasks[name]["pyfile"]) for name in notebook_tasks
    ]
    file_dep = [
        "./README.md",
        "./chartbook.toml",
        *notebook_scripts,
    ]

    return {
        "actions": [
            "chartbook docs generate -f",
        ],
        "targets": sphinx_targets,
        "file_dep": file_dep,
        "task_dep": ["run_notebooks"],
        "clean": True,
    }
