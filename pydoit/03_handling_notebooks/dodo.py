"""Run or update the project. This file uses the `doit` Python package. It works
like a Makefile, but is Python-based

"""

#######################################
## Configuration and Helpers for PyDoit
#######################################
## Make sure the src folder is in the path
import sys

sys.path.insert(1, "./src/")

import shutil
from os import environ, getcwd, path
from pathlib import Path
from settings import config

BASE_DIR = config("BASE_DIR")
DATA_DIR = config("DATA_DIR")
OUTPUT_DIR = config("OUTPUT_DIR")

## Helpers for handling Jupyter Notebook tasks
# fmt: off
## Helper functions for automatic execution of Jupyter notebooks
environ["PYDEVD_DISABLE_FILE_VALIDATION"] = "1"
def jupyter_execute_notebook(notebook):
    return f"jupyter nbconvert --execute --to notebook --ClearMetadataPreprocessor.enabled=True --log-level WARN --inplace ./src/{notebook}.ipynb"
def jupyter_to_html(notebook, output_dir=OUTPUT_DIR):
    return f"jupyter nbconvert --to html --log-level WARN --output-dir={output_dir} ./src/{notebook}.ipynb"
def jupyter_to_md(notebook, output_dir=OUTPUT_DIR):
    """Requires jupytext"""
    return f"jupytext --to markdown --log-level WARN --output-dir={output_dir} ./src/{notebook}.ipynb"
def jupyter_to_python(notebook, build_dir):
    """Convert a notebook to a python script"""
    return f"jupyter nbconvert --log-level WARN --to python ./src/{notebook}.ipynb --output _{notebook}.py --output-dir {build_dir}"
def jupyter_clear_output(notebook):
    return f"jupyter nbconvert --log-level WARN --ClearOutputPreprocessor.enabled=True --ClearMetadataPreprocessor.enabled=True --inplace ./src/{notebook}.ipynb"
# fmt: on


def copy_file(origin_path, destination_path, mkdir=True):
    """Create a Python action for copying a file."""

    def _copy_file():
        origin = Path(origin_path)
        dest = Path(destination_path)
        if mkdir:
            dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(origin, dest)

    return _copy_file


##################################
## Begin rest of PyDoit tasks here
##################################


def task_config():
    """Create empty directories for data and output if they don't exist"""
    return {
        "actions": ["ipython ./src/settings.py"],
        "targets": [DATA_DIR, OUTPUT_DIR],
        "file_dep": ["./src/settings.py"],
        "clean": [],
    }


def task_pull_public_repo_data():
    """Pull public data from FRED and OFR API"""

    return {
        "actions": [
            "ipython ./src/settings.py",
            "ipython ./src/pull_fred.py",
            "ipython ./src/pull_ofr_api_data.py",
        ],
        "targets": [
            DATA_DIR / "fred.parquet",
            DATA_DIR / "ofr_public_repo_data.parquet",
        ],
        "file_dep": [
            "./src/settings.py",
            "./src/pull_fred.py",
            "./src/pull_ofr_api_data.py",
        ],
        "clean": [],  # Don't clean these files by default. The ideas
        # is that a data pull might be expensive, so we don't want to
        # redo it unless we really mean it. So, when you run
        # doit clean, all other tasks will have their targets
        # cleaned and will thus be rerun the next time you call doit.
        # But this one wont.
        # Use doit forget --all to redo all tasks. Use doit clean
        # to clean and forget the cheaper tasks.
    }


def task_pull_ken_french_data():
    """Pull public data from FRED and OFR API"""

    return {
        "actions": [
            "ipython ./src/settings.py",
            "ipython ./src/pull_ken_french_data.py",
        ],
        "targets": [
            DATA_DIR / "25_Portfolios_OP_INV_5x5_daily.parquet",
        ],
        "file_dep": [
            "./src/settings.py",
            "./src/pull_ken_french_data.py",
        ],
        "clean": [],  # Don't clean these files by default. The ideas
        # is that a data pull might be expensive, so we don't want to
        # redo it unless we really mean it. So, when you run
        # doit clean, all other tasks will have their targets
        # cleaned and will thus be rerun the next time you call doit.
        # But this one wont.
        # Use doit forget --all to redo all tasks. Use doit clean
        # to clean and forget the cheaper tasks.
    }



##############################$
## Demo: Other misc. data pulls
##############################$
# def task_pull_other():
#     """ """
#     file_dep = [
#         "./src/pull_bloomberg.py",
#         "./src/pull_CRSP_Compustat.py",
#         "./src/pull_CRSP_stock.py",
#         "./src/pull_fed_yield_curve.py",
#         ]
#     file_output = [
#         "bloomberg.parquet",
#         "CRSP_Compustat.parquet",
#         "CRSP_stock.parquet",
#         "fed_yield_curve.parquet",
#         ]
#     targets = [DATA_DIR / file for file in file_output]

#     return {
#         "actions": [
#             "ipython ./src/pull_bloomberg.py",
#             "ipython ./src/pull_CRSP_Compustat.py",
#             "ipython ./src/pull_CRSP_stock.py",
#             "ipython ./src/pull_fed_yield_curve.py",
#         ],
#         "targets": targets,
#         "file_dep": file_dep,
#         "clean": [],  # Don't clean these files by default.
#     }


def task_summary_stats():
    """Generate table of summary statistics"""

    return {
        "actions": [
            "ipython ./src/example_table.py",
            "ipython ./src/pandas_to_latex_demo.py",
        ],
        "targets": [
            OUTPUT_DIR / "example_table.tex",
            OUTPUT_DIR / "pandas_to_latex_simple_table1.tex",
        ],
        "file_dep": [
            "./src/example_table.py",
            "./src/pandas_to_latex_demo.py",
        ],
        "clean": True,
    }


def task_example_plot():
    """Example plots"""

    return {
        "actions": [
            # "date 1>&2",
            # "time ipython ./src/example_plot.py",
            "ipython ./src/example_plot.py",
        ],
        "targets": [
            OUTPUT_DIR / "example_plot.png",
        ],
        "file_dep": [
            "./src/example_plot.py",
            "./src/pull_fred.py",
        ],
        "clean": True,
    }


def task_chart_repo_rates():
    """Example charts for Chart Book"""

    return {
        "actions": [
            # "date 1>&2",
            # "time ipython ./src/chart_relative_repo_rates.py",
            "ipython ./src/chart_relative_repo_rates.py",
        ],
        "targets": [
            DATA_DIR / "repo_public.parquet",
            DATA_DIR / "repo_public_relative_fed.parquet",
            OUTPUT_DIR / "repo_rates.html",
            OUTPUT_DIR / "repo_rates_normalized.html",
            OUTPUT_DIR / "repo_rates_normalized_w_balance_sheet.html",
        ],
        "file_dep": [
            "./src/pull_fred.py",
            "./src/chart_relative_repo_rates.py",
        ],
        "clean": True,
    }


notebook_tasks = {
    "01_example_notebook_interactive.ipynb": {
        "file_dep": [],
        "targets": [
            Path("./docs") / "01_example_notebook_interactive.html",
        ],
    },
    "02_example_with_dependencies.ipynb": {
        "file_dep": ["./src/pull_fred.py"],
        "targets": [
            Path(OUTPUT_DIR) / "GDP_graph.png",
            Path("./docs") / "02_example_with_dependencies.html",
        ],
    },
    "03_public_repo_summary_charts.ipynb": {
        "file_dep": [
            "./src/pull_fred.py",
            "./src/pull_ofr_api_data.py",
            "./src/pull_public_repo_data.py",
        ],
        "targets": [
            OUTPUT_DIR / "repo_rate_spikes_and_relative_reserves_levels.png",
            OUTPUT_DIR / "rates_relative_to_midpoint.png",
            Path("./docs") / "03_public_repo_summary_charts.html",
        ],
    },
    "04_ken_french_data.ipynb": {
        "file_dep": [],
        "targets": [
            Path("./docs") / "04_ken_french_data.html",
        ],
    },
    "index.ipynb": {
        "file_dep": [],
        "targets": [
            Path("./docs") / "index.html",
        ],
    },
}


def task_convert_notebooks_to_scripts():
    """Convert notebooks to script form to detect changes to source code rather
    than to the notebook's metadata.
    """
    build_dir = Path(OUTPUT_DIR)

    for notebook in notebook_tasks.keys():
        notebook_name = notebook.split(".")[0]
        yield {
            "name": notebook,
            "actions": [
                jupyter_clear_output(notebook_name),
                jupyter_to_python(notebook_name, build_dir),
            ],
            "file_dep": [Path("./src") / notebook],
            "targets": [OUTPUT_DIR / f"_{notebook_name}.py"],
            "clean": True,
            "verbosity": 0,
        }


# fmt: off
def task_run_notebooks():
    """Preps the notebooks for presentation format.
    Execute notebooks if the script version of it has been changed.
    """
    for notebook in notebook_tasks.keys():
        notebook_name = notebook.split(".")[0]
        yield {
            "name": notebook,
            "actions": [
                """python -c "import sys; from datetime import datetime; print(f'Start """ + notebook + """: {datetime.now()}', file=sys.stderr)" """,
                jupyter_execute_notebook(notebook_name),
                jupyter_to_html(notebook_name),
                copy_file(
                    Path("./src") / f"{notebook_name}.ipynb",
                    OUTPUT_DIR / f"{notebook_name}.ipynb",
                    mkdir=True,
                ),
                copy_file(
                    OUTPUT_DIR / f"{notebook_name}.html",
                    Path("./docs") / f"{notebook_name}.html",
                    mkdir=True,
                ),
                jupyter_clear_output(notebook_name),
                # jupyter_to_python(notebook_name, build_dir),
                """python -c "import sys; from datetime import datetime; print(f'End """ + notebook + """: {datetime.now()}', file=sys.stderr)" """,
            ],
            "file_dep": [
                OUTPUT_DIR / f"_{notebook_name}.py",
                *notebook_tasks[notebook]["file_dep"],
            ],
            "targets": [
                OUTPUT_DIR / f"{notebook_name}.html",
                OUTPUT_DIR / f"{notebook_name}.ipynb",
                *notebook_tasks[notebook]["targets"],
            ],
            "clean": True,
        }
# fmt: on
