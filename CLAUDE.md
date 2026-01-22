# Claude Instructions

This repository contains in-class examples for **FINM 33200: Generative and Agentic AI for Finance**. Examples should be self-contained, runnable, and suitable for in-class demonstration using the course tech stack (Python, TypeScript, LangChain, etc.).

## References

**Textbook:** `/Users/jbejarano/GitRepositories/finm33200/finm33200_textbook` ([GitHub](https://github.com/finm-33200/finm33200_textbook))
Consult for topics, notation, and conventions when creating examples.

**Finance Projects:** `/Users/jbejarano/GitRepositories/ftsfr_repos`
Production-quality reference implementations. Search for data loading patterns (WRDS queries), pandas/numpy transforms, and financial calculations.

Available: `basis_tips_treas`, `basis_treas_sf`, `basis_treas_swap`, `cds_bond_basis`, `cds_returns`, `cip`, `commodities`, `corp_bond_returns`, `crsp_treasury`, `fed_yield_curve`, `foreign_exchange`, `he_kelly_manela`, `ken_french_data_library`, `nyu_call_report`, `options`, `sovereign_bonds`, `us_treasury_returns`, `wrds_bank_premium`, `wrds_crsp_compustat`

## Directory Structure

Organized by **topic** at the top level, with numbered subdirectories (`01_`, `02_`, ...) progressing from simple to complex:

```
ai_inclass_examples/
├── env_vars/
│   ├── 01_cli_basics/
│   ├── 02_dotenv_file/
│   ├── 03_decouple/
│   ├── 04_path_exploration/
│   └── 05_wrds_credentials/
├── project_paths/
│   └── install_approach/
└── pydoit/
    ├── 01_simplest/
    ├── 02_example_project/
    ├── 03_handling_notebooks/
    └── 04_jupytext_notebooks/
```

- Each example should be self-contained and runnable independently
- Typical example contains 1-3 scripts working together
- **Numbering convention:** When a subdirectory contains two or more example scripts, number them with `01_`, `02_`, etc. to make the pedagogical order clear (e.g., `01_json_mode.py`, `02_json_schema.py`)

## README Guidelines

**Topic-level README** (e.g., `basic_llm_api/README.md`): Overview, rationale for progression, table of subdirectories.

**Example-level README** (e.g., `01_gemini_hello/README.md`): What is learned, key concepts, setup steps, how to run, "Try it" exercises.

## Environment Variables

Copy `.env.example` to `.env` at repo root and fill in values. Examples load from there:

```python
from decouple import Config, RepositoryEnv
from pathlib import Path

repo_root = Path(__file__).resolve().parents[2]  # adjust based on nesting
config = Config(RepositoryEnv(repo_root / ".env"))
```
