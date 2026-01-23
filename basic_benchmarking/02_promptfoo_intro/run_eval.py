"""
Helper script to run promptfoo with environment variables from repo root .env
"""

import os
import subprocess
from pathlib import Path

from decouple import Config, RepositoryEnv


def main():
    # Load env vars from repo root
    repo_root = Path(__file__).resolve().parents[2]
    config = Config(RepositoryEnv(repo_root / ".env"))

    # Set the API key in environment for promptfoo
    env = os.environ.copy()
    env["OPENAI_API_KEY"] = config("OPENAI_API_KEY")

    # Run promptfoo
    # Note: exit code 100 means some tests failed (not an error)
    print("Running promptfoo eval with OPENAI_API_KEY from .env...\n")
    result = subprocess.run(["npx", "promptfoo@latest", "eval"], env=env)

    if result.returncode == 0:
        print("\nAll tests passed!")
    elif result.returncode == 100:
        print("\nSome tests failed (this is expected when comparing model quality)")
    else:
        print(f"\nError running promptfoo (exit code {result.returncode})")


if __name__ == "__main__":
    main()
