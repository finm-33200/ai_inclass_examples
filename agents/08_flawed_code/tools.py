"""
tools.py

Tools for flawed_code - read, write, list, search, execute.
Simple implementations focused on clarity over production-readiness.
"""

import re
import subprocess
import tempfile
from pathlib import Path
from typing import Annotated

from langchain_core.tools import tool


# Working directory - set by flawed_code.py before running
WORK_DIR: Path = Path(".")


# =============================================================================
# FILE TOOLS
# =============================================================================

@tool
def read_file(path: Annotated[str, "Relative path to file"]) -> str:
    """Read the contents of a file."""
    try:
        full_path = WORK_DIR / path
        if not full_path.exists():
            return f"Error: File not found: {path}"
        return full_path.read_text()
    except Exception as e:
        return f"Error: {e}"


@tool
def write_file(
    path: Annotated[str, "Relative path to file"],
    content: Annotated[str, "Content to write"]
) -> str:
    """Write content to a file. Creates or overwrites."""
    try:
        full_path = WORK_DIR / path
        full_path.parent.mkdir(parents=True, exist_ok=True)
        full_path.write_text(content)
        return f"Wrote {len(content)} chars to {path}"
    except Exception as e:
        return f"Error: {e}"


@tool
def list_files(
    directory: Annotated[str, "Directory path, '.' for current"] = "."
) -> str:
    """List files and directories."""
    try:
        full_path = WORK_DIR / directory
        if not full_path.exists():
            return f"Error: Directory not found: {directory}"

        items = []
        for item in sorted(full_path.iterdir()):
            prefix = "[dir] " if item.is_dir() else "[file]"
            items.append(f"{prefix} {item.name}")
        return "\n".join(items) if items else "[Empty directory]"
    except Exception as e:
        return f"Error: {e}"


@tool
def search_files(
    pattern: Annotated[str, "Text to search for (case-insensitive)"],
    glob: Annotated[str, "File pattern like '*.py'"] = "**/*"
) -> str:
    """Search for text pattern in files."""
    try:
        results = []
        for file_path in WORK_DIR.glob(glob):
            if file_path.is_file():
                try:
                    content = file_path.read_text()
                    for i, line in enumerate(content.split("\n"), 1):
                        if re.search(pattern, line, re.IGNORECASE):
                            rel = file_path.relative_to(WORK_DIR)
                            results.append(f"{rel}:{i}: {line.strip()}")
                except:
                    pass

        if not results:
            return f"No matches for '{pattern}'"
        return "\n".join(results[:15])  # Limit results
    except Exception as e:
        return f"Error: {e}"


# =============================================================================
# CODE EXECUTION
# =============================================================================

@tool
def run_python(
    code: Annotated[str, "Python code to execute"]
) -> str:
    """Execute Python code and return output. Use print() for output."""
    try:
        # Write to temp file
        with tempfile.NamedTemporaryFile(
            mode='w', suffix='.py', delete=False, dir=WORK_DIR
        ) as f:
            f.write(code)
            temp_path = f.name

        # Run with timeout
        result = subprocess.run(
            ['python', temp_path],
            capture_output=True,
            text=True,
            timeout=10,
            cwd=WORK_DIR,
        )

        output = result.stdout
        if result.stderr:
            output += f"\n[stderr]: {result.stderr}"
        if result.returncode != 0:
            output += f"\n[exit: {result.returncode}]"

        return output.strip() if output.strip() else "[No output]"

    except subprocess.TimeoutExpired:
        return "[Timeout: 10 second limit exceeded]"
    except Exception as e:
        return f"[Error: {e}]"
    finally:
        Path(temp_path).unlink(missing_ok=True)
