"""
07_file_operations/file_agent.py

Agent with file system tools: read, write, list, search.
These are the core capabilities for a coding assistant.
"""

import os
import re
from pathlib import Path
from typing import Annotated

from decouple import Config, RepositoryEnv
from langchain_core.tools import tool
from langchain_core.messages import SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent


# =============================================================================
# SETUP
# =============================================================================

repo_root = Path(__file__).resolve().parents[2]
config = Config(RepositoryEnv(repo_root / ".env"))

llm = ChatOpenAI(
    model="openai/gpt-4o-mini",
    base_url="https://openrouter.ai/api/v1",
    api_key=config("OPENROUTER_API_KEY"),
)

# Working directory for file operations (for safety, limit to demo_project)
WORK_DIR = Path(__file__).parent / "demo_project"


# =============================================================================
# FILE OPERATION TOOLS
# =============================================================================

@tool
def read_file(path: Annotated[str, "Relative path to file"]) -> str:
    """Read the contents of a file. Returns the file content as a string."""
    try:
        full_path = WORK_DIR / path
        if not full_path.exists():
            return f"Error: File not found: {path}"
        if not full_path.is_file():
            return f"Error: Not a file: {path}"

        content = full_path.read_text()
        return content
    except Exception as e:
        return f"Error reading file: {e}"


@tool
def write_file(
    path: Annotated[str, "Relative path to file"],
    content: Annotated[str, "Content to write to the file"]
) -> str:
    """Write content to a file. Creates the file if it doesn't exist, overwrites if it does."""
    try:
        full_path = WORK_DIR / path
        full_path.parent.mkdir(parents=True, exist_ok=True)
        full_path.write_text(content)
        return f"Successfully wrote {len(content)} characters to {path}"
    except Exception as e:
        return f"Error writing file: {e}"


@tool
def list_files(
    directory: Annotated[str, "Relative path to directory (use '.' for current)"] = "."
) -> str:
    """List files and directories in a given directory."""
    try:
        full_path = WORK_DIR / directory
        if not full_path.exists():
            return f"Error: Directory not found: {directory}"
        if not full_path.is_dir():
            return f"Error: Not a directory: {directory}"

        items = []
        for item in sorted(full_path.iterdir()):
            if item.is_dir():
                items.append(f"[dir]  {item.name}/")
            else:
                size = item.stat().st_size
                items.append(f"[file] {item.name} ({size} bytes)")

        return "\n".join(items) if items else "[Empty directory]"
    except Exception as e:
        return f"Error listing directory: {e}"


@tool
def search_files(
    pattern: Annotated[str, "Text pattern to search for (case-insensitive)"],
    file_pattern: Annotated[str, "Glob pattern for files to search (e.g., '*.py')"] = "**/*"
) -> str:
    """Search for a text pattern across files. Returns matching lines with file and line number."""
    try:
        results = []
        for file_path in WORK_DIR.glob(file_pattern):
            if file_path.is_file():
                try:
                    content = file_path.read_text()
                    for i, line in enumerate(content.split("\n"), 1):
                        if re.search(pattern, line, re.IGNORECASE):
                            rel_path = file_path.relative_to(WORK_DIR)
                            results.append(f"{rel_path}:{i}: {line.strip()}")
                except:
                    pass  # Skip binary files

        if not results:
            return f"No matches found for '{pattern}'"
        return "\n".join(results[:20])  # Limit to 20 results
    except Exception as e:
        return f"Error searching: {e}"


tools = [read_file, write_file, list_files, search_files]


# =============================================================================
# AGENT
# =============================================================================

SYSTEM_MESSAGE = SystemMessage(
    content="""You are a helpful coding assistant with file system access.

You can:
- Read files to understand code
- Write/modify files to make changes
- List directory contents
- Search for patterns across files

When modifying code:
1. First read the file to understand the current state
2. Make targeted changes
3. Confirm the change was successful

Always show what you're doing and explain your changes."""
)


def create_agent():
    return create_react_agent(llm, tools, prompt=SYSTEM_MESSAGE)


# =============================================================================
# SETUP DEMO PROJECT
# =============================================================================

def setup_demo_project():
    """Create a demo project for the agent to work with."""
    WORK_DIR.mkdir(exist_ok=True)

    # Create a simple Python file with a bug
    (WORK_DIR / "calculator.py").write_text('''"""Simple calculator module."""

def add(a, b):
    """Add two numbers."""
    return a + b

def subtract(a, b):
    """Subtract b from a."""
    return a - b

def multiply(a, b):
    """Multiply two numbers."""
    return a * b

def divide(a, b):
    """Divide a by b."""
    # BUG: No check for division by zero!
    return a / b

def calculate_returns(prices):
    """Calculate percentage returns from a list of prices."""
    returns = []
    for i in range(1, len(prices)):
        ret = (prices[i] - prices[i-1]) / prices[i-1]
        returns.append(ret)
    return returns
''')

    # Create a data file
    (WORK_DIR / "sample_data.csv").write_text('''date,ticker,price
2024-01-01,AAPL,185.50
2024-01-02,AAPL,186.25
2024-01-03,AAPL,184.90
2024-01-01,MSFT,375.00
2024-01-02,MSFT,378.50
2024-01-03,MSFT,380.25
''')

    # Create a README
    (WORK_DIR / "README.md").write_text('''# Demo Project

A simple project for demonstrating file operations.

## Files

- `calculator.py` - Basic math functions (has a bug!)
- `sample_data.csv` - Sample stock price data
''')

    print(f"Demo project created at: {WORK_DIR}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("File Operations Agent")
    print("=" * 60)

    # Setup demo project
    setup_demo_project()
    print()

    agent = create_agent()

    # Test questions
    questions = [
        "What files are in the project?",
        "Read the calculator.py file and describe what it does.",
        "Find the bug in calculator.py and fix it.",
    ]

    for question in questions:
        print(f"\n{'='*60}")
        print(f"Request: {question}")
        print("-" * 60)

        for event in agent.stream({"messages": [("user", question)]}):
            for key, value in event.items():
                if key == "agent":
                    msg = value["messages"][-1]
                    if hasattr(msg, "tool_calls") and msg.tool_calls:
                        for tc in msg.tool_calls:
                            args_str = ", ".join(f"{k}={repr(v)[:50]}" for k, v in tc["args"].items())
                            print(f"\n  [{tc['name']}] {args_str}")
                    elif hasattr(msg, "content") and msg.content:
                        print(f"\nResponse:\n{msg.content}")
                elif key == "tools":
                    msg = value["messages"][-1]
                    # Show truncated output
                    content = msg.content
                    if len(content) > 200:
                        content = content[:200] + "..."
                    print(f"  → {content}")


if __name__ == "__main__":
    main()
