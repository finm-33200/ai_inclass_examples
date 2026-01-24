"""
06_code_execution/code_agent.py

Agent that can write and execute Python code.
Uses subprocess for basic sandboxing with timeout.

Key concept: Giving LLMs the ability to run code.
"""

import json
import subprocess
import tempfile
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


# =============================================================================
# CODE EXECUTION TOOL
# =============================================================================

@tool
def run_python(
    code: Annotated[str, "Python code to execute. Use print() to output results."]
) -> str:
    """
    Execute Python code and return the output.

    The code runs in a subprocess with:
    - 10 second timeout
    - Captured stdout and stderr
    - Common libraries available (numpy, pandas if installed)

    Use print() statements to output results.
    """
    print(f"  [Executing Python code...]")

    # Create a temporary file for the code
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(code)
        temp_path = f.name

    try:
        # Run with timeout and capture output
        result = subprocess.run(
            ['python', temp_path],
            capture_output=True,
            text=True,
            timeout=10,  # 10 second timeout
        )

        output = ""
        if result.stdout:
            output += result.stdout
        if result.stderr:
            output += f"\n[stderr]: {result.stderr}"

        if result.returncode != 0:
            output += f"\n[exit code: {result.returncode}]"

        return output.strip() if output.strip() else "[No output]"

    except subprocess.TimeoutExpired:
        return "[Error: Code execution timed out (10 second limit)]"
    except Exception as e:
        return f"[Error: {str(e)}]"
    finally:
        # Clean up temp file
        Path(temp_path).unlink(missing_ok=True)


@tool
def calculate(
    expression: Annotated[str, "Simple math expression (e.g., '100 * 1.05')"]
) -> str:
    """Evaluate a simple math expression. For complex calculations, use run_python instead."""
    allowed = set("0123456789+-*/.(). ")
    if not all(c in allowed for c in expression):
        return "Error: Use run_python for complex calculations"
    try:
        return str(eval(expression))
    except Exception as e:
        return f"Error: {e}"


tools = [run_python, calculate]


# =============================================================================
# AGENT
# =============================================================================

SYSTEM_MESSAGE = SystemMessage(
    content="""You are a helpful data analysis assistant with Python execution capability.

When asked to analyze data or perform calculations:
1. Write Python code using print() to output results
2. Use run_python tool to execute it
3. Explain the results

Available libraries: math, statistics, collections, itertools, json
For complex analysis, write proper Python code rather than simple expressions.
"""
)


def create_agent():
    return create_react_agent(llm, tools, prompt=SYSTEM_MESSAGE)


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("Code Execution Agent")
    print("=" * 60)

    agent = create_agent()

    # Test questions that require code execution
    questions = [
        "Calculate the compound annual growth rate (CAGR) if an investment grows from $10,000 to $15,000 over 3 years.",
        "Generate the first 10 Fibonacci numbers.",
        "Calculate the standard deviation of these returns: [0.05, -0.02, 0.08, -0.01, 0.03]",
    ]

    for question in questions:
        print(f"\n{'='*60}")
        print(f"Question: {question}")
        print("-" * 60)

        for event in agent.stream({"messages": [("user", question)]}):
            for key, value in event.items():
                if key == "agent":
                    msg = value["messages"][-1]
                    if hasattr(msg, "tool_calls") and msg.tool_calls:
                        for tc in msg.tool_calls:
                            if tc["name"] == "run_python":
                                print(f"\n  [Code to execute]:")
                                code = tc["args"].get("code", "")
                                for line in code.split("\n"):
                                    print(f"    {line}")
                            else:
                                print(f"  [Tool] {tc['name']}({tc['args']})")
                    elif hasattr(msg, "content") and msg.content:
                        print(f"\nAnswer:\n{msg.content}")
                elif key == "tools":
                    msg = value["messages"][-1]
                    print(f"\n  [Output]: {msg.content}")


if __name__ == "__main__":
    main()
