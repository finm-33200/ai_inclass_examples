#!/usr/bin/env python3
"""
flawed_code.py

A minimal Claude Code clone. Intentionally simple and uses cheap models.
"Flawed" because it uses inexpensive models that make mistakes.

Usage:
    python flawed_code.py [directory]

If no directory is given, uses demo_project/
"""

import sys
from pathlib import Path

from decouple import Config, RepositoryEnv
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent

from tools import read_file, write_file, list_files, search_files, run_python


# =============================================================================
# SETUP
# =============================================================================

repo_root = Path(__file__).resolve().parents[2]
config = Config(RepositoryEnv(repo_root / ".env"))

# The "flawed" part: using cheap models
# Students can swap these to see quality differences
MODEL = "openai/gpt-4o-mini"  # $0.15/1M - reliable but not the best
# MODEL = "meta-llama/llama-3.1-8b-instruct"  # $0.02/1M - often flawed!
# MODEL = "google/gemini-2.0-flash-exp:free"  # Free - variable quality

llm = ChatOpenAI(
    model=MODEL,
    base_url="https://openrouter.ai/api/v1",
    api_key=config("OPENROUTER_API_KEY"),
)


# =============================================================================
# AGENT
# =============================================================================

SYSTEM_PROMPT = """You are FlawedCode, a simple coding assistant.

You have access to these tools:
- read_file: Read file contents
- write_file: Write/create files
- list_files: List directory contents
- search_files: Search for patterns
- run_python: Execute Python code

When helping with code:
1. First understand the codebase (list_files, read_file)
2. Make focused changes (write_file)
3. Test if needed (run_python)

Be concise but helpful. You may make mistakes - you're running on a cheap model!
When you make changes, explain what you did.

Working directory: {work_dir}
"""


def create_flawed_agent(work_dir: Path):
    """Create the flawed code agent."""
    # Configure tools with the working directory
    import tools
    tools.WORK_DIR = work_dir

    all_tools = [read_file, write_file, list_files, search_files, run_python]

    prompt = SystemMessage(content=SYSTEM_PROMPT.format(work_dir=work_dir))

    return create_react_agent(llm, all_tools, prompt=prompt)


# =============================================================================
# REPL
# =============================================================================

def run_repl(agent, work_dir: Path):
    """Run the interactive REPL."""
    print(f"flawed_code v0.1 - A minimal Claude Code clone")
    print(f"Using model: {MODEL}")
    print(f"Working directory: {work_dir}")
    print(f"Commands: 'quit' to exit, 'clear' to reset conversation")
    print("-" * 50)

    messages = []

    while True:
        try:
            user_input = input("\nflawed_code> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not user_input:
            continue

        if user_input.lower() == "quit":
            print("Goodbye!")
            break

        if user_input.lower() == "clear":
            messages = []
            print("[Conversation cleared]")
            continue

        # Add user message
        messages.append(("user", user_input))

        # Stream the response
        print()
        try:
            for event in agent.stream({"messages": messages}):
                for key, value in event.items():
                    if key == "agent":
                        msg = value["messages"][-1]
                        if hasattr(msg, "tool_calls") and msg.tool_calls:
                            for tc in msg.tool_calls:
                                args_preview = str(tc["args"])[:60]
                                print(f"  [{tc['name']}] {args_preview}...")
                        elif hasattr(msg, "content") and msg.content:
                            print(msg.content)
                    elif key == "tools":
                        msg = value["messages"][-1]
                        content = msg.content[:100] if len(msg.content) > 100 else msg.content
                        print(f"    → {content}{'...' if len(msg.content) > 100 else ''}")

            # Add assistant response to conversation
            # (simplified - just track the last response)
            messages.append(("assistant", "[response]"))

        except Exception as e:
            print(f"[Error: {e}]")


# =============================================================================
# MAIN
# =============================================================================

def main():
    # Determine working directory
    if len(sys.argv) > 1:
        work_dir = Path(sys.argv[1]).resolve()
    else:
        work_dir = Path(__file__).parent / "demo_project"

    # Ensure directory exists
    if not work_dir.exists():
        print(f"Creating directory: {work_dir}")
        work_dir.mkdir(parents=True)

    # Create demo files if demo_project is empty
    if work_dir.name == "demo_project" and not any(work_dir.iterdir()):
        setup_demo_project(work_dir)

    # Create agent and run REPL
    agent = create_flawed_agent(work_dir)
    run_repl(agent, work_dir)


def setup_demo_project(work_dir: Path):
    """Create demo files for testing."""
    (work_dir / "analyze.py").write_text('''"""Stock price analyzer."""

def calculate_returns(prices):
    """Calculate daily returns from price list."""
    returns = []
    for i in range(1, len(prices)):
        daily_return = (prices[i] - prices[i-1]) / prices[i-1]
        returns.append(daily_return)
    return returns

def calculate_volatility(returns):
    """Calculate volatility (standard deviation of returns)."""
    # BUG: This doesn't handle empty lists!
    mean = sum(returns) / len(returns)
    variance = sum((r - mean) ** 2 for r in returns) / len(returns)
    return variance ** 0.5

def main():
    # Sample AAPL prices
    prices = [150.0, 152.5, 151.0, 155.0, 153.5]
    returns = calculate_returns(prices)
    vol = calculate_volatility(returns)
    print(f"Returns: {returns}")
    print(f"Volatility: {vol:.4f}")

if __name__ == "__main__":
    main()
''')

    (work_dir / "data.csv").write_text('''date,ticker,close
2024-01-01,AAPL,185.50
2024-01-02,AAPL,187.25
2024-01-03,AAPL,186.00
2024-01-04,AAPL,189.50
2024-01-05,AAPL,191.00
''')

    (work_dir / "README.md").write_text('''# Demo Project

Simple stock analysis tools.

## Files
- analyze.py - Stock price analysis functions
- data.csv - Sample price data

## Known Issues
- calculate_volatility has a bug with empty lists
''')

    print("Demo project files created.")


if __name__ == "__main__":
    main()
