# 08 - FlawedCode

A **minimal Claude Code clone** in ~200 lines. Intentionally uses cheap models.

**Why "Flawed"?** It uses inexpensive models (~$0.15/1M tokens) that make mistakes. The flaws are the point - students learn both agent capabilities AND limitations.

## What It Does

FlawedCode is a REPL-based coding assistant with:
- **read_file** - Read file contents
- **write_file** - Create/modify files
- **list_files** - Navigate directories
- **search_files** - Find patterns in code
- **run_python** - Execute Python code

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run

```bash
# Use demo_project (default)
python flawed_code.py

# Use your own directory
python flawed_code.py /path/to/your/project
```

## Example Session

```
flawed_code v0.1 - A minimal Claude Code clone
Using model: openai/gpt-4o-mini
Working directory: demo_project
Commands: 'quit' to exit, 'clear' to reset conversation
--------------------------------------------------

flawed_code> what files are here?
  [list_files] {}...
    → [file] README.md
       [file] analyze.py
       [file] data.csv
There are 3 files: README.md, analyze.py, and data.csv.

flawed_code> read analyze.py and describe it
  [read_file] {'path': 'analyze.py'}...
    → """Stock price analyzer."""...
The analyze.py file contains stock analysis functions:
- calculate_returns: Computes daily returns from prices
- calculate_volatility: Computes standard deviation (has a bug!)
- main: Demo with sample AAPL prices

flawed_code> fix the bug in calculate_volatility
  [read_file] {'path': 'analyze.py'}...
  [write_file] {'path': 'analyze.py', 'content': '...'}...
    → Wrote 650 chars to analyze.py
Fixed! Added a check for empty lists that returns 0.0.

flawed_code> run it
  [run_python] {'code': 'exec(open("analyze.py").read())'}...
    → Returns: [0.0167, -0.0098, 0.0265, -0.0097]
       Volatility: 0.0150
The code runs successfully now.

flawed_code> quit
Goodbye!
```

## Architecture

```
08_flawed_code/
├── flawed_code.py    # Main REPL (~100 lines)
├── tools.py          # Tool definitions (~100 lines)
├── requirements.txt
├── README.md
└── demo_project/     # Sample files
    ├── analyze.py
    ├── data.csv
    └── README.md
```

**Total: ~200 lines** of actual code.

## The Code

### flawed_code.py
```python
# Setup LLM with cheap model via OpenRouter
llm = ChatOpenAI(
    model="openai/gpt-4o-mini",  # Cheap but capable
    base_url="https://openrouter.ai/api/v1",
    api_key=config("OPENROUTER_API_KEY"),
)

# Create agent with all tools
tools = [read_file, write_file, list_files, search_files, run_python]
agent = create_react_agent(llm, tools, prompt=SYSTEM_PROMPT)

# REPL loop
while True:
    user_input = input("flawed_code> ")
    for event in agent.stream({"messages": messages}):
        # Print tool calls and responses
        ...
```

### tools.py
```python
@tool
def read_file(path: str) -> str:
    return (WORK_DIR / path).read_text()

@tool
def write_file(path: str, content: str) -> str:
    (WORK_DIR / path).write_text(content)
    return f"Wrote {len(content)} chars"

@tool
def run_python(code: str) -> str:
    result = subprocess.run(['python', '-c', code], capture_output=True)
    return result.stdout
```

## What's NOT Included (vs Claude Code)

| Feature | Claude Code | FlawedCode |
|---------|-------------|-------------|
| Multi-file edits | Yes | No |
| Git integration | Yes | No |
| MCP servers | Yes | No |
| Planning mode | Yes | No |
| Image support | Yes | No |
| Permission system | Yes | No |
| Streaming UI | Fancy | Basic |

This simplicity is intentional - it's for learning, not production.

## Models to Try

Edit `flawed_code.py` to change models:

```python
# Cheap and capable
MODEL = "openai/gpt-4o-mini"  # $0.15/1M - default

# Even cheaper (more flaws!)
MODEL = "meta-llama/llama-3.1-8b-instruct"  # $0.02/1M

# Free tier
MODEL = "google/gemini-2.0-flash-exp:free"  # $0

# Better quality
MODEL = "anthropic/claude-3-haiku"  # $0.25/1M
```

## Finance Applications

Use FlawedCode to:
- Navigate quant codebases
- Fix bugs in analysis scripts
- Write data processing functions
- Explore unfamiliar code

## Try It

1. **Fix a bug**: "the calculate_volatility function crashes on empty lists"
2. **Add a feature**: "add a function to calculate Sharpe ratio"
3. **Analyze code**: "explain what calculate_returns does"
4. **Run code**: "calculate returns for [100, 102, 101, 105]"

## Learning Points

1. **Agents are simple**: Just a loop with tools
2. **LLMs make mistakes**: Especially cheap ones
3. **Tools are the interface**: LLM reasons, tools act
4. **Context is limited**: Memory is just message history
5. **Framework vs scratch**: LangGraph handles the plumbing

## The Journey

You've now built up to a working coding assistant:
1. **01_react_loop** - The core agent pattern
2. **02_agent_with_memory** - Conversation context
3. **03_multi_tool_agent** - Multiple tool selection
4. **04_langgraph_intro** - Graph-based workflows
5. **05_langgraph_agent** - Framework-based agents
6. **06_code_execution** - Running code safely
7. **07_file_operations** - File system tools
8. **08_flawed_code** - The complete assistant

From raw loops to a working Claude Code clone!
