# 07 - File Operations Agent

This example creates an agent with **file system tools**: read, write, list, and search. Combined with code execution (example 06), these form the core capabilities for a coding assistant.

## Why File Operations?

Coding assistants need to:
- Read existing code to understand it
- Modify code to fix bugs or add features
- Navigate project structure
- Search for patterns across files

## What You'll Learn

1. **File tools** - read_file, write_file, list_files, search_files
2. **Path handling** - Safe relative paths, directory constraints
3. **Agent-driven edits** - LLM reads, understands, then modifies
4. **Real code modification** - Actually changes files

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run

```bash
python file_agent.py
```

The script creates a `demo_project/` directory with sample files for the agent to work with.

## Example Output

```
Request: What files are in the project?
------------------------------------------------------------

  [list_files] directory='.'
  → [dir]  __pycache__/
    [file] README.md (150 bytes)
    [file] calculator.py (580 bytes)
    [file] sample_data.csv (220 bytes)

Response:
The project contains:
- calculator.py - Python module
- sample_data.csv - Data file
- README.md - Documentation

============================================================
Request: Find the bug in calculator.py and fix it.
------------------------------------------------------------

  [read_file] path='calculator.py'
  → """Simple calculator module..."""

Response:
I found a bug in the divide function - it doesn't check for division by zero.
Let me fix that.

  [write_file] path='calculator.py', content='...'
  → Successfully wrote 650 characters to calculator.py

Response:
Fixed! The divide function now raises a ValueError if b is zero.
```

## The Tools

### read_file
```python
@tool
def read_file(path: str) -> str:
    """Read file contents."""
    content = (WORK_DIR / path).read_text()
    return content
```

### write_file
```python
@tool
def write_file(path: str, content: str) -> str:
    """Write content to file."""
    (WORK_DIR / path).write_text(content)
    return f"Successfully wrote to {path}"
```

### list_files
```python
@tool
def list_files(directory: str = ".") -> str:
    """List directory contents."""
    items = []
    for item in (WORK_DIR / directory).iterdir():
        items.append(f"{'[dir]' if item.is_dir() else '[file]'} {item.name}")
    return "\n".join(items)
```

### search_files
```python
@tool
def search_files(pattern: str, file_pattern: str = "**/*") -> str:
    """Search for pattern across files."""
    results = []
    for file in WORK_DIR.glob(file_pattern):
        for i, line in enumerate(file.read_text().split("\n"), 1):
            if pattern.lower() in line.lower():
                results.append(f"{file.name}:{i}: {line}")
    return "\n".join(results)
```

## Safety Considerations

**Implemented:**
- Working directory constraint (WORK_DIR)
- Relative paths only
- Error handling

**For Production:**
- Path traversal prevention (`..` in paths)
- File type restrictions
- Size limits
- Backup before overwrite
- Permission checks

## Demo Project

The script creates a `demo_project/` with:
- `calculator.py` - Has a division by zero bug
- `sample_data.csv` - Sample financial data
- `README.md` - Project documentation

## Finance Application

File operations enable:
- Reading config files and data
- Modifying trading scripts
- Navigating large codebases
- Searching for specific patterns (e.g., "TODO", "BUG")

## Try It

1. **Ask the agent**: "Add a new function to calculate compound interest"
2. **Search**: "Find all functions in the project"
3. **Bug fix**: "The divide function needs better error handling"
4. **Documentation**: "Update the README to describe all functions"

## Next: FlawedCode

This example provides the file operation tools we need for "flawed_code" (example 08). Combined with code execution from example 06, we have all the building blocks!
