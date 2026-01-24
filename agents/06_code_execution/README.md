# 06 - Code Execution Agent

This example creates an agent that can **write and execute Python code**. This is a key capability for data analysis and computation agents.

## Why Code Execution?

Some tasks can't be solved with pre-defined tools:
- Complex calculations
- Data transformations
- Dynamic analysis
- Generating visualizations

Solution: Let the LLM write code, then execute it.

## What You'll Learn

1. **Code execution tool** - Running Python in subprocess
2. **Sandboxing basics** - Timeout, output capture
3. **Security considerations** - What's safe, what's not
4. **Agent-generated code** - LLM writes the code

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run

```bash
python code_agent.py
```

## Example Output

```
Question: Calculate CAGR if investment grows from $10,000 to $15,000 over 3 years.
------------------------------------------------------------

  [Code to execute]:
    start_value = 10000
    end_value = 15000
    years = 3
    cagr = (end_value / start_value) ** (1 / years) - 1
    print(f"CAGR: {cagr:.2%}")

  [Output]: CAGR: 14.47%

Answer:
The Compound Annual Growth Rate (CAGR) is 14.47%. This means the investment
grew at an average rate of 14.47% per year over the 3-year period.
```

## The Execution Tool

```python
@tool
def run_python(code: str) -> str:
    """Execute Python code and return output."""

    # Write code to temp file
    with tempfile.NamedTemporaryFile(suffix='.py') as f:
        f.write(code)

        # Execute with timeout
        result = subprocess.run(
            ['python', f.name],
            capture_output=True,
            timeout=10,  # Safety limit
        )

        return result.stdout + result.stderr
```

## Security Considerations

**What's Implemented:**
- 10 second timeout (prevents infinite loops)
- Subprocess isolation (not in main process)
- Output capture (see what happened)

**What's NOT Implemented (Production Needs):**
- Network isolation (code can make HTTP requests)
- File system restrictions (code can read/write files)
- Resource limits (memory, CPU)
- Docker containerization

**For Production:**
```python
# Use Docker for real sandboxing
subprocess.run([
    'docker', 'run', '--rm',
    '--network=none',        # No network
    '--memory=100m',         # Memory limit
    '--cpus=0.5',            # CPU limit
    '-v', f'{temp_dir}:/code:ro',  # Read-only mount
    'python:3.10-slim',
    'python', '/code/script.py'
], timeout=10)
```

## Finance Applications

Code execution enables:
- **Custom calculations**: Greeks, risk metrics, portfolio optimization
- **Data transforms**: Cleaning, normalization, feature engineering
- **Backtesting**: Running trading strategies on historical data
- **Report generation**: Dynamic charts and tables

## Try It

1. **Test limits**: Ask for code that takes >10 seconds
2. **Complex analysis**: "Calculate the Sharpe ratio for these returns..."
3. **Add libraries**: Install numpy, try asking for matrix operations
4. **Security test**: What happens if you ask it to read /etc/passwd?

## The Path to Flawed Code

This code execution capability is one of the core features we need for "flawed_code" (example 08). Combined with file operations (example 07), we'll have the basic building blocks of a coding assistant.
