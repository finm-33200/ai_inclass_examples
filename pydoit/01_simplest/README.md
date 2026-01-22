# 01_simplest: The Most Basic PyDoit Example

This is the simplest possible example of using PyDoit (doit) as a task runner.

## What This Example Demonstrates

- **Task definitions**: Functions starting with `task_` are recognized as tasks
- **File dependencies**: Tasks can depend on files produced by other tasks
- **Targets**: Each task declares what files it produces
- **Automatic ordering**: doit figures out the correct order to run tasks

## Project Structure

```
01_simplest/
├── dodo.py           # Task runner configuration
├── create_data.py    # Script to create initial data
├── process_data.py   # Script to process the data
├── create_report.py  # Script to create the final report
├── _output/          # Generated files (created automatically)
│   ├── data.txt
│   ├── processed.txt
│   └── report.txt
└── README.md
```

## The Workflow

```
task_create_data ──► task_process_data ──► task_create_report
     │                      │                      │
     ▼                      ▼                      ▼
 data.txt            processed.txt            report.txt
```

## How to Use

Make sure you have doit installed:
```bash
pip install doit
```

Then run from this directory:

```bash
# Run all tasks
doit

# List available tasks
doit list

# Run a specific task (and its dependencies)
doit create_report

# See detailed info about a task
doit info create_data

# Clean up all generated files
doit clean

# Forget task state (force re-run next time)
doit forget
```

## Key Concepts

### 1. Task Functions

Any function starting with `task_` is recognized as a task:

```python
def task_create_data():
    return {
        'actions': ['python create_data.py'],
        'file_dep': ['create_data.py'],
        'targets': ['_output/data.txt'],
    }
```

### 2. Actions

Actions are shell commands that run the Python scripts:
```python
'actions': ['python create_data.py']
```

### 3. Dependencies

- `file_dep`: Files that must exist before the task runs
- If a file in `file_dep` changes, the task will re-run
- doit automatically runs tasks that produce those files first

### 4. Targets

- Files that the task creates
- Used by doit to track what's been done
- If a target exists and dependencies haven't changed, task is skipped

## Try It!

1. Run `doit` - all three tasks run
2. Run `doit` again - nothing runs (everything up-to-date)
3. Edit `create_data.py` - then run `doit` - all tasks re-run
4. Edit `_output/data.txt` - then run `doit` - only dependent tasks re-run
5. Run `doit clean` - removes all generated files
6. Run `doit` - everything runs again from scratch
