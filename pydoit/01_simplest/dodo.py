"""
The Simplest dodo.py Example
============================

This is the most basic example of a dodo.py file for PyDoit.
It demonstrates three tasks with dependencies between them:

1. task_create_data - Creates initial data (no dependencies)
2. task_process_data - Processes the data (depends on task 1's output)
3. task_create_report - Creates a report (depends on task 2's output)

To run:
    doit              # Run all tasks
    doit list         # List all tasks
    doit info <task>  # Show task details
    doit clean        # Remove all target files
    doit forget       # Clear the dependency database
    doit -a          # Run all tasks and print the commands that are being executed
    doit process_data
    doit -sa process_data
    doit -s process_data

Key concepts:
- 'actions': List of commands or Python functions to execute
- 'file_dep': Files that the task depends on (task runs if these change)
- 'targets': Files that the task produces (used for dependency tracking)
"""


# =============================================================================
# Task definitions (PyDoit looks for functions starting with 'task_')
# =============================================================================

def task_create_data():
    """Task: Create the initial data file."""
    return {
        'actions': ['python create_data.py'],
        'file_dep': ['create_data.py'],
        'targets': ['_output/data.txt'],
        'clean': True,  # Allow 'doit clean' to remove targets
    }


def task_process_data():
    """Task: Process the data file."""
    return {
        'actions': ['python process_data.py'],
        'file_dep': [
            'process_data.py',
            '_output/data.txt',  # Depends on output of task_create_data
        ],
        'targets': ['_output/processed.txt'],
        'clean': True,
    }


def task_create_report():
    """Task: Create the final report."""
    return {
        'actions': ['python create_report.py'],
        'file_dep': [
            'create_report.py',
            '_output/processed.txt',  # Depends on output of task_process_data
        ],
        'targets': ['_output/report.txt'],
        'clean': True,
    }
