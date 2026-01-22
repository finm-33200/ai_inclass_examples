"""
Demonstrates accessing an environment variable from Python.

Run this after setting MYVAR in your shell:
    export MYVAR="hello world"
    python print_myvar.py
"""
import os

value = os.getenv("MYVAR")
print(f"MYVAR = {value}")
