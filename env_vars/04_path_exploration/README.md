# 04 PATH Exploration

Understand the PATH environment variable.

## What is PATH?

PATH is a list of directories where your shell looks for executables.
When you type `python`, `git`, or any command, the shell searches these directories in order.

- On Unix/Mac: directories are separated by `:`
- On Windows: directories are separated by `;`

## Run

View your PATH from the shell:
```bash
echo $PATH
```
(Windows: `echo %PATH%`)

View it from Python:
```bash
python explore_path.py
```

## Key points

- First match wins (order matters)
- When you activate a conda environment, it prepends its `bin/` directory to PATH
- This is why `python` points to different interpreters in different environments
