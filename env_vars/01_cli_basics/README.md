# 01 CLI Basics

Set an environment variable on the command line, then access it.

## Steps

1. Set the variable:
   ```bash
   export MYVAR="hello world"
   ```
   (Windows: `set MYVAR=hello world`)

2. Verify with echo:
   ```bash
   echo $MYVAR
   ```
   (Windows: `echo %MYVAR%`)

3. Access from Python:
   ```bash
   python print_myvar.py
   ```

## Try it

- What happens if you run `print_myvar.py` without setting MYVAR first?
- What happens if you close the terminal and reopen it?
