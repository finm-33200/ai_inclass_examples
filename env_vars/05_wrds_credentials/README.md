# 05 WRDS Credentials

A practical example: storing WRDS credentials using environment variables.

## Setup

1. Install the package:
   ```bash
   pip install -r requirements.txt
   ```

2. Copy the example file and add your username:
   ```bash
   cp .env.example .env
   ```

3. Edit `.env` with your WRDS username.

## Run

```bash
python demo_wrds_config.py
```

## Key points

- NEVER commit `.env` with real credentials to git
- `.env.example` documents which variables are needed without exposing values
- For the password, WRDS uses `~/.pgpass` (see course materials)

## Course materials

- https://finm-32900.github.io/Week2/env_files.html
- https://finm-32900.github.io/notebooks/_01_wrds_python_package.html
