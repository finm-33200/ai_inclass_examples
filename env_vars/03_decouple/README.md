# 03 Decouple

Use `python-decouple` for configuration with defaults and type casting.

## Setup

1. Install the package:
   ```bash
   pip install -r requirements.txt
   ```

2. Copy the example file:
   ```bash
   cp .env.example .env
   ```

## Run

```bash
python demo_decouple.py
```

## Key points

- `config()` reads from `.env` (or environment variables)
- `default=` provides a fallback if the variable is not set
- `cast=bool` or `cast=int` converts strings to proper types

## Try it

- Delete `.env` and run again. What happens?
- Change `DEBUG=False` in `.env` and see the type conversion work.
