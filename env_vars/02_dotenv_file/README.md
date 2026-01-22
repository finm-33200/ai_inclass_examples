# 02 Dotenv File

Load environment variables from a `.env` file using `python-dotenv`.

## Setup

1. Install the package:
   ```bash
   pip install -r requirements.txt
   ```

2. Copy the example file:
   ```bash
   cp .env.example .env
   ```

3. Edit `.env` with your values (optional).

## Run

```bash
python demo_dotenv.py
```

## Key points

- `.env` files should NOT be committed to git (see `.gitignore`)
- `.env.example` shows the structure without real values
- `load_dotenv()` reads `.env` into `os.environ`
