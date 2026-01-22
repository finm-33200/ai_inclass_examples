# 03 - OpenRouter Hello World

Access multiple LLM providers through a single API using OpenRouter.

## Why OpenRouter?

OpenRouter is an aggregator that provides access to models from OpenAI, Anthropic, Google, Meta, Mistral, and many others through a unified API. Benefits:

- **Model flexibility**: Switch between providers without code changes
- **Cost comparison**: Easily compare pricing across models
- **Unified API**: Uses the OpenAI SDK format you already know
- **Free models**: Some models have free tiers

## What You'll Learn

- Using OpenRouter with the OpenAI SDK
- Listing available models and their pricing
- Running the same prompt across different models

## Setup

### 1. Get an API Key

1. Go to [openrouter.ai](https://openrouter.ai/)
2. Sign up or log in
3. Navigate to Keys in your account
4. Create a new API key

### 2. Configure Environment

Add your key to the `.env` file at the repository root:

```
OPENROUTER_API_KEY=your_api_key_here
```

### 3. Install Dependencies

```bash
cd 03_openrouter_hello
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Examples

### Basic Call
```bash
python hello.py
```

### List Available Models
```bash
python list_models.py
```

### Compare Models
```bash
python compare_simple.py
```

## Key Concepts

### Using OpenAI SDK with OpenRouter

OpenRouter uses the same API format as OpenAI. Just change the base URL:

```python
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=openrouter_key,
)
```

### Model Naming

OpenRouter uses the format `provider/model-name`:
- `openai/gpt-4o-mini`
- `anthropic/claude-3-5-sonnet`
- `google/gemini-2.0-flash-exp:free`
- `meta-llama/llama-3.1-70b-instruct`

### Free Models

Some models have free tiers (marked with `:free` suffix):
- `google/gemini-2.0-flash-exp:free`
- `meta-llama/llama-3.2-3b-instruct:free`

## Try It

1. Run `list_models.py` and find the cheapest model for your use case
2. Modify `compare_simple.py` to test different models
3. Try a free model vs a paid model - notice any quality differences?
4. Compare response times across different providers
