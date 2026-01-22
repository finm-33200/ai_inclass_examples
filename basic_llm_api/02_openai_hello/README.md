# 02 - OpenAI Hello World

Basic API call using OpenAI's GPT models - the industry standard for LLM APIs.

## Why OpenAI?

OpenAI's API is the most widely used LLM API. Most tooling, documentation, and tutorials are built around it. Understanding this API makes it easier to work with other providers that follow similar patterns.

## What You'll Learn

- The OpenAI chat completions API pattern
- Message roles: system, user, and assistant
- Using the OpenAI Python SDK

## Setup

### 1. Get an API Key

1. Go to [platform.openai.com](https://platform.openai.com/)
2. Sign up or log in
3. Navigate to API Keys in your account settings
4. Create a new API key

Note: OpenAI requires a credit card and charges per token. Start with small prompts to minimize costs.

### 2. Configure Environment

Add your key to the `.env` file at the repository root:

```
OPENAI_API_KEY=your_api_key_here
```

### 3. Install Dependencies

```bash
cd 02_openai_hello
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Run

```bash
python hello.py
```

## Key Concepts

### Message Roles

OpenAI uses a conversation format with three roles:

- **system**: Sets the behavior/persona of the assistant
- **user**: The human's input
- **assistant**: The model's responses

```python
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What is an LLM?"}
]
```

### Models

- `gpt-4o-mini` - Fast, cheap, good for most tasks
- `gpt-4o` - More capable, better for complex reasoning
- `gpt-4-turbo` - Older but still capable
- `o1-mini`, `o1` - Reasoning models (different API pattern)

### Tokens and Pricing

OpenAI charges per token (roughly 4 characters = 1 token). Check current pricing at [openai.com/pricing](https://openai.com/pricing).

## Try It

1. Modify the system message to make the assistant respond like a finance expert
2. Change the model from `gpt-4o-mini` to `gpt-4o` and compare response quality
3. Add a multi-turn conversation by including previous messages
4. Try: "Explain the yield curve in simple terms"
