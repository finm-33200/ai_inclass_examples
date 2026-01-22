# 01 - Gemini Hello World

The absolute simplest LLM API call using Google's Gemini model.

## Why Start Here?

Google AI Studio offers a **free tier** with no credit card required. This is the lowest barrier to entry for experimenting with LLM APIs.

## What You'll Learn

- How to obtain and configure an API key
- The basic pattern for making an LLM API call
- Understanding prompts and responses

## Setup

### 1. Get a Free API Key

1. Go to [Google AI Studio](https://aistudio.google.com/)
2. Sign in with your Google account
3. Click "Get API Key" in the left sidebar
4. Create a new API key or use an existing one

### 2. Configure Environment

Add your key to the `.env` file at the repository root:

```
GOOGLE_API_KEY=your_api_key_here
```

### 3. Install Dependencies

```bash
cd 01_gemini_hello
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Run

```bash
python hello.py
```

## Key Concepts

### API Keys
API keys authenticate your requests. Keep them secret - never commit them to git.

### The Basic Pattern
Every LLM API call follows this pattern:
1. Configure the client with your API key
2. Create a prompt (the input text)
3. Call the model
4. Process the response

### Models
Gemini offers several model variants (as of 2025):
- `gemini-2.0-flash-lite` - Most generous free tier quota, good for learning
- `gemini-2.0-flash` - Fast and capable, good for most tasks
- `gemini-2.5-flash` - Newer version with improved capabilities
- `gemini-2.5-pro` - Most capable, better for complex reasoning

## Try It

1. Modify the prompt in `hello.py` to ask a different question
2. Try changing the model from `gemini-2.0-flash-lite` to `gemini-2.5-flash`
3. What happens if you provide an invalid API key?
4. Try asking a finance-related question: "What is the P/E ratio and why do investors use it?"
