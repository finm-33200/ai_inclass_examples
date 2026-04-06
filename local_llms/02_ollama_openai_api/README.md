# 02_ollama_openai_api - OpenAI SDK Compatibility

Use the familiar OpenAI Python SDK to interact with local Ollama models. This demonstrates how easy it is to switch between cloud APIs and local models.

## What You'll Learn

- Ollama's OpenAI-compatible API endpoint
- Using the OpenAI SDK with local models
- Streaming responses for real-time output

## Key Concepts

- **API Compatibility**: Ollama exposes an OpenAI-compatible endpoint at `http://localhost:11434/v1`
- **Drop-in Replacement**: Same code works with OpenAI cloud or local Ollama - just change the base URL
- **Streaming**: Get tokens as they're generated for responsive UIs

## Prerequisites

- Ollama installed and running (`ollama serve`)
- TinyLlama model pulled (`ollama pull tinyllama:1.1b`)

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Running the Examples

```bash
# Basic completion
python 01_basic_completion.py

# Streaming tokens
python 02_streaming.py
```

## Try It

1. Compare response times between local and cloud APIs
2. Try different models: change `model="tinyllama:1.1b"` to `model="phi3:mini"` (after pulling)
3. Modify the code to use your OpenAI API key and see how similar the interface is

## Why This Matters

Many applications are built with the OpenAI SDK. This compatibility means:
- Easy migration between cloud and local
- Test locally before incurring API costs
- Privacy-sensitive workloads can stay on-premises
- No code changes needed beyond the base URL
