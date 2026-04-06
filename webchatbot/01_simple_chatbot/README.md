# 01 Simple Chatbot

The simplest possible web chatbot: a Gradio app connected to the OpenAI API.

## What You'll Learn

- How to use **Gradio's `ChatInterface`** to create a web chat UI with zero HTML/CSS/JS
- How to wire up the **OpenAI Chat Completions API** to a web interface
- How conversation history is passed to the API for multi-turn chat

## Key Concepts

- **Gradio** is a Python library that creates web UIs from simple functions. `gr.ChatInterface` expects a function with signature `(message, history) -> response`.
- The `history` parameter is a list of `{"role": ..., "content": ...}` dicts that Gradio manages automatically, matching the OpenAI message format.

## Setup

```bash
cd webchatbot/01_simple_chatbot
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Make sure your repo-root `.env` file contains:

```
OPENAI_API_KEY=your_openai_api_key
```

## How to Run

```bash
python app.py
```

Open http://127.0.0.1:7860 in your browser. Type a message and press Enter.

## Try It

1. Have a multi-turn conversation and verify the model remembers prior messages.
2. Change the model from `gpt-4o-mini` to `gpt-4o` and compare response quality.
3. Add a `system` message at the start of the messages list to give the chatbot a persona (e.g., "You are a helpful finance tutor.").
