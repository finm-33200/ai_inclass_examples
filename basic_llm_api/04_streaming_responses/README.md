# 04 - Streaming Responses

Stream LLM responses token-by-token for a more responsive user experience.

## Why Streaming?

Without streaming, you wait for the entire response before seeing anything. With streaming:

- **Better UX**: Users see output immediately
- **Faster perceived latency**: First token appears quickly
- **Interruptible**: Can stop generation mid-response
- **Progress indication**: Users know the model is working

## What You'll Learn

- How to enable streaming in API calls
- Processing streamed chunks
- Printing tokens as they arrive

## Setup

```bash
cd 04_streaming_responses
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Examples

### OpenAI Streaming
```bash
python stream_openai.py
```

### OpenRouter Streaming
```bash
python stream_openrouter.py
```

## Key Concepts

### Enabling Streaming

Add `stream=True` to your API call:

```python
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=messages,
    stream=True,  # Enable streaming
)
```

### Processing Streamed Chunks

The response becomes an iterator of chunks:

```python
for chunk in response:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="", flush=True)
```

### Delta vs Message

- **Non-streaming**: `response.choices[0].message.content` (complete)
- **Streaming**: `chunk.choices[0].delta.content` (incremental)

## Try It

1. Run both scripts and observe the token-by-token output
2. Compare the perceived latency with non-streaming calls
3. Try a longer prompt: "Write a brief market analysis of the tech sector"
4. What happens if you remove `flush=True` from the print statement?
