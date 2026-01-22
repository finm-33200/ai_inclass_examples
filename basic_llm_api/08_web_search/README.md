# 08 - Web Search

Ground LLM responses with current information from the web.

## Why Web Search?

LLMs have knowledge cutoffs and can't access real-time information. Web search enables:
- **Current data**: Latest stock prices, news, economic indicators
- **Grounding**: Reduce hallucinations with sourced facts
- **Citations**: Provide links to sources

## What You'll Learn

- Using OpenAI's built-in web search
- Combining search results with LLM reasoning
- Handling citations in responses

## Setup

```bash
cd 08_web_search
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Examples

### OpenAI Web Search
```bash
python openai_web_search.py
```

## Key Concepts

### OpenAI Responses API with Web Search

OpenAI's newer Responses API supports built-in web search:

```python
response = client.responses.create(
    model="gpt-4o",
    tools=[{"type": "web_search_preview"}],
    input="What is the current Fed funds rate?",
)
```

### Handling Search Results

The response includes both the answer and citations:

```python
# The model's answer
print(response.output_text)

# Citations are included in the response
for item in response.output:
    if item.type == "web_search_call":
        print(f"Searched for: {item.query}")
```

### Limitations

- Web search adds latency
- Results may be cached
- Some sources may be unreliable
- Cost is higher than plain LLM calls

## Finance Applications

- "What are the latest Fed interest rate expectations?"
- "Summarize today's market movements"
- "What are analysts saying about [company]'s earnings?"
- "What is the current 10-year Treasury yield?"

## Try It

1. Ask about a recent financial event
2. Compare the web-grounded answer to a plain LLM response
3. Try asking about very recent news (last 24 hours)
4. Notice how the model cites its sources
