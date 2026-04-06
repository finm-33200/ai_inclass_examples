# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.4
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # OpenAI API Overview
#
# This notebook walks through the core capabilities of the OpenAI-compatible
# chat completions API, progressing from basic calls to advanced features.
#
# **Topics covered:**
# 1. Basic chat completion (OpenAI)
# 2. Comparing models via OpenRouter
# 3. Streaming responses
# 4. Structured outputs (JSON mode, JSON schema, Pydantic)
# 5. Function calling / tool use
# 6. Web search grounding

# %% [markdown]
# ## 1. Setup
#
# We create two clients:
# - **`client`** talks directly to OpenAI.
# - **`openrouter_client`** talks to OpenRouter, a unified gateway that
#   lets us hit models from Google, Qwen, Anthropic, and others through
#   the same OpenAI-compatible interface.

# %%
import json
import random
import time
from datetime import datetime
from pathlib import Path
from typing import Literal

from decouple import Config, RepositoryEnv
from openai import OpenAI
from pydantic import BaseModel, Field, ValidationError

# Load API keys from .env at the repository root
# When executed by nbconvert, the cwd may be the notebook's directory,
# so we walk up until we find the .env file.
_here = Path(".").resolve()
_repo_root = _here
for _parent in [_here] + list(_here.parents):
    if (_parent / ".env").exists():
        _repo_root = _parent
        break

config = Config(RepositoryEnv(_repo_root / ".env"))

# Direct OpenAI client
client = OpenAI(api_key=config("OPENAI_API_KEY"))

# OpenRouter client (unified gateway to many models)
openrouter_client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=config("OPENROUTER_API_KEY"),
)

# %% [markdown]
# ---
# ## 2. Basic Chat Completion
#
# The chat completions endpoint takes a list of **messages**, each with a
# `role` (`system`, `user`, or `assistant`) and `content`. The model
# returns a completion with token-usage metadata.

# %%
messages = [
    {
        "role": "system",
        "content": "You are a helpful assistant. Keep responses concise.",
    },
    {
        "role": "user",
        "content": "What is an LLM? Explain in 2-3 sentences.",
    },
]

print(f"User: {messages[-1]['content']}\n")

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=messages,
)

assistant_message = response.choices[0].message.content
print(f"Assistant:\n{assistant_message}")

print(f"\n--- Token Usage ---")
print(f"Prompt tokens: {response.usage.prompt_tokens}")
print(f"Completion tokens: {response.usage.completion_tokens}")
print(f"Total tokens: {response.usage.total_tokens}")

# %% [markdown]
# ---
# ## 3. Comparing Models via OpenRouter
#
# OpenRouter exposes many models through a single API. Here we send the
# same prompt to three models and compare their responses and latency.

# %%
MODELS = [
    "google/gemini-2.5-flash",    # Google model
    "anthropic/claude-3.5-haiku",  # Anthropic model
    "openai/gpt-4o-mini",         # OpenAI model
]


def call_model(client: OpenAI, model: str, prompt: str) -> tuple[str, float]:
    """Call a model and return (response_text, elapsed_seconds)."""
    messages = [
        {"role": "system", "content": "You are a helpful assistant. Be concise."},
        {"role": "user", "content": prompt},
    ]
    start = time.time()
    response = client.chat.completions.create(model=model, messages=messages)
    elapsed = time.time() - start
    return response.choices[0].message.content, elapsed


prompt = "What is the difference between stocks and bonds? Answer in 2 sentences."

print("=" * 80)
print(f"Prompt: {prompt}")
print("=" * 80)

for model in MODELS:
    print(f"\n--- {model} ---")
    try:
        text, elapsed = call_model(openrouter_client, model, prompt)
        print(f"Time: {elapsed:.2f}s")
        print(f"Response:\n{text}")
    except Exception as e:
        print(f"Error: {e}")

# %% [markdown]
# ---
# ## 4. Streaming Responses
#
# Setting `stream=True` returns tokens incrementally as the model
# generates them. In a terminal this creates a real-time typewriter
# effect. Streaming doesn't render well in notebooks (each token lands
# on its own line), so we show the pattern here without executing it.
#
# ```python
# stream = client.chat.completions.create(
#     model="gpt-4o-mini",
#     messages=messages,
#     stream=True,  # Enable streaming
# )
#
# for chunk in stream:
#     if chunk.choices[0].delta.content:
#         print(chunk.choices[0].delta.content, end="", flush=True)
# ```
#
# Each `chunk` contains a `delta` with the next token(s). The loop
# prints them without newlines so they appear as a continuous sentence.
# See `basic_llm_api/04_streaming_responses/stream_openai.py` for a
# runnable terminal example.

# %% [markdown]
# ---
# ## 5. Structured Outputs
#
# LLMs return free-form text by default. When we need machine-readable
# data we can constrain the output to JSON. There are three levels of
# strictness:
#
# | Approach | Guarantee |
# |---|---|
# | JSON mode | Valid JSON, but schema is best-effort |
# | JSON schema | Output **must** match an explicit schema |
# | Pydantic schema | Same as above, plus Python type validation |

# %% [markdown]
# ### Shared test data
#
# We'll reuse this earnings snippet across the structured-output examples.

# %%
financial_text = """
Apple Inc. (AAPL) reported Q4 2024 earnings that beat analyst expectations.
Revenue came in at $94.9 billion, up 6% year-over-year. The company's
Services segment showed strong growth at 12% YoY. iPhone sales were
slightly below expectations but Mac sales exceeded forecasts. CEO Tim Cook
noted concerns about China market conditions but remained optimistic about
AI product integration. The company announced a $110 billion share buyback
program.
"""

# %% [markdown]
# ### 5a. JSON Mode
#
# The simplest approach: ask the model to return JSON and set
# `response_format={"type": "json_object"}`. The model will produce valid
# JSON, but there is no schema enforcement -- the structure depends on the
# prompt.

# %%
messages = [
    {
        "role": "system",
        "content": """You are a financial analyst. Extract key information
        from the provided text and return it as JSON with these fields:
        - ticker: string
        - company_name: string
        - revenue_billions: number or null
        - sentiment: "bullish", "bearish", or "neutral"
        - key_points: array of strings (2-3 main takeaways)

        Return ONLY valid JSON, no other text.""",
    },
    {
        "role": "user",
        "content": f"Analyze this text:\n\n{financial_text}",
    },
]

response = openrouter_client.chat.completions.create(
    model="openai/gpt-4o-mini",
    messages=messages,
    response_format={"type": "json_object"},
)

data = json.loads(response.choices[0].message.content)

print("Extracted data (JSON mode):")
print(json.dumps(data, indent=2))

# %% [markdown]
# ### 5b. Explicit JSON Schema
#
# We define an exact JSON schema and pass it via `response_format`. The
# model is **required** to produce output matching this schema -- field
# names, types, and enum values are enforced.

# %%
company_analysis_schema = {
    "type": "object",
    "properties": {
        "ticker": {"type": "string", "description": "Stock ticker symbol"},
        "company_name": {"type": "string", "description": "Full company name"},
        "revenue_billions": {
            "type": ["number", "null"],
            "description": "Reported revenue in billions USD, or null if not mentioned",
        },
        "sentiment": {
            "type": "string",
            "enum": ["bullish", "bearish", "neutral"],
            "description": "Overall sentiment based on the text",
        },
        "key_points": {
            "type": "array",
            "items": {"type": "string"},
            "description": "2-4 key takeaways from the text",
        },
    },
    "required": ["ticker", "company_name", "revenue_billions", "sentiment", "key_points"],
    "additionalProperties": False,
}

messages = [
    {
        "role": "system",
        "content": "You are a financial analyst. Extract structured data from the provided text.",
    },
    {"role": "user", "content": f"Analyze this text:\n\n{financial_text}"},
]

response = openrouter_client.chat.completions.create(
    model="openai/gpt-4o-mini",
    messages=messages,
    response_format={
        "type": "json_schema",
        "json_schema": {
            "name": "company_analysis",
            "strict": True,
            "schema": company_analysis_schema,
        },
    },
)

data = json.loads(response.choices[0].message.content)

print("Extracted data (schema-enforced):")
print(json.dumps(data, indent=2))

print(f"\nCompany: {data['company_name']} ({data['ticker']})")
print(f"Revenue: ${data['revenue_billions']}B")
print(f"Sentiment: {data['sentiment']}")

print("\nKey Points:")
for i, point in enumerate(data["key_points"], 1):
    print(f"  {i}. {point}")

# %% [markdown]
# ### 5c. Pydantic Schema
#
# Pydantic gives us a Python class with typed fields, automatic schema
# generation, and validation. We convert the Pydantic schema to the
# strict format the API expects, then validate the response back into a
# typed object.

# %%
class CompanyAnalysis(BaseModel):
    """Structured analysis of a company from financial text."""

    ticker: str = Field(description="Stock ticker symbol")
    company_name: str = Field(description="Full company name")
    revenue_billions: float | None = Field(
        description="Reported revenue in billions USD, or null if not mentioned"
    )
    revenue_growth_pct: float | None = Field(
        description="Year-over-year revenue growth percentage, or null if not mentioned"
    )
    sentiment: Literal["bullish", "bearish", "neutral"] = Field(
        description="Overall sentiment based on the text"
    )
    confidence: float = Field(
        description="Confidence in the analysis from 0.0 to 1.0", ge=0.0, le=1.0
    )
    key_points: list[str] = Field(description="2-4 key takeaways from the text")
    risk_factors: list[str] = Field(
        description="Any mentioned risks or concerns", default_factory=list
    )


def pydantic_to_strict_schema(model: type[BaseModel]) -> dict:
    """
    Convert a Pydantic model to a strict JSON schema compatible with
    OpenAI / OpenRouter.

    Strict mode requires:
    - additionalProperties: false on all objects
    - All properties listed in 'required'
    - Nullable fields use anyOf with null type
    """
    schema = model.model_json_schema()
    schema["additionalProperties"] = False
    if "properties" in schema:
        schema["required"] = list(schema["properties"].keys())
    if "$defs" in schema:
        del schema["$defs"]
    return schema

# %%
messages = [
    {
        "role": "system",
        "content": "You are a financial analyst. Extract structured data from the provided text.",
    },
    {"role": "user", "content": f"Analyze this text:\n\n{financial_text}"},
]

schema = pydantic_to_strict_schema(CompanyAnalysis)

response = openrouter_client.chat.completions.create(
    model="openai/gpt-4o-mini",
    messages=messages,
    response_format={
        "type": "json_schema",
        "json_schema": {
            "name": "CompanyAnalysis",
            "strict": True,
            "schema": schema,
        },
    },
)

# Validate the response into a typed Pydantic object
analysis = CompanyAnalysis.model_validate_json(response.choices[0].message.content)

print("Extracted Analysis:")
print(f"  Company: {analysis.company_name} ({analysis.ticker})")
print(f"  Revenue: ${analysis.revenue_billions}B")
print(f"  Growth: {analysis.revenue_growth_pct}%")
print(f"  Sentiment: {analysis.sentiment}")
print(f"  Confidence: {analysis.confidence:.0%}")

print("\nKey Points:")
for i, point in enumerate(analysis.key_points, 1):
    print(f"  {i}. {point}")

if analysis.risk_factors:
    print("\nRisk Factors:")
    for risk in analysis.risk_factors:
        print(f"  - {risk}")

print("\n--- As JSON ---")
print(analysis.model_dump_json(indent=2))

# %% [markdown]
# ### 5d. Model Comparison -- Structured Output Reliability
#
# Not every model handles JSON schema enforcement equally well. Here we
# test the same structured-output request against two models and compare.

# %%
class CompanyAnalysisSimple(BaseModel):
    """Simpler schema for cross-model comparison."""

    ticker: str = Field(description="Stock ticker symbol")
    company_name: str = Field(description="Full company name")
    revenue_billions: float | None = Field(
        description="Reported revenue in billions USD, or null if not mentioned"
    )
    sentiment: Literal["bullish", "bearish", "neutral"] = Field(
        description="Overall sentiment based on the text"
    )
    key_points: list[str] = Field(description="2-4 key takeaways from the text")


def test_model(client: OpenAI, model: str, messages: list, schema: dict) -> dict:
    """Test a model's structured output capability and return results."""
    result = {"model": model, "success": False, "error": None, "data": None}
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=messages,
            response_format={
                "type": "json_schema",
                "json_schema": {"name": "CompanyAnalysisSimple", "strict": True, "schema": schema},
            },
        )
        raw = resp.choices[0].message.content
        data = json.loads(raw)
        validated = CompanyAnalysisSimple.model_validate(data)
        result["data"] = validated.model_dump()
        result["success"] = True
    except json.JSONDecodeError as e:
        result["error"] = f"JSON parse error: {e}"
    except ValidationError as e:
        result["error"] = f"Pydantic validation error: {e}"
    except Exception as e:
        result["error"] = f"API error: {e}"
    return result

# %%
models_to_compare = [
    "openai/gpt-4o-mini",
    "google/gemini-2.5-flash",
]

messages = [
    {
        "role": "system",
        "content": "You are a financial analyst. Extract structured data from the provided text.",
    },
    {"role": "user", "content": f"Analyze this text:\n\n{financial_text}"},
]

schema = pydantic_to_strict_schema(CompanyAnalysisSimple)

print("=" * 60)
print("STRUCTURED OUTPUT MODEL COMPARISON")
print("=" * 60)

for model in models_to_compare:
    print(f"\n--- {model} ---")
    result = test_model(openrouter_client, model, messages, schema)
    if result["success"]:
        print("Status: SUCCESS")
        print(f"Data: {json.dumps(result['data'], indent=2)}")
    else:
        print("Status: FAILED")
        print(f"Error: {result['error']}")

print(f"\n{'=' * 60}")
print("Key takeaways:")
print("- GPT-4o-mini reliably follows JSON schema constraints")
print("- Some models (especially free tiers) may have compatibility issues")
print("- Always validate responses with Pydantic for type safety")

# %% [markdown]
# ---
# ## 6. Function Calling / Tool Use
#
# Function calling lets the model decide **when** to invoke tools you
# define. The flow is a loop:
#
# 1. Send messages + tool definitions to the model
# 2. If the model returns `tool_calls`, execute them locally
# 3. Append the tool results and call the model again
# 4. Repeat until the model returns a plain text answer

# %% [markdown]
# ### 6a. Simple Tools
#
# Two basic tools: `get_current_time` and `calculate`.

# %%
SIMPLE_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_current_time",
            "description": "Get the current date and time",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": "Perform a mathematical calculation",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "Math expression to evaluate (e.g., '2 + 2', '100 * 1.05')",
                    }
                },
                "required": ["expression"],
            },
        },
    },
]


def get_current_time() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def calculate(expression: str) -> str:
    allowed = set("0123456789+-*/.(). ")
    if not all(c in allowed for c in expression):
        return "Error: Invalid characters in expression"
    try:
        result = eval(expression)
        return str(result)
    except Exception as e:
        return f"Error: {e}"


def execute_simple_tool(name: str, arguments: dict) -> str:
    if name == "get_current_time":
        return get_current_time()
    elif name == "calculate":
        return calculate(arguments["expression"])
    return f"Unknown tool: {name}"

# %%
user_question = "What time is it, and what is 15% of 850?"

messages = [
    {
        "role": "system",
        "content": "You are a helpful assistant. Use the available tools to answer questions accurately.",
    },
    {"role": "user", "content": user_question},
]

print(f"User: {user_question}\n")

iteration = 0
max_iterations = 10

while iteration < max_iterations:
    iteration += 1
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        tools=SIMPLE_TOOLS,
    )
    message = response.choices[0].message

    if message.tool_calls:
        messages.append(message)
        print("Tool calls requested:")
        for tool_call in message.tool_calls:
            name = tool_call.function.name
            args = json.loads(tool_call.function.arguments)
            print(f"  - {name}({args})")
            result = execute_simple_tool(name, args)
            print(f"    Result: {result}")
            messages.append(
                {"role": "tool", "tool_call_id": tool_call.id, "content": result}
            )
        print()
    else:
        print(f"Assistant: {message.content}")
        break

# %% [markdown]
# ### 6b. Finance Tools
#
# A more realistic example with simulated stock data. The model must
# call multiple tools (get prices, info, calculate returns) to answer
# a comparative question.

# %%
FINANCE_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_stock_price",
            "description": "Get the current price of a stock",
            "parameters": {
                "type": "object",
                "properties": {
                    "ticker": {"type": "string", "description": "Stock ticker symbol (e.g., AAPL, MSFT)"}
                },
                "required": ["ticker"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_stock_info",
            "description": "Get basic information about a stock including market cap and P/E ratio",
            "parameters": {
                "type": "object",
                "properties": {
                    "ticker": {"type": "string", "description": "Stock ticker symbol"}
                },
                "required": ["ticker"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculate_return",
            "description": "Calculate the percentage return between two prices",
            "parameters": {
                "type": "object",
                "properties": {
                    "start_price": {"type": "number", "description": "Starting price"},
                    "end_price": {"type": "number", "description": "Ending price"},
                },
                "required": ["start_price", "end_price"],
            },
        },
    },
]

# Simulated stock data (in production, use a real API like yfinance)
STOCK_DATA = {
    "AAPL": {"price": 178.50, "name": "Apple Inc.", "pe": 28.5, "market_cap": "2.8T"},
    "MSFT": {"price": 378.90, "name": "Microsoft Corp.", "pe": 35.2, "market_cap": "2.9T"},
    "GOOGL": {"price": 141.20, "name": "Alphabet Inc.", "pe": 24.1, "market_cap": "1.8T"},
    "AMZN": {"price": 178.25, "name": "Amazon.com Inc.", "pe": 62.3, "market_cap": "1.9T"},
    "NVDA": {"price": 875.30, "name": "NVIDIA Corp.", "pe": 65.8, "market_cap": "2.2T"},
}

random.seed(42)


def get_stock_price(ticker: str) -> str:
    ticker = ticker.upper()
    if ticker in STOCK_DATA:
        base_price = STOCK_DATA[ticker]["price"]
        price = base_price * (1 + random.uniform(-0.02, 0.02))
        return json.dumps({"ticker": ticker, "price": round(price, 2), "currency": "USD"})
    return json.dumps({"error": f"Unknown ticker: {ticker}"})


def get_stock_info(ticker: str) -> str:
    ticker = ticker.upper()
    if ticker in STOCK_DATA:
        d = STOCK_DATA[ticker]
        return json.dumps(
            {"ticker": ticker, "name": d["name"], "price": d["price"], "pe_ratio": d["pe"], "market_cap": d["market_cap"]}
        )
    return json.dumps({"error": f"Unknown ticker: {ticker}"})


def calculate_return(start_price: float, end_price: float) -> str:
    if start_price <= 0:
        return json.dumps({"error": "Start price must be positive"})
    pct_return = ((end_price - start_price) / start_price) * 100
    return json.dumps({"start_price": start_price, "end_price": end_price, "return_pct": round(pct_return, 2)})


def execute_finance_tool(name: str, arguments: dict) -> str:
    if name == "get_stock_price":
        return get_stock_price(arguments["ticker"])
    elif name == "get_stock_info":
        return get_stock_info(arguments["ticker"])
    elif name == "calculate_return":
        return calculate_return(arguments["start_price"], arguments["end_price"])
    return json.dumps({"error": f"Unknown tool: {name}"})

# %%
user_question = (
    "Compare AAPL and NVDA - which has a better P/E ratio? "
    "Also, if I bought AAPL at $150 and it's now at its current price, what's my return?"
)

messages = [
    {
        "role": "system",
        "content": "You are a financial analyst assistant. Use the available tools "
        "to get accurate data. Always cite the actual numbers from the tools.",
    },
    {"role": "user", "content": user_question},
]

print(f"User: {user_question}\n")

iteration = 0
max_iterations = 10

while iteration < max_iterations:
    iteration += 1
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        tools=FINANCE_TOOLS,
    )
    message = response.choices[0].message

    if message.tool_calls:
        messages.append(message)
        print(f"[Iteration {iteration}] Tool calls:")
        for tool_call in message.tool_calls:
            name = tool_call.function.name
            args = json.loads(tool_call.function.arguments)
            print(f"  {name}({args})")
            result = execute_finance_tool(name, args)
            print(f"    -> {result}")
            messages.append(
                {"role": "tool", "tool_call_id": tool_call.id, "content": result}
            )
        print()
    else:
        print(f"Assistant:\n{message.content}")
        break

# %% [markdown]
# ---
# ## 7. Web Search
#
# LLM knowledge has a cutoff date. Web search tools let the model ground
# its answers in current information -- critical for finance where data
# changes daily.

# %% [markdown]
# ### 7a. OpenAI Web Search (Responses API)
#
# OpenAI's Responses API has a built-in `web_search_preview` tool. The
# model decides when to search and incorporates results automatically.

# %%
question = (
    "What is the current Federal Reserve interest rate target, "
    "and what are market expectations for the next Fed meeting?"
)

print(f"Question: {question}\n")
print("Searching the web and generating response...\n")

response = client.responses.create(
    model="gpt-4o-mini",
    tools=[{"type": "web_search_preview"}],
    input=question,
)

print("Answer:")
print(response.output_text)
print()

print("--- Search Details ---")
for item in response.output:
    if hasattr(item, "type") and item.type == "web_search_call":
        print(f"Query: {item.query if hasattr(item, 'query') else 'N/A'}")

# %% [markdown]
# ### 7b. OpenRouter Web Search
#
# OpenRouter enables web search by appending `:online` to any model slug.
# This works across providers -- GPT, Claude, Gemini, etc.

# %%
question = (
    "What is the current Federal Reserve interest rate target, "
    "and what are market expectations for the next Fed meeting?"
)

print(f"Question: {question}\n")
print("Searching the web and generating response...\n")

response = openrouter_client.chat.completions.create(
    model="openai/gpt-4o-mini:online",
    messages=[{"role": "user", "content": question}],
    extra_body={
        "plugins": [{"id": "web", "max_results": 5}],
    },
)

print("Answer:")
print(response.choices[0].message.content)

# Show any citations if available
message = response.choices[0].message
if hasattr(message, "annotations") and message.annotations:
    print("\n--- Sources ---")
    for annotation in message.annotations:
        if hasattr(annotation, "url"):
            print(f"- {annotation.url}")

# %% [markdown]
# ---
# ## Summary
#
# | Capability | Key API parameter |
# |---|---|
# | Basic completion | `client.chat.completions.create(model=..., messages=...)` |
# | Model comparison | Change `model` or `base_url` (OpenRouter) |
# | Streaming | `stream=True` |
# | JSON mode | `response_format={"type": "json_object"}` |
# | JSON schema | `response_format={"type": "json_schema", "json_schema": {...}}` |
# | Pydantic schema | Same as above, schema from `model.model_json_schema()` |
# | Function calling | `tools=[...]`, then handle `message.tool_calls` in a loop |
# | Web search (OpenAI) | `client.responses.create(tools=[{"type": "web_search_preview"}])` |
# | Web search (OpenRouter) | Append `:online` to model slug |
