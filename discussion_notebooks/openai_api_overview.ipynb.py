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
# 7. Vision & OCR

# %% [markdown]
# ## 1. Setup
#
# We create two clients:
# - **`client`** talks directly to OpenAI.
# - **`openrouter_client`** talks to OpenRouter, a unified gateway that
#   lets us hit models from Google, Qwen, Anthropic, and others through
#   the same OpenAI-compatible interface.

# %%
import base64
import io
import json
import random
import time
from datetime import datetime, timedelta
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
# So far, the model has only produced text. But what if it needs to
# **do** something — check the time, run a calculation, look up a stock
# price, or query a database? The model can't do any of these things
# itself. It has no access to the internet, no file system, no
# calculator. It can only read and write text.
#
# **Function calling** (also called "tool use") solves this by splitting
# the work in two:
#
# - **The model** decides *what* to call and *with what arguments*.
# - **Your code** actually executes the function and returns the result.
#
# The model is the **brain** (it reasons about what information it needs);
# your code is the **hands** (it acts on the world). The API is the
# nervous system connecting them.
#
# ### Why this design?
#
# Why doesn't the model just execute the tools itself on OpenAI's
# servers? Three reasons:
#
# 1. **Security.** Tools often touch sensitive systems — databases,
#    brokerage APIs, credentials. Execution stays in *your* sandbox,
#    so your API keys and data never leave your environment.
# 2. **Control.** You can validate, log, rate-limit, or override any
#    tool call before executing it. In finance, where a malformed
#    query could be expensive, this is critical.
# 3. **Generality.** OpenAI can't anticipate every tool you might need.
#    By defining a protocol (JSON schema in, JSON result back), *any*
#    local function becomes available without OpenAI needing to host it.
#
# ### The tool-calling loop
#
# The interaction follows a loop. The model and your code take turns
# until the model has enough information to answer in plain text:
#
# ```
# ┌─────────────────────────────────────────────────────────┐
# │                     YOUR CODE                           │
# │                                                         │
# │  1. Send messages + tool definitions ──────────►  API   │
# │                                                         │
# │  2. API returns tool_calls             ◄──────── API    │
# │     (model says: "please call                           │
# │      calculate('15 * 850 / 100')")                      │
# │                                                         │
# │  3. Execute locally, append result ────────────► API    │
# │     (you run the function, send                         │
# │      back "127.5")                                      │
# │                                                         │
# │  4. Repeat until model returns ────────◄──────── API    │
# │     plain text (no more tool_calls)                     │
# └─────────────────────────────────────────────────────────┘
# ```
#
# Each round trip adds messages to the conversation: the model's
# tool-call request (an assistant message) and your tool results (tool
# messages). The model sees the full history each time, so it can
# reason across multiple rounds.
#
# ### What the model actually returns
#
# When the model wants to use a tool, its response message has:
# - **`content = None`** — it is *not* answering the user yet
# - **`tool_calls = [...]`** — a list of function calls it wants you
#   to execute, each with a name, JSON arguments, and a unique ID
#
# The arguments are **inert JSON data**, not executable code. This keeps
# the attack surface minimal — your dispatcher maps the JSON to a
# real function call.
#
# Let's see this in action.
#
# ### 6a. Simple Tools
#
# We start with two basic tools: `get_current_time` (no arguments) and
# `calculate` (takes a math expression). We define each tool's schema
# as a JSON object following the OpenAI tool format, then write the
# corresponding Python functions that will actually execute when called.

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

# %% [markdown]
# ### Step-by-step walkthrough
#
# Before running the full loop, let's walk through **one round trip**
# manually so we can inspect every object along the way.

# %% [markdown]
# **Step 1 — Send the user's question along with tool definitions.**
#
# We give the model two things: the conversation so far, and a list of
# tools it is allowed to call. The model decides whether to answer
# directly or request one or more tool calls.

# %%
user_question = "What time is it, and what is 15% of 850?"

messages = [
    {
        "role": "system",
        "content": "You are a helpful assistant. Use the available tools to answer questions accurately.",
    },
    {"role": "user", "content": user_question},
]

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=messages,
    tools=SIMPLE_TOOLS,
)

# What did the model send back?
message = response.choices[0].message
message

# %% [markdown]
# Look at two fields on that message:
# - **`content`** is `None` — the model did *not* answer our question.
# - **`tool_calls`** is a list — the model is **asking us to execute
#   functions on its behalf**. It cannot run code; it can only say
#   "please call this function with these arguments."

# %%
print(f"content:    {message.content!r}")
print(f"tool_calls: {message.tool_calls}")

# %% [markdown]
# Let's unpack each tool call. Every entry has three fields:
# - **`id`** — a unique identifier we'll use to link the result back
# - **`function.name`** — which function the model wants called
# - **`function.arguments`** — a JSON string with the arguments

# %%
# Each tool_call has an ID, a function name, and a JSON argument string.
for tc in message.tool_calls:
    print(f"ID:        {tc.id}")
    print(f"Function:  {tc.function.name}")
    print(f"Arguments: {tc.function.arguments}")
    print()

# %% [markdown]
# **Step 2 — Execute the requested tools and report results back.**
#
# We must:
# 1. Append the assistant's message (the one containing `tool_calls`)
#    to the conversation history.
# 2. Run each tool locally and append a `"role": "tool"` message whose
#    `tool_call_id` links it to the original request.

# %%
# Append the assistant's tool-calling message
messages.append(message)

# Execute each requested tool and append results
for tc in message.tool_calls:
    name = tc.function.name
    args = json.loads(tc.function.arguments)
    result = execute_simple_tool(name, args)
    print(f"{name}({args}) → {result}")
    messages.append(
        {"role": "tool", "tool_call_id": tc.id, "content": result}
    )

# %% [markdown]
# The conversation now looks like this — we started with 2 messages
# (system + user), and after one round of tool calling we have 2 more
# (the assistant's tool request + our tool results):

# %%
for i, m in enumerate(messages):
    role = m["role"] if isinstance(m, dict) else m.role
    if isinstance(m, dict):
        content = m.get("content", "")
    else:
        content = m.content or "(tool_calls — see above)"
    print(f"[{i}] {role:10s} {content[:80]}")

# %% [markdown]
# **Step 3 — Send the updated conversation back to the model.**
#
# The model now sees the tool outputs and can compose a final answer.

# %%
response2 = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=messages,
    tools=SIMPLE_TOOLS,
)

message2 = response2.choices[0].message
print(f"tool_calls: {message2.tool_calls}")
print(f"content:    {message2.content}")

# %% [markdown]
# This time `tool_calls` is `None` and `content` has the answer. The
# model used both tool results to write a human-readable response.
#
# **Why a loop?** The model might need *multiple* rounds of tool calls
# before it has enough information. For instance, a tool result might
# prompt the model to call *another* tool. The loop keeps going until the
# model returns `content` instead of `tool_calls` (or we hit a safety
# limit on iterations).

# %% [markdown]
# ### Full tool-calling loop
#
# Here is the same logic packaged as a reusable loop:

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
# ### 6b. Finance Tools — Multi-Tool Orchestration
#
# The simple example above used one or two tools in a single round.
# In practice, financial analysis often requires **chaining** several
# data lookups and computations to answer a single question. This is
# where tool use becomes powerful: the model acts as an **orchestrator**
# that breaks a complex question into a sequence of tool calls.
#
# Consider the question: *"Compare AAPL and NVDA — which has a better
# P/E ratio? Also, if I bought AAPL at \$150, what's my return?"*
#
# To answer, the model needs to:
# 1. Look up stock info for both AAPL and NVDA (to get P/E ratios)
# 2. Look up the current price of AAPL (to compute a return)
# 3. Calculate the percentage return from \$150 to the current price
# 4. Compose a response using all three results
#
# No single tool can answer this. The model must decide which tools to
# call, in what order, and how to combine the results — exactly the
# kind of multi-step reasoning that makes tool use different from a
# simple API wrapper.
#
# ```
#  User: "Compare AAPL and NVDA P/E; also my AAPL return from $150"
#    │
#    ▼
#  ┌──────────────────────────────────────────────────────┐
#  │  Model (iteration 1):                                │
#  │    tool_calls:                                       │
#  │      get_stock_info("AAPL")                          │
#  │      get_stock_info("NVDA")                          │
#  │      get_stock_price("AAPL")                         │
#  └──────────────────────────────────────────────────────┘
#    │ execute all three locally, return results
#    ▼
#  ┌──────────────────────────────────────────────────────┐
#  │  Model (iteration 2):                                │
#  │    tool_calls:                                       │
#  │      calculate_return(start=150, end=178.50)         │
#  └──────────────────────────────────────────────────────┘
#    │ execute, return result
#    ▼
#  ┌──────────────────────────────────────────────────────┐
#  │  Model (iteration 3):                                │
#  │    content: "AAPL has a P/E of 28.5 vs NVDA's 65.8, │
#  │    so AAPL is cheaper by this metric. Your return    │
#  │    from $150 is +19.0%."                             │
#  └──────────────────────────────────────────────────────┘
# ```
#
# We simulate this with three tools backed by a local dictionary of
# stock data. In production you would swap in real API calls (e.g.,
# `yfinance`, Bloomberg, WRDS) — the loop structure stays the same.
#
# **Tools we define:**
#
# | Tool | Purpose | Arguments |
# |------|---------|-----------|
# | `get_stock_price` | Current price for a ticker | `ticker` |
# | `get_stock_info` | Company name, P/E, market cap | `ticker` |
# | `calculate_return` | Percentage return between two prices | `start_price`, `end_price` |

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

# %% [markdown]
# Now we run the same loop from Section 6a, but with the finance tools.
# Watch the output: the model will make multiple rounds of tool calls,
# gathering data before composing its final answer.

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
# ## 8. Vision & OCR
#
# Vision-capable models can read images: charts, tables, scanned
# documents, screenshots. In finance, this is useful for extracting data
# from earnings PDFs, reading chart patterns, or OCR-ing historical
# records.
#
# The API accepts images as **base64-encoded data URLs** inside a
# multi-part `content` array (text + image together).

# %% [markdown]
# ### 8a. Describing a Chart
#
# We generate a simple stock-price chart with matplotlib, encode it as
# base64, and ask the model to describe what it sees.

# %%
import matplotlib.pyplot as plt
import numpy as np

# Generate a sample stock chart
np.random.seed(42)
days = np.arange(100)
prices = 100 + np.cumsum(np.random.randn(100) * 2)

fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(days, prices, "b-", linewidth=2)
ax.fill_between(days, prices, alpha=0.3)
ax.set_title("ACME Corp Stock Price (2024)", fontsize=14)
ax.set_xlabel("Trading Days")
ax.set_ylabel("Price ($)")
ax.grid(True, alpha=0.3)

max_idx, min_idx = int(np.argmax(prices)), int(np.argmin(prices))
ax.annotate(f"High: ${prices[max_idx]:.2f}", xy=(max_idx, prices[max_idx]),
            xytext=(max_idx + 5, prices[max_idx] + 5), fontsize=10)
ax.annotate(f"Low: ${prices[min_idx]:.2f}", xy=(min_idx, prices[min_idx]),
            xytext=(min_idx + 5, prices[min_idx] - 5), fontsize=10)

# Save to bytes and encode as base64
buf = io.BytesIO()
plt.savefig(buf, format="png", dpi=100, bbox_inches="tight")
plt.close()
buf.seek(0)
chart_b64 = base64.b64encode(buf.read()).decode("utf-8")

print(f"Image size: {len(chart_b64)} chars (base64)")

# %% [markdown]
# The key part: images go inside the `content` array alongside text.
# Each element has a `type` — either `"text"` or `"image_url"`.

# %%
messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": (
                    "Describe this stock chart. What trends do you see? "
                    "Is the stock trending up or down overall?"
                ),
            },
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{chart_b64}"},
            },
        ],
    }
]

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=messages,
    max_tokens=500,
)

print(response.choices[0].message.content)

# %% [markdown]
# ### 8b. Extracting Structured Data from a Table Image
#
# This combines **vision** with **structured outputs** (Section 5). We
# render a financial table as an image, then ask the model to extract
# typed data — demonstrating OCR that goes straight to a Pydantic object.

# %%
# Generate a table image
fig, ax = plt.subplots(figsize=(10, 4))
ax.axis("off")

headers = ["Company", "Ticker", "Revenue ($B)", "YoY Growth", "P/E Ratio"]
data = [
    ["Apple Inc.", "AAPL", "383.3", "+2.8%", "28.5"],
    ["Microsoft", "MSFT", "211.9", "+15.6%", "35.2"],
    ["Alphabet", "GOOGL", "307.4", "+8.7%", "24.1"],
    ["Amazon", "AMZN", "574.8", "+11.8%", "62.3"],
    ["NVIDIA", "NVDA", "60.9", "+125.9%", "65.8"],
]

table = ax.table(
    cellText=data,
    colLabels=headers,
    cellLoc="center",
    loc="center",
    colColours=["#4472C4"] * len(headers),
)
table.auto_set_font_size(False)
table.set_fontsize(11)
table.scale(1.2, 1.8)

for j in range(len(headers)):
    table[(0, j)].set_text_props(color="white", weight="bold")

ax.set_title("Tech Giants — Q4 2024 Financial Summary", fontsize=14, weight="bold", pad=20)

buf = io.BytesIO()
plt.savefig(buf, format="png", dpi=150, bbox_inches="tight", facecolor="white")
plt.close()
buf.seek(0)
table_b64 = base64.b64encode(buf.read()).decode("utf-8")

print(f"Table image: {len(table_b64)} chars (base64)")

# %% [markdown]
# Define Pydantic models for the table structure, then use
# `client.beta.chat.completions.parse` — a convenience method that
# handles the `response_format` wiring and returns a typed `.parsed`
# object directly.

# %%
class FinancialRow(BaseModel):
    """A single row from a financial table."""

    company: str
    ticker: str
    revenue_billions: float
    growth_pct: float
    pe_ratio: float


class FinancialTable(BaseModel):
    """Extracted financial table data."""

    title: str = Field(description="Title of the table")
    rows: list[FinancialRow] = Field(description="Data rows from the table")

# %%
messages = [
    {
        "role": "system",
        "content": "Extract the data from this financial table image into structured format.",
    },
    {
        "role": "user",
        "content": [
            {"type": "text", "text": "Extract all the data from this table."},
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{table_b64}"},
            },
        ],
    },
]

response = client.beta.chat.completions.parse(
    model="gpt-4o-mini",
    messages=messages,
    response_format=FinancialTable,
)

table_data = response.choices[0].message.parsed

print(f"Table Title: {table_data.title}\n")
print(f"{'Company':<15} {'Ticker':<8} {'Revenue':>10} {'Growth':>10} {'P/E':>8}")
print("-" * 55)
for row in table_data.rows:
    print(f"{row.company:<15} {row.ticker:<8} ${row.revenue_billions:>8.1f} {row.growth_pct:>+9.1f}% {row.pe_ratio:>8.1f}")

# %% [markdown]
# ### 8c. Analyzing a Chart with Structured Output
#
# A more complex example: a multi-panel chart (price + volume) analyzed
# into a typed `ChartAnalysis` object.

# %%
# Generate a more complex chart with price + volume panels
np.random.seed(123)

start_date = datetime(2024, 1, 1)
dates = [start_date + timedelta(days=i) for i in range(250) if (start_date + timedelta(days=i)).weekday() < 5][:180]

base_price = 150
returns = np.random.randn(len(dates)) * 0.015 + np.linspace(0, 0.3, len(dates)) / len(dates)
prices = base_price * np.cumprod(1 + returns)
prices[60:90] *= np.linspace(1, 0.85, 30)
prices[90:] *= 0.85

ma20 = np.convolve(prices, np.ones(20) / 20, mode="valid")
ma50 = np.convolve(prices, np.ones(50) / 50, mode="valid")

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), height_ratios=[3, 1], sharex=True)
fig.suptitle("TECH ETF — 2024 YTD Performance", fontsize=14, weight="bold")

ax1.plot(dates, prices, "b-", linewidth=1.5, label="Price")
ax1.plot(dates[19:], ma20, "orange", linewidth=1, label="20-day MA", alpha=0.8)
ax1.plot(dates[49:], ma50, "red", linewidth=1, label="50-day MA", alpha=0.8)
ax1.axhline(y=140, color="green", linestyle="--", alpha=0.5, label="Support")
ax1.axhline(y=175, color="red", linestyle="--", alpha=0.5, label="Resistance")
ax1.set_ylabel("Price ($)")
ax1.legend(loc="upper left")
ax1.grid(True, alpha=0.3)

volumes = np.random.randint(1_000_000, 5_000_000, len(dates))
volumes[60:90] *= 2
colors = ["g"] + ["g" if prices[i] > prices[i - 1] else "r" for i in range(1, len(prices))]
ax2.bar(dates, volumes, color=colors, alpha=0.7, width=0.8)
ax2.set_ylabel("Volume")
ax2.set_xlabel("Date")

import matplotlib.dates as mdates
ax2.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
ax2.xaxis.set_major_locator(mdates.MonthLocator())
plt.tight_layout()

buf = io.BytesIO()
plt.savefig(buf, format="png", dpi=150, bbox_inches="tight", facecolor="white")
plt.close()
buf.seek(0)
etf_b64 = base64.b64encode(buf.read()).decode("utf-8")

print(f"Chart image: {len(etf_b64)} chars (base64)")

# %%
class ChartAnalysis(BaseModel):
    """Structured analysis of a financial chart."""

    chart_type: str = Field(description="Type of chart (line, bar, candlestick, etc.)")
    title: str = Field(description="Title of the chart")
    time_period: str = Field(description="Time period covered by the chart")
    overall_trend: str = Field(description="Overall trend: upward, downward, sideways, or volatile")
    key_observations: list[str] = Field(description="3-5 key observations from the chart")
    support_level: float | None = Field(description="Approximate support level if visible", default=None)
    resistance_level: float | None = Field(description="Approximate resistance level if visible", default=None)

# %%
messages = [
    {
        "role": "system",
        "content": "You are a technical analyst. Analyze the provided chart "
        "and extract key insights about price trends, support/resistance levels, "
        "and notable patterns.",
    },
    {
        "role": "user",
        "content": [
            {"type": "text", "text": "Analyze this financial chart in detail."},
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{etf_b64}"},
            },
        ],
    },
]

response = client.beta.chat.completions.parse(
    model="gpt-4o-mini",
    messages=messages,
    response_format=ChartAnalysis,
)

analysis = response.choices[0].message.parsed

print(f"Chart Type: {analysis.chart_type}")
print(f"Title: {analysis.title}")
print(f"Time Period: {analysis.time_period}")
print(f"Overall Trend: {analysis.overall_trend}")
if analysis.support_level:
    print(f"Support Level: ${analysis.support_level:.2f}")
if analysis.resistance_level:
    print(f"Resistance Level: ${analysis.resistance_level:.2f}")

print("\nKey Observations:")
for i, obs in enumerate(analysis.key_observations, 1):
    print(f"  {i}. {obs}")

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
# | Vision / OCR | Multi-part `content` with `"type": "image_url"` |
# | Vision + structured | Combine image input with `response_format` or `beta.chat.completions.parse` |
