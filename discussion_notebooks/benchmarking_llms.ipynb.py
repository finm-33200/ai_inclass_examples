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
# # Benchmarking LLMs and LLM Agents
#
# This notebook walks through how to measure, evaluate, and benchmark
# large language models — progressing from simple latency measurements
# to standardized academic benchmarks.
#
# **Topics covered:**
# 1. Why benchmarking matters — the benchmark taxonomy
# 2. Latency and cost measurement
# 3. DIY evaluation with assertions
# 4. LLM-as-judge
# 5. Thinking vs non-thinking models
# 6. Tool use boosts performance
# 7. Standardized benchmarks (lm-evaluation-harness)
# 8. Pitfalls and takeaways

# %% [markdown]
# ## 1. Setup

# %%
import json
import re
import time
from pathlib import Path

from decouple import Config, RepositoryEnv
from openai import OpenAI

# Load API keys from .env at the repository root
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
# ## 2. Why Benchmarking Matters
#
# "Which model is best?" has no single answer. The same models rank
# differently on different tasks. Benchmarks give us a structured way
# to compare, but they come with serious caveats.
#
# ### Benchmark Taxonomy
#
# | Tier | Examples | Status |
# |------|----------|--------|
# | **Saturated** | MMLU (88–95%), GSM8K (95%+), HumanEval (90%+) | Top models nearly perfect — no longer differentiates |
# | **Current frontier** | GPQA Diamond, SWE-bench Verified, AIME 2025, Chatbot Arena | Where the action is today |
# | **Unsolved** | Humanity's Last Exam (~44%), FrontierMath (~2%) | Even the best models struggle |
#
# **Key insight:** Benchmarks expire. As models improve, yesterday's hard
# test becomes today's easy baseline. The field constantly needs new
# benchmarks to tell models apart.
#
# ### What qualitative properties matter?
#
# Beyond raw knowledge, two capabilities dramatically change model
# performance on benchmarks:
#
# - **Thinking / reasoning**: Models that can "think step-by-step"
#   (chain-of-thought) crush math and logic benchmarks compared to
#   models that answer immediately.
# - **Tool use**: Giving a model access to a calculator, code
#   interpreter, or data lookup tool can turn a mediocre score into a
#   perfect one.
#
# We will demonstrate both of these live in this notebook.

# %% [markdown]
# ---
# ## 3. Latency and Cost Measurement
#
# The most basic benchmark: how fast is it, and how much does it cost?
#
# Every API response includes a `usage` field with token counts. We
# combine that with published per-million-token prices to compute cost.
# Wrapping the call with `time.time()` gives us wall-clock latency.
#
# Below we define two models to compare (GPT-4o Mini vs GPT-4o), two
# finance prompts, and helper functions to run a prompt and calculate
# its cost.

# %%
# Models to compare (prices per million tokens, from openai.com/api/pricing)
MODELS = [
    {"id": "gpt-4o-mini", "name": "GPT-4o Mini", "input_price": 0.15, "output_price": 0.60},
    {"id": "gpt-4o", "name": "GPT-4o", "input_price": 2.50, "output_price": 10.00},
]

PROMPTS = [
    {"name": "current_yield", "text": "What is the current yield formula for a bond? Explain briefly."},
    {"name": "sharpe_ratio", "text": "A portfolio has annual return of 12%, risk-free rate is 3%, and standard deviation is 15%. Calculate the Sharpe ratio and show your work."},
]


def run_prompt(api_client, model_id, prompt, max_tokens=300):
    """Run a single prompt and capture timing + token metrics."""
    start = time.time()
    response = api_client.chat.completions.create(
        model=model_id,
        messages=[
            {"role": "system", "content": "You are a financial analyst. Be concise."},
            {"role": "user", "content": prompt},
        ],
        max_tokens=max_tokens,
    )
    latency = time.time() - start
    return {
        "output": response.choices[0].message.content,
        "latency_s": round(latency, 3),
        "input_tokens": response.usage.prompt_tokens,
        "output_tokens": response.usage.completion_tokens,
        "total_tokens": response.usage.total_tokens,
    }


def calculate_cost(input_tokens, output_tokens, input_price, output_price):
    """Cost in USD from token counts and per-million-token prices."""
    return round((input_tokens * input_price + output_tokens * output_price) / 1_000_000, 6)

# %% [markdown]
# Now we send each prompt to both models and collect latency, token
# counts, and cost. The results are printed as a comparison table.

# %%
print("=" * 70)
print("LATENCY AND COST BENCHMARK")
print("=" * 70)

results = []
for model in MODELS:
    for prompt_info in PROMPTS:
        metrics = run_prompt(client, model["id"], prompt_info["text"])
        cost = calculate_cost(
            metrics["input_tokens"], metrics["output_tokens"],
            model["input_price"], model["output_price"],
        )
        results.append({**metrics, "model": model["name"], "prompt": prompt_info["name"], "cost": cost})

# Print comparison table
print(f"\n{'Model':<16} {'Prompt':<16} {'Latency':<10} {'Tokens':<8} {'Cost':>10}")
print("-" * 62)
for r in results:
    print(f"{r['model']:<16} {r['prompt']:<16} {r['latency_s']:>6.2f}s   {r['total_tokens']:<8} ${r['cost']:.6f}")

# Cost ratio
costs_by_model = {}
for r in results:
    costs_by_model.setdefault(r["model"], 0)
    costs_by_model[r["model"]] += r["cost"]
model_names = list(costs_by_model.keys())
if len(model_names) == 2:
    ratio = costs_by_model[model_names[1]] / costs_by_model[model_names[0]]
    print(f"\n{model_names[1]} costs {ratio:.1f}x more than {model_names[0]}")

# %% [markdown]
# ---
# ## 4. DIY Evaluation with Assertions
#
# A simple but powerful evaluation pattern: send a prompt to the model,
# then check whether the response contains specific keywords or phrases
# you expect in a correct answer.
#
# For example, any good explanation of current yield should mention
# "coupon" and "price." We can automate this with two assertion types:
#
# - **contains** — a specific string must appear in the response
# - **contains_any** — at least one string from a list must appear
#
# Below we define three finance test cases with their expected
# assertions, plus a helper function that checks each assertion
# against a model's response.

# %%
TEST_CASES = [
    {
        "question": "What is a bond's current yield? Give the formula.",
        "assertions": [
            {"type": "contains", "value": "coupon"},
            {"type": "contains", "value": "price"},
        ],
    },
    {
        "question": "What is duration in bond investing?",
        "assertions": [
            {"type": "contains_any", "values": ["interest rate", "sensitivity", "price change"]},
        ],
    },
    {
        "question": "What is the Sharpe ratio and how is it interpreted?",
        "assertions": [
            {"type": "contains", "value": "risk-free"},
            {"type": "contains_any", "values": ["standard deviation", "volatility"]},
        ],
    },
]


def check_assertion(response_text, assertion):
    """Check a single assertion against a response. Returns (pass, detail)."""
    text_lower = response_text.lower()
    if assertion["type"] == "contains":
        found = assertion["value"].lower() in text_lower
        return found, f"contains '{assertion['value']}'"
    elif assertion["type"] == "contains_any":
        for v in assertion["values"]:
            if v.lower() in text_lower:
                return True, f"contains '{v}'"
        return False, f"missing all of {assertion['values']}"
    return False, "unknown assertion type"

# %% [markdown]
# Now we run each test case against two models and report pass/fail
# for every question-model pair.

# %%
eval_models = ["gpt-4o-mini", "gpt-4o"]

print("=" * 70)
print("ASSERTION-BASED EVALUATION")
print("=" * 70)

for tc in TEST_CASES:
    print(f"\nQ: {tc['question'][:60]}...")
    for model_id in eval_models:
        metrics = run_prompt(client, model_id, tc["question"])
        passes = 0
        for assertion in tc["assertions"]:
            passed, detail = check_assertion(metrics["output"], assertion)
            passes += int(passed)
        total = len(tc["assertions"])
        status = "PASS" if passes == total else "FAIL"
        print(f"  {model_id:<16} {status}  ({passes}/{total} assertions)")

# %% [markdown]
# ---
# ## 5. LLM-as-Judge
#
# Instead of checking for keywords, we can use a stronger model to
# **grade** a weaker model's response. This is the LLM-as-judge
# pattern:
#
# ```
# Question → Model Under Test → Response → Judge Model → Score (1-5)
# ```
#
# This is more flexible than keyword assertions — the judge can
# evaluate nuance, correctness of reasoning, and completeness.
#
# We define two rubrics (accuracy and completeness) and a
# `judge_response` function that sends the model's answer to the
# judge along with the grading criteria, requesting a 1–5 score in
# JSON format.

# %%
RUBRICS = {
    "accuracy": """
    Evaluate the ACCURACY of this financial explanation.
    - Are the formulas and calculations correct?
    - Are the concepts explained without errors?
    - Would a finance professional agree with this answer?
    """,
    "completeness": """
    Evaluate the COMPLETENESS of this response.
    - Does it answer all parts of the question?
    - Are key concepts and caveats mentioned?
    - Is anything important missing?
    """,
}


def judge_response(api_client, question, response, rubric, judge_model="gpt-4o"):
    """Use an LLM to grade another LLM's response. Returns {score, reasoning}."""
    prompt = f"""You are an expert financial analyst grading an AI assistant's response.

QUESTION ASKED:
{question}

AI RESPONSE TO GRADE:
{response}

GRADING CRITERIA:
{rubric}

Grade the response on a scale of 1-5:
- 1: Completely incorrect or missing
- 2: Major errors or significant gaps
- 3: Partially correct, some issues
- 4: Mostly correct, minor issues
- 5: Excellent, accurate and complete

Respond in JSON format:
{{"score": <1-5>, "reasoning": "<brief explanation>"}}
"""
    result = api_client.chat.completions.create(
        model=judge_model,
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
        max_tokens=200,
    )
    return json.loads(result.choices[0].message.content)

# %% [markdown]
# Let's put it to work. We ask gpt-4o-mini two finance questions, then
# have gpt-4o judge each response on accuracy and completeness.

# %%
judge_questions = [
    "What is the current yield formula for a bond?",
    "Should I invest all my savings in a single stock?",
]

model_under_test = "gpt-4o-mini"
judge_model = "gpt-4o"

print("=" * 70)
print(f"LLM-AS-JUDGE — Testing: {model_under_test}, Judge: {judge_model}")
print("=" * 70)

for question in judge_questions:
    print(f"\nQ: {question}")
    metrics = run_prompt(client, model_under_test, question)
    print(f"  Response: {metrics['output'][:80]}...")

    for rubric_name, rubric_text in RUBRICS.items():
        judgment = judge_response(client, question, metrics["output"], rubric_text)
        print(f"  {rubric_name}: {judgment['score']}/5 — {judgment['reasoning'][:60]}...")

# %% [markdown]
# **Trade-offs of LLM-as-judge:**
#
# | Consideration | Detail |
# |---------------|--------|
# | **Cost** | Each judgment is an additional API call (adds up with many rubrics) |
# | **Consistency** | LLM judges have ~0.5–1 point variance on a 5-point scale |
# | **Human correlation** | ~0.7–0.9 for factual accuracy; lower for creativity |
# | **Best for** | Domain-specific evaluation where keyword checks are too brittle |

# %% [markdown]
# ---
# ## 6. Thinking vs Non-Thinking Models
#
# One of the most dramatic qualitative differences between models is
# whether they can "think" — i.e., perform multi-step reasoning before
# answering.
#
# We compare two models from the same family:
# - **DeepSeek Chat** (standard, non-thinking)
# - **DeepSeek R1** (thinking model — shows its reasoning in `<think>` tags)
#
# Both are accessed cheaply via OpenRouter. We test them on GSM8K-style
# math word problems where the answer is a single number we can verify.
#
# First we define five problems with known numeric answers. We also
# write a helper that extracts the last number from a model's response
# so we can score it automatically.

# %%
# GSM8K-style math word problems with known answers
MATH_PROBLEMS = [
    {
        "question": "Janet's ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?",
        "answer": 18,
    },
    {
        "question": "A trader buys 120 shares at $45 each and pays a $9.99 commission. Later she sells all shares at $52 each and pays another $9.99 commission. What is her total profit in dollars?",
        "answer": 820.02,
    },
    {
        "question": "A bond has a face value of $1000 and pays a 6% annual coupon. If the bond is currently trading at $950, what is the current yield as a percentage, rounded to two decimal places?",
        "answer": 6.32,
    },
    {
        "question": "A portfolio manager allocates $500,000 across 3 funds. Fund A gets 40%, Fund B gets 35%, and Fund C gets 25%. If Fund A returns 8%, Fund B returns 12%, and Fund C returns -3%, what is the portfolio's total dollar return?",
        "answer": 33250,
    },
    {
        "question": "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?",
        "answer": 72,
    },
]


def extract_number(text):
    """Extract the last number from a model response (handles $, commas, %)."""
    # Remove commas from numbers like 33,250
    cleaned = text.replace(",", "")
    # Find all numbers (including decimals and negatives)
    numbers = re.findall(r"-?\d+\.?\d*", cleaned)
    if numbers:
        return float(numbers[-1])
    return None

# %% [markdown]
# Now we run the same five problems through both models. For the
# thinking model, we extract the answer from after the `</think>` tag
# (where the final answer lives). Each answer is compared to the known
# ground truth.

# %%
thinking_models = [
    {"id": "deepseek/deepseek-chat", "name": "DeepSeek Chat (standard)"},
    {"id": "deepseek/deepseek-r1", "name": "DeepSeek R1 (thinking)"},
]

print("=" * 70)
print("THINKING vs NON-THINKING MODEL COMPARISON")
print("=" * 70)
print(f"Problems: {len(MATH_PROBLEMS)} GSM8K-style math questions\n")

model_results = {}
think_trace_example = None

for model in thinking_models:
    print(f"--- {model['name']} ---")
    correct = 0
    model_answers = []

    for i, problem in enumerate(MATH_PROBLEMS):
        response = openrouter_client.chat.completions.create(
            model=model["id"],
            messages=[
                {"role": "system", "content": "Solve this math problem. Show your work, then give the final numeric answer."},
                {"role": "user", "content": problem["question"]},
            ],
            max_tokens=1024,
        )
        raw_text = response.choices[0].message.content or ""

        # Save one thinking trace for display
        if think_trace_example is None and "<think>" in raw_text:
            think_trace_example = raw_text

        # For thinking models, extract answer from after </think> if present
        answer_text = raw_text.split("</think>")[-1] if "</think>" in raw_text else raw_text
        extracted = extract_number(answer_text)
        expected = problem["answer"]
        is_correct = extracted is not None and abs(extracted - expected) < 0.1
        correct += int(is_correct)
        mark = "CORRECT" if is_correct else "WRONG"
        print(f"  Q{i+1}: expected={expected}, got={extracted} — {mark}")
        model_answers.append({"expected": expected, "got": extracted, "correct": is_correct})

    accuracy = correct / len(MATH_PROBLEMS) * 100
    model_results[model["name"]] = {"accuracy": accuracy, "correct": correct, "answers": model_answers}
    print(f"  Score: {correct}/{len(MATH_PROBLEMS)} ({accuracy:.0f}%)\n")

# Summary
print("=" * 70)
print("SUMMARY")
print(f"{'Model':<35} {'Accuracy':>10}")
print("-" * 47)
for name, res in model_results.items():
    print(f"{name:<35} {res['accuracy']:>8.0f}%")

# %% [markdown]
# The thinking model wraps its step-by-step reasoning inside
# `<think>...</think>` tags before giving the final answer. Here is a
# sample of that internal reasoning trace.

# %%
# Show a sample of the thinking trace
if think_trace_example:
    # Extract just the thinking portion
    think_match = re.search(r"<think>(.*?)</think>", think_trace_example, re.DOTALL)
    if think_match:
        trace = think_match.group(1).strip()
        # Show first 600 chars of reasoning
        preview = trace[:600] + ("..." if len(trace) > 600 else "")
        print("Sample thinking trace from DeepSeek R1:")
        print("-" * 50)
        print(preview)

# %% [markdown]
# **Why thinking models win at math:**
#
# Standard models generate answers token-by-token without "scratch
# space." Thinking models (o3-mini, DeepSeek-R1, QwQ) allocate extra
# computation to reason step-by-step before committing to an answer.
# On multi-step math problems, this is the difference between guessing
# and solving.

# %% [markdown]
# ---
# ## 7. Tool Use Boosts Performance
#
# Even a small model can ace hard arithmetic if you give it a
# calculator. Here we test gpt-4o-mini on problems with large numbers
# — first without tools, then with a `calculator` function tool.
#
# This demonstrates why "model + tools" benchmarks can look very
# different from "model alone" benchmarks.
#
# We define five arithmetic problems with large numbers (where mental
# math is unreliable), a calculator tool the model can call, and a
# `safe_eval` function that executes the expression locally.

# %%
ARITHMETIC_PROBLEMS = [
    {"question": "What is 1247 * 863?", "answer": 1076161},
    {"question": "What is 98713 / 247?", "answer": round(98713 / 247, 2)},
    {"question": "What is 4567 * 891 + 3210?", "answer": 4567 * 891 + 3210},
    {"question": "What is 15% of 8943, rounded to 2 decimal places?", "answer": round(8943 * 0.15, 2)},
    {"question": "What is (12500 * 1.08^5) rounded to 2 decimal places?", "answer": round(12500 * 1.08**5, 2)},
]

# Tool definition
CALCULATOR_TOOL = [
    {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": "Evaluate a mathematical expression and return the numeric result",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "Math expression to evaluate (e.g., '1247 * 863', '12500 * 1.08**5')",
                    }
                },
                "required": ["expression"],
            },
        },
    }
]


def safe_eval(expression):
    """Evaluate a math expression safely (only allows numbers and operators)."""
    allowed = set("0123456789+-*/.(). %e")
    cleaned = expression.replace("^", "**")
    if not all(c in allowed for c in cleaned):
        return f"Error: invalid characters"
    try:
        return str(eval(cleaned))
    except Exception as e:
        return f"Error: {e}"

# %% [markdown]
# We run gpt-4o-mini on the same problems twice — once without tools
# (the model must do the math in its head) and once with the calculator
# tool available. When the model calls the tool, we execute the
# expression and feed the result back, then let it give a final answer.

# %%
tool_use_model = "gpt-4o-mini"

print("=" * 70)
print(f"TOOL USE COMPARISON — {tool_use_model}")
print("=" * 70)

# --- Without tools ---
print("\n--- Without calculator tool ---")
no_tool_correct = 0
for i, prob in enumerate(ARITHMETIC_PROBLEMS):
    response = client.chat.completions.create(
        model=tool_use_model,
        messages=[
            {"role": "system", "content": "Solve this math problem. Give only the numeric answer."},
            {"role": "user", "content": prob["question"]},
        ],
        max_tokens=100,
    )
    raw = response.choices[0].message.content
    extracted = extract_number(raw)
    expected = prob["answer"]
    is_correct = extracted is not None and abs(extracted - expected) < 0.1
    no_tool_correct += int(is_correct)
    mark = "CORRECT" if is_correct else "WRONG"
    print(f"  Q{i+1}: expected={expected}, got={extracted} — {mark}")

print(f"  Score: {no_tool_correct}/{len(ARITHMETIC_PROBLEMS)}")

# --- With calculator tool ---
print("\n--- With calculator tool ---")
tool_correct = 0
for i, prob in enumerate(ARITHMETIC_PROBLEMS):
    messages = [
        {"role": "system", "content": "Solve this math problem. Use the calculator tool for computation. Give the numeric answer."},
        {"role": "user", "content": prob["question"]},
    ]

    # Tool-call loop (max 3 iterations)
    final_text = None
    for _ in range(3):
        response = client.chat.completions.create(
            model=tool_use_model,
            messages=messages,
            tools=CALCULATOR_TOOL,
        )
        msg = response.choices[0].message

        if msg.tool_calls:
            messages.append(msg)
            for tool_call in msg.tool_calls:
                args = json.loads(tool_call.function.arguments)
                result = safe_eval(args["expression"])
                messages.append({"role": "tool", "tool_call_id": tool_call.id, "content": result})
        else:
            final_text = msg.content
            break

    extracted = extract_number(final_text) if final_text else None
    expected = prob["answer"]
    is_correct = extracted is not None and abs(extracted - expected) < 0.1
    tool_correct += int(is_correct)
    mark = "CORRECT" if is_correct else "WRONG"
    print(f"  Q{i+1}: expected={expected}, got={extracted} — {mark}")

print(f"  Score: {tool_correct}/{len(ARITHMETIC_PROBLEMS)}")

# Summary
print(f"\n{'='*70}")
print("SUMMARY")
print(f"  Without tools: {no_tool_correct}/{len(ARITHMETIC_PROBLEMS)} ({no_tool_correct/len(ARITHMETIC_PROBLEMS)*100:.0f}%)")
print(f"  With tools:    {tool_correct}/{len(ARITHMETIC_PROBLEMS)} ({tool_correct/len(ARITHMETIC_PROBLEMS)*100:.0f}%)")

# %% [markdown]
# **Implications for benchmarking:**
#
# - A model's "raw" score on a math benchmark can be very different
#   from its score when augmented with tools.
# - When evaluating agents (model + tools), you are benchmarking the
#   **system**, not just the model.
# - This is why agent benchmarks (SWE-bench, WebArena) test the full
#   loop: reasoning, tool selection, and execution.

# %% [markdown]
# ---
# ## 8. Standardized Benchmarks with lm-evaluation-harness
#
# [EleutherAI's lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness)
# is the industry standard for LLM benchmarking. It powers the
# Hugging Face Open LLM Leaderboard and supports 60+ built-in tasks.
#
# ### Key benchmarks
#
# | Benchmark | What it tests | Format | Finance relevance |
# |-----------|---------------|--------|-------------------|
# | **GSM8K** | Math word problems | Free-form numeric answer | Same structure as quant interview Qs |
# | **MMLU** | 57-subject knowledge | Multiple choice | Includes economics, accounting |
# | **TruthfulQA** | Resistance to common misconceptions | Generation | Avoiding hallucinations |
# | **HumanEval** | Code generation | Python functions | Automating data pipelines |
# | **GPQA Diamond** | PhD-level science | Multiple choice | Deep technical reasoning |
#
# ### How it works
#
# ```bash
# # Install
# pip install lm-eval
#
# # Run GSM8K on a model via OpenRouter (5 examples, fast demo)
# lm_eval --model openai-chat-completions \
#     --model_args model=meta-llama/llama-3.1-8b-instruct,\
#                   base_url=https://openrouter.ai/api/v1/chat/completions,\
#                   num_concurrent=50,timeout=120 \
#     --tasks gsm8k \
#     --limit 5 \
#     --apply_chat_template \
#     --output_path results/
# ```
#
# ### API limitations
#
# Standard MMLU tasks require **logprob** scoring (comparing token
# probabilities for A/B/C/D). API endpoints don't return logprobs, so
# we use the **chain-of-thought variants** (`mmlu_cot_llama_*`) that
# work with any chat completion API.
#
# Below we load pre-computed results that were previously generated
# by running `lm_eval` on three open-source models (Llama 3.1 8B,
# Gemma 2 9B, Qwen 2.5 7B) across GSM8K, MMLU, and TruthfulQA. We
# parse the JSON output and display scores side-by-side.

# %%
# Load pre-computed results from benchmarks/04_multi_benchmark/

benchmark_results_dir = _repo_root / "benchmarks" / "04_multi_benchmark" / "results"
model_dirs = {"llama-3.1-8b": "Llama 3.1 8B", "gemma-2-9b": "Gemma 2 9B", "qwen-2.5-7b": "Qwen 2.5 7B"}

print("=" * 70)
print("PRE-COMPUTED MULTI-BENCHMARK RESULTS (lm-evaluation-harness)")
print("=" * 70)
print("(Run with --limit 5 for cost efficiency — small sample, high variance)\n")

for dir_name, display_name in model_dirs.items():
    model_path = benchmark_results_dir / dir_name
    if not model_path.exists():
        print(f"  {display_name}: results not found (run benchmarks/04_multi_benchmark/ first)")
        continue

    # Find the most recent results file
    json_files = sorted(model_path.glob("**/results*.json"))
    if not json_files:
        continue
    latest = json_files[-1]

    with open(latest) as f:
        data = json.load(f)

    results = data.get("results", {})
    scores = {}

    # GSM8K
    if "gsm8k" in results:
        gsm = results["gsm8k"]
        for key in ["exact_match,flexible-extract", "exact_match,strict-match"]:
            if key in gsm:
                scores["GSM8K"] = gsm[key]
                break

    # MMLU (econometrics)
    for task_name in results:
        if "mmlu" in task_name:
            mmlu = results[task_name]
            for key in ["exact_match,strict_match", "acc,none"]:
                if key in mmlu:
                    scores["MMLU (Econ)"] = mmlu[key]
                    break

    # TruthfulQA
    if "truthfulqa_gen" in results:
        tqa = results["truthfulqa_gen"]
        for key in ["rougeL_acc,none", "bleu_acc,none"]:
            if key in tqa:
                scores["TruthfulQA"] = tqa[key]
                break

    print(f"  {display_name}:")
    for bench, score in scores.items():
        print(f"    {bench:<20} {score*100:>6.1f}%")
    print()

# %% [markdown]
# > **Note:** These results use only 5 examples per benchmark, so
# > standard errors are large. In practice, you would run the full
# > test sets (hundreds to thousands of examples) for reliable scores.

# %% [markdown]
# ---
# ## 9. Pitfalls and Takeaways
#
# ### Common pitfalls
#
# | Pitfall | What happens | Example |
# |---------|--------------|---------|
# | **Data contamination** | Model memorizes test questions from training data | GSM8K questions found verbatim in training corpora |
# | **Benchmark overfitting** | Models tuned to pass specific benchmarks but fail on rephrased versions | Study of 51 LLMs: 39.4% avg performance drop on rephrased benchmarks |
# | **Saturation** | All top models score ~same, benchmark stops differentiating | MMLU at 88–95% for frontier models |
# | **Gaming** | Agents exploit evaluation setup instead of solving the task | SWE-bench agents inspecting `.git` history to find human patches |
#
# ### Finance-specific benchmarks
#
# - **FinanceBench**: 10,000+ questions from SEC filings and earnings reports
# - **FinBen**: 42 datasets across 24 financial tasks (forecasting, risk, sentiment)
# - Key finding: LLMs excel at text extraction but struggle with multi-step quantitative reasoning
#
# ### Takeaways
#
# 1. **No single benchmark is sufficient.** Always evaluate on multiple dimensions.
# 2. **Tailor evaluations to your use case.** A generic leaderboard score won't tell you
#    if a model works for *your* task.
# 3. **Qualitative capabilities matter.** Thinking and tool use can be more important
#    than raw model size.
# 4. **Build custom evals.** For production, the LLM-as-judge pattern with
#    domain-specific rubrics is often more useful than standardized benchmarks.
#
# | Use case | Prioritize |
# |----------|-----------|
# | Financial calculations | GSM8K, tool-use eval |
# | Research & analysis | MMLU + TruthfulQA |
# | Code generation | HumanEval, SWE-bench |
# | General assistant | Chatbot Arena Elo |
# | Hallucination risk | TruthfulQA, custom fact-checking |
