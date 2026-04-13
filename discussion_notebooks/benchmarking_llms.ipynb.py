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
# # Benchmarking LLMs
#
# This notebook walks through how to measure, evaluate, and benchmark
# large language models — progressing from simple latency measurements
# to standardized academic benchmarks.
#
# **Topics covered:**
# 1. Why benchmarking matters — the benchmark taxonomy
# 2. DIY evaluation with assertions
# 3. LLM-as-judge
# 4. Thinking vs non-thinking models
# 5. Tool use boosts performance
# 6. Standardized benchmarks (lm-evaluation-harness)
# 7. Pitfalls and takeaways

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

# %%
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

# %% [markdown]
# ---
# ## 3. DIY Evaluation with Assertions
#
# A better pattern than keyword-checking: give the model problems with
# **known numeric answers**, require a specific output format, then use
# regex to extract and verify the answer.
#
# This tests two things at once:
# 1. **Format compliance** — did the model follow instructions?
# 2. **Correctness** — is the extracted number right?

# %%
MATH_TEST_CASES = [
    {
        "question": "A bond has a face value of $1,000 and pays a 5% annual coupon. It currently trades at $980. What is the current yield? Give your answer as a percentage rounded to two decimal places.",
        "answer": 5.10,
    },
    {
        "question": "A portfolio is 60% stocks and 40% bonds. Stocks returned 12% and bonds returned -2%. What is the portfolio return as a percentage?",
        "answer": 6.4,
    },
    {
        "question": "You invest $10,000 at 7% annual interest compounded annually. What is the value after 3 years, rounded to two decimal places?",
        "answer": 12250.43,
    },
]

FORMAT_INSTRUCTION = "\n\nRespond with EXACTLY this format on the last line:\nANSWER: <number>"


def extract_answer(text):
    """Extract the number after 'ANSWER:' using regex. Returns (number, matched)."""
    match = re.search(r"ANSWER:\s*\$?([\d,]+\.?\d*)", text)
    if match:
        return float(match.group(1).replace(",", "")), True
    return None, False

# %% [markdown]
# Run each problem against two models. We check format compliance
# (did the regex match?) and correctness (within 0.1 of expected).

# %%
eval_models = [
    {"client": openrouter_client, "id": "meta-llama/llama-3.1-8b-instruct", "name": "Llama 3.1 8B"},
    {"client": openrouter_client, "id": "google/gemma-2-9b-it", "name": "Gemma 2 9B"},
    {"client": client, "id": "gpt-4o-mini", "name": "gpt-4o-mini"},
    {"client": client, "id": "gpt-4o", "name": "gpt-4o"},
]

print("=" * 70)
print("DIY MATH EVALUATION")
print("=" * 70)

for tc in MATH_TEST_CASES:
    print(f"\nQ: {tc['question'][:65]}...")
    for model in eval_models:
        metrics = run_prompt(model["client"], model["id"], tc["question"] + FORMAT_INSTRUCTION)
        extracted, format_ok = extract_answer(metrics["output"])
        correct = extracted is not None and abs(extracted - tc["answer"]) < 0.1
        fmt_tag = "format:OK" if format_ok else "format:FAIL"
        ans_tag = "correct" if correct else f"wrong (got {extracted})"
        print(f"  {model['name']:<20} {fmt_tag}, {ans_tag}  [expected {tc['answer']}]")

# %% [markdown]
# ---
# ## 4. LLM-as-Judge
#
# Assertion-based checks work well when the answer is a single number,
# but many questions have open-ended answers where there is no exact
# value to compare against. For those, we can use a stronger model to
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
# Let's put it to work. We ask two models the same finance questions,
# then have gpt-4o judge each response on accuracy and completeness.

# %%
judge_questions = [
    "What is the current yield formula for a bond?",
    "Should I invest all my savings in a single stock?",
]

models_under_test = [
    {"client": openrouter_client, "id": "meta-llama/llama-3.1-8b-instruct", "name": "Llama 3.1 8B"},
    {"client": client, "id": "gpt-4o-mini", "name": "gpt-4o-mini"},
]
judge_model = "gpt-4o"

print("=" * 70)
print(f"LLM-AS-JUDGE — Judge: {judge_model}")
print("=" * 70)

for question in judge_questions:
    print(f"\nQ: {question}")
    for model in models_under_test:
        metrics = run_prompt(model["client"], model["id"], question)
        print(f"\n  [{model['name']}] {metrics['output'][:80]}...")
        for rubric_name, rubric_text in RUBRICS.items():
            judgment = judge_response(client, question, metrics["output"], rubric_text)
            print(f"    {rubric_name}: {judgment['score']}/5 — {judgment['reasoning'][:60]}...")

# %% [markdown]
# ### When is LLM-as-judge useful?
#
# The obvious objection is: **if you need a stronger model to judge a
# weaker one, what do you do when you're evaluating the best model?**
# This is a real limitation, but LLM-as-judge is still broadly useful
# for several reasons:
#
# 1. **Most production systems use cheaper models.** In practice, most
#    deployed applications use smaller, faster, cheaper models
#    (gpt-4o-mini, Llama 8B, etc.) because they're called thousands of
#    times per day. You very much want to evaluate these, and a stronger
#    judge works well here.
#
# 2. **Judging is easier than generating.** A model can often identify
#    flaws in a response it couldn't have produced itself — the same
#    way a film critic can spot a bad movie without being able to direct
#    a good one. Research finds that even models of similar capability
#    can serve as useful judges when the rubric is specific enough
#    (Zheng et al., 2023).
#
# 3. **Pairwise comparison, not absolute grading.** The Chatbot Arena
#    approach (LMSYS) sidesteps the "no stronger judge" problem
#    entirely: show two responses side by side and ask which is better.
#    Humans and LLMs are both much more reliable at comparative
#    judgments than absolute scoring.
#
# 4. **Scaling human evaluation.** Even when human evaluation is the
#    gold standard, you can't run it on every commit. LLM-as-judge
#    gives fast, cheap, per-commit signal — then you validate with
#    humans periodically.
#
# ### When does it break down?
#
# - **Frontier model evaluation.** If you're trying to determine
#   whether GPT-5 is better than GPT-4, no current LLM judge is
#   reliable. This is where human evaluation and Chatbot Arena
#   (crowd-sourced human preferences) remain essential.
# - **Tasks with verifiable answers.** For math, code, and factual
#   recall, the assertion-based approach from Section 3 is strictly
#   better — it's cheaper, deterministic, and perfectly reliable.
#   Use LLM-as-judge for open-ended tasks (explanation quality,
#   writing style, reasoning coherence) where there's no single
#   correct answer.
# - **Known biases.** LLM judges exhibit systematic biases:
#   position bias (preferring whichever response comes first),
#   verbosity bias (preferring longer responses), and self-enhancement
#   bias (a model rates its own outputs higher). These are documented
#   in Zheng et al. (2023) and Stureborg et al. (2024).
#
# ### Connection to RLHF and RLAIF
#
# LLM-as-judge is closely related to how models are trained, not just
# evaluated. In **Reinforcement Learning from Human Feedback (RLHF)**,
# human annotators rank model outputs to train a *reward model* — a
# classifier that predicts which response a human would prefer. The
# LLM is then fine-tuned to maximize this reward signal.
#
# **RLAIF (Reinforcement Learning from AI Feedback)** replaces the
# human annotators with an LLM judge. Google's 2023 RLAIF paper
# (Lee et al.) showed that using an LLM to generate preference labels
# produces models of comparable quality to RLHF at a fraction of the
# cost. Anthropic's **Constitutional AI** is a related approach where
# the model critiques and revises its own outputs against a set of
# principles.
#
# In short: the same "LLM grading LLM" pattern we use here for
# *evaluation* is also used during *training* to align models. The
# evaluation use case is a lightweight, inference-time version of
# what happens at massive scale during RLAIF training.
#
# ### Summary of trade-offs
#
# | Consideration | Detail |
# |---------------|--------|
# | **Best for** | Evaluating cheaper models on open-ended tasks where no ground truth exists |
# | **Not for** | Frontier model evaluation; tasks with verifiable answers |
# | **Cost** | Each judgment is an additional API call (adds up with many rubrics) |
# | **Consistency** | LLM judges have ~0.5–1 point variance on a 5-point scale |
# | **Human correlation** | ~0.7–0.9 for factual accuracy; lower for subjective tasks |
# | **Key references** | Zheng et al. 2023 (MT-Bench), Lee et al. 2023 (RLAIF), Stureborg et al. 2024 (biases) |

# %% [markdown]
# ---
# ## 5. Thinking vs Non-Thinking Models
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
# ## 6. Tool Use Boosts Performance
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
# ## 7. Standardized Benchmarks with lm-evaluation-harness
#
# [EleutherAI's lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness)
# is the industry standard for LLM benchmarking. It powers the
# Hugging Face Open LLM Leaderboard and supports 60+ built-in tasks.
#
# ### Key benchmarks at a glance
#
# | Benchmark | What it tests | Format | Finance relevance |
# |-----------|---------------|--------|-------------------|
# | **GSM8K** | Math word problems | Free-form numeric answer | Same structure as quant interview Qs |
# | **MMLU** | 57-subject knowledge | Multiple choice | Includes economics, accounting |
# | **TruthfulQA** | Resistance to common misconceptions | Generation | Avoiding hallucinations |
# | **HumanEval** | Code generation | Python functions | Automating data pipelines |
# | **GPQA Diamond** | PhD-level science | Multiple choice | Deep technical reasoning |
#
# Below we walk through each benchmark in detail so that the names
# on a leaderboard actually mean something concrete.
#
# ---
#
# ### GSM8K — Grade School Math 8K
#
# **What it is.** A dataset of ~8,500 grade-school-level math word
# problems created by OpenAI (Cobbe et al., 2021). Each problem
# requires 2–8 steps of elementary arithmetic and reasoning — no
# advanced math, but you do have to chain the steps correctly.
#
# **Why it matters.** GSM8K was one of the first benchmarks to
# clearly show that chain-of-thought prompting dramatically
# improves LLM math performance. It is now largely *saturated*
# (frontier models score 95%+), but remains a useful baseline
# for smaller models.
#
# **Example problem:**
#
# > *Natalia sold clips to 48 of her friends in April, and then
# > she sold half as many clips in May. How many clips did Natalia
# > sell altogether in April and May?*
# >
# > **Answer:** 48 + 24 = **72**
#
# **Scoring:** The model's final numeric answer is extracted and
# compared to the ground truth — exact match. No partial credit.
#
# ---
#
# ### MMLU — Massive Multitask Language Understanding
#
# **What it is.** A collection of ~16,000 multiple-choice questions
# spanning 57 academic subjects — from abstract algebra and
# anatomy to US history and professional law (Hendrycks et al.,
# 2021). Each question has four choices (A/B/C/D).
#
# **Why it matters.** MMLU is the most widely cited general
# knowledge benchmark. It was the first to show that scaling model
# size leads to broad knowledge gains across many domains. Today
# frontier models score 88–95%, so it no longer differentiates the
# top tier, but it remains useful for comparing smaller or
# fine-tuned models.
#
# **Example problem (Econometrics):**
#
# > *For a stationary autoregressive process, shocks will:*
# >
# > (A) Eventually die away
# > (B) Persist indefinitely
# > (C) Have no effect on the series
# > (D) Lead to an explosive path
# >
# > **Answer:** (A)
#
# **Example problem (Professional Accounting):**
#
# > *Under the equity method, an investor recognizes its share of
# > the earnings of an investee in the period in which the:*
# >
# > (A) Investee declares a dividend
# > (B) Investor receives the dividend
# > (C) Investee earns the income
# > (D) Investor sells the investment
# >
# > **Answer:** (C)
#
# ---
#
# ### TruthfulQA
#
# **What it is.** A dataset of 817 questions specifically designed
# to elicit common misconceptions, conspiracy theories, and
# popular-but-false claims (Lin et al., 2022). The questions are
# adversarial — they are phrased in ways that *invite* the model
# to repeat a widely-believed falsehood.
#
# **Why it matters.** LLMs learn from the internet, where popular
# misconceptions are everywhere. TruthfulQA measures whether a
# model can resist parroting these errors. Interestingly, the
# original paper found that *larger* models were initially *less*
# truthful — they had memorized more misconceptions. RLHF and
# careful alignment have since reversed this trend.
#
# **Example problems:**
#
# > *"What happens if you crack your knuckles a lot?"*
# >
# > **Common misconception:** It causes arthritis.
# > **Truthful answer:** Nothing harmful; the sound comes from gas
# > bubbles in synovial fluid. Studies find no link to arthritis.
#
# > *"What percentage of the brain do we use?"*
# >
# > **Common misconception:** We only use 10% of our brain.
# > **Truthful answer:** We use virtually all of our brain; different
# > regions are active at different times.
#
# **Scoring:** TruthfulQA uses a fine-tuned "truth judge" model
# to score whether the generation is both truthful and informative.
# This is itself an early form of the LLM-as-judge pattern.
#
# ---
#
# ### HumanEval
#
# **What it is.** A set of 164 Python programming problems created
# by OpenAI (Chen et al., 2021). Each problem gives a function
# signature and docstring; the model must generate the function
# body. The output is scored by running it against hidden unit
# tests — the code either passes or it doesn't.
#
# **Why it matters.** HumanEval was the foundational code
# generation benchmark. It is now largely saturated (frontier
# models score 90%+), but spawned harder successors like
# SWE-bench (which asks models to fix real GitHub issues across
# entire repositories).
#
# **Example problem:**
#
# ```python
# def has_close_elements(numbers: list[float],
#                        threshold: float) -> bool:
#     """Check if in given list of numbers, are any two
#     numbers closer to each other than given threshold.
#     >>> has_close_elements([1.0, 2.0, 3.0], 0.5)
#     False
#     >>> has_close_elements([1.0, 2.8, 3.0, 4.0], 0.3)
#     True
#     """
# ```
#
# **Scoring:** Pass@k — the code is run against unit tests.
# Typically reported as pass@1 (does the first generation pass?).
#
# ---
#
# ### GPQA Diamond — Graduate-level PhD QA
#
# **What it is.** 198 extremely hard multiple-choice questions
# written by PhD domain experts in physics, chemistry, and biology
# (Rein et al., 2023). "Diamond" is the hardest subset, filtered
# so that even *other* PhD holders in adjacent fields struggle.
# Non-expert humans score around 34% (barely above the 25% random
# baseline for 4-choice questions).
#
# **Why it matters.** GPQA Diamond is a current frontier
# benchmark — it differentiates the strongest models. Top
# reasoning models (o3, Claude Opus, Gemini Ultra) score in the
# 60–75% range, while smaller models hover near chance.
#
# **Example problem (Physics, paraphrased):**
#
# > *A quantum system is prepared in state |ψ⟩. After a
# > measurement of observable A yields eigenvalue a₁, a subsequent
# > measurement of observable B is performed. If [A, B] ≠ 0, what
# > can be said about the outcome?*
# >
# > (A) B will yield a definite eigenvalue determined by a₁
# > (B) The outcome of B is generally probabilistic
# > (C) The system collapses to the ground state
# > (D) Measurement of B is impossible after measuring A
# >
# > **Answer:** (B)
#
# ---
#
# ### Aside: How logprob scoring works (and why it matters)
#
# When you see MMLU results on a leaderboard, the model usually
# didn't "answer" the question the way a human would. Instead, the
# evaluation harness uses **log-probability scoring** — a technique
# that looks at the model's internal confidence across the answer
# choices.
#
# **The mechanics.** A language model assigns a probability to every
# possible next token. For a multiple-choice question, the harness:
#
# 1. Feeds the question into the model
# 2. Asks: "What probability does the model assign to token `A`
#    vs. `B` vs. `C` vs. `D` as the next token?"
# 3. Whichever token has the highest probability is the model's
#    "answer"
#
# The model never generates a full response — the harness just
# peeks at the probability distribution over the four choice
# tokens. A "log-probability" is the natural log of the
# probability: if the model assigns 70% to the correct answer,
# its log-prob is ln(0.70) ≈ −0.36; if it assigns only 5%, the
# log-prob is ln(0.05) ≈ −3.0.
#
# **Why not just generate an answer?** Logprob scoring is:
#
# - **Faster**: One forward pass, no generation loop
# - **Deterministic**: No sampling randomness — the same model
#   always gives the same probabilities
# - **More informative**: You can see *how confident* the model
#   is. A model that assigns 90% to the right answer and 10% to
#   the rest "knows" the answer more confidently than one that
#   assigns 30% to the right answer and 25% to a distractor.
#
# **The API problem.** Most commercial API endpoints (OpenAI,
# Anthropic, Google) either don't expose logprobs at all, or
# only return logprobs for tokens the model *did* generate — not
# for arbitrary tokens like A/B/C/D. This means you can't do
# standard logprob-based MMLU evaluation through an API.
#
# **The workaround.** The `lm-evaluation-harness` provides
# **chain-of-thought (CoT) variants** of MMLU (e.g.,
# `mmlu_cot_llama_*`) that work via generation instead:
#
# 1. The model is prompted to reason through the problem
# 2. It generates a full text response ending with its answer
#    choice
# 3. The harness extracts the letter (A/B/C/D) from the
#    generated text and checks it against ground truth
#
# This is compatible with any chat API, but is slower, non-
# deterministic, and loses the confidence information that
# logprob scoring provides. When a paper reports MMLU scores,
# it's worth checking *which* scoring method was used — logprob
# and CoT results are not directly comparable.
#
# ### How to run it
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
# ## 8. Pitfalls and Takeaways
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
