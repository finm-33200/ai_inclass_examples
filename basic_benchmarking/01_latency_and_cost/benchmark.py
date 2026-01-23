"""
01_latency_and_cost/benchmark.py

Manual benchmarking: measure latency, token usage, and cost across models.
This teaches the fundamentals before introducing evaluation frameworks.
"""

import json
import time
from datetime import datetime
from pathlib import Path

from decouple import Config, RepositoryEnv
from openai import OpenAI


# Models to compare (using OpenAI directly)
# Prices from https://openai.com/api/pricing/ (per million tokens)
MODELS = [
    {
        "id": "gpt-4o-mini",
        "name": "GPT-4o Mini",
        "input_price": 0.15,
        "output_price": 0.60,
    },
    {
        "id": "gpt-4o",
        "name": "GPT-4o",
        "input_price": 2.50,
        "output_price": 10.00,
    },
]

# Finance-focused benchmark prompts
PROMPTS = [
    {
        "name": "current_yield",
        "text": "What is the current yield formula for a bond? Explain briefly.",
    },
    {
        "name": "duration_convexity",
        "text": "Explain the difference between duration and convexity in 2-3 sentences.",
    },
    {
        "name": "sharpe_ratio",
        "text": "A portfolio has annual return of 12%, risk-free rate is 3%, and standard deviation is 15%. Calculate the Sharpe ratio and show your work.",
    },
]


def run_prompt(client: OpenAI, model_id: str, prompt: str) -> dict:
    """
    Run a single prompt and capture timing + token metrics.

    Key pattern: wrap the API call with time.time() to measure latency.
    """
    start = time.time()

    response = client.chat.completions.create(
        model=model_id,
        messages=[
            {"role": "system", "content": "You are a financial analyst. Be concise."},
            {"role": "user", "content": prompt},
        ],
        max_tokens=300,
    )

    latency = time.time() - start

    # Extract metrics from API response
    return {
        "output": response.choices[0].message.content,
        "latency_seconds": round(latency, 3),
        "input_tokens": response.usage.prompt_tokens,
        "output_tokens": response.usage.completion_tokens,
        "total_tokens": response.usage.total_tokens,
    }


def calculate_cost(input_tokens: int, output_tokens: int,
                   input_price: float, output_price: float) -> float:
    """
    Calculate cost in USD from token counts and per-million-token prices.

    Formula: cost = (input_tokens * input_price + output_tokens * output_price) / 1,000,000
    """
    input_cost = input_tokens * input_price / 1_000_000
    output_cost = output_tokens * output_price / 1_000_000
    return round(input_cost + output_cost, 6)


def main():
    # Load API key from repo root .env
    repo_root = Path(__file__).resolve().parents[2]
    config = Config(RepositoryEnv(repo_root / ".env"))

    client = OpenAI(api_key=config("OPENAI_API_KEY"))

    print("=" * 60)
    print("LATENCY AND COST BENCHMARK")
    print("=" * 60)
    print(f"Models: {[m['name'] for m in MODELS]}")
    print(f"Prompts: {len(PROMPTS)}")
    print()

    results = []
    timestamp = datetime.now().isoformat()

    for model in MODELS:
        print(f"\n--- {model['name']} ---")

        for prompt_info in PROMPTS:
            print(f"  {prompt_info['name']}...", end=" ", flush=True)

            try:
                metrics = run_prompt(client, model["id"], prompt_info["text"])
                cost = calculate_cost(
                    metrics["input_tokens"],
                    metrics["output_tokens"],
                    model["input_price"],
                    model["output_price"],
                )

                print(f"{metrics['latency_seconds']:.2f}s, "
                      f"{metrics['total_tokens']} tokens, ${cost:.6f}")

                results.append({
                    "timestamp": timestamp,
                    "model_id": model["id"],
                    "model_name": model["name"],
                    "prompt_name": prompt_info["name"],
                    "prompt_text": prompt_info["text"],
                    "cost_usd": cost,
                    **metrics,
                })

            except Exception as e:
                print(f"ERROR: {e}")
                results.append({
                    "timestamp": timestamp,
                    "model_id": model["id"],
                    "model_name": model["name"],
                    "prompt_name": prompt_info["name"],
                    "error": str(e),
                })

    # Save results to JSON
    results_dir = Path(__file__).parent / "results"
    results_dir.mkdir(exist_ok=True)

    output_file = results_dir / f"benchmark_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n\nResults saved to: {output_file.name}")


if __name__ == "__main__":
    main()
