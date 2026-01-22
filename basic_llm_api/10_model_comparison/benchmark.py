"""
10_model_comparison/benchmark.py

Run benchmark prompts across multiple models and collect metrics.
Uses OpenRouter for unified access to different model providers.
"""

import json
import time
from datetime import datetime
from pathlib import Path

from decouple import Config, RepositoryEnv
from openai import OpenAI


# Models to benchmark (via OpenRouter)
# Prices are per million tokens - check openrouter.ai for current pricing
MODELS = [
    {
        "id": "google/gemini-2.0-flash-exp:free",
        "name": "Gemini 2.0 Flash (Free)",
        "input_price": 0,  # Per million tokens
        "output_price": 0,
    },
    {
        "id": "openai/gpt-4o-mini",
        "name": "GPT-4o Mini",
        "input_price": 0.15,
        "output_price": 0.60,
    },
    {
        "id": "qwen/qwen3-coder:free",
        "name": "Qwen3 Coder (Free)",
        "input_price": 0,
        "output_price": 0,
    },
]

# Benchmark prompts - finance focused
PROMPTS = [
    {
        "name": "concept_explanation",
        "prompt": "Explain what a yield curve inversion is in 2-3 sentences.",
        "category": "explanation",
    },
    {
        "name": "stock_comparison",
        "prompt": "Compare growth stocks vs value stocks. Which performs better during recessions and why? Answer in 3-4 sentences.",
        "category": "analysis",
    },
    {
        "name": "risk_identification",
        "prompt": "What are the main risks of investing in emerging market bonds? List 3 key risks with brief explanations.",
        "category": "risk",
    },
    {
        "name": "calculation",
        "prompt": "If a bond has a face value of $1000, a coupon rate of 5%, and is trading at $950, what is its current yield? Show your calculation.",
        "category": "math",
    },
]


def run_benchmark(client: OpenAI, model_id: str, prompt: str) -> dict:
    """Run a single benchmark and return metrics."""
    start_time = time.time()

    try:
        response = client.chat.completions.create(
            model=model_id,
            messages=[
                {
                    "role": "system",
                    "content": "You are a knowledgeable financial assistant. Be concise and accurate.",
                },
                {"role": "user", "content": prompt},
            ],
            max_tokens=500,
        )

        elapsed = time.time() - start_time
        output = response.choices[0].message.content

        return {
            "success": True,
            "output": output,
            "latency_seconds": round(elapsed, 2),
            "input_tokens": response.usage.prompt_tokens if response.usage else 0,
            "output_tokens": response.usage.completion_tokens if response.usage else 0,
            "total_tokens": response.usage.total_tokens if response.usage else 0,
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "latency_seconds": time.time() - start_time,
        }


def calculate_cost(
    result: dict, input_price: float, output_price: float
) -> float | None:
    """Calculate cost in USD."""
    if not result["success"]:
        return None
    input_cost = result.get("input_tokens", 0) * input_price / 1_000_000
    output_cost = result.get("output_tokens", 0) * output_price / 1_000_000
    return round(input_cost + output_cost, 6)


def main():
    # Load API key from .env at repository root
    repo_root = Path(__file__).resolve().parents[2]
    config = Config(RepositoryEnv(repo_root / ".env"))

    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=config("OPENROUTER_API_KEY"),
    )

    print("=" * 70)
    print("LLM MODEL BENCHMARK")
    print("=" * 70)
    print(f"Models: {len(MODELS)}")
    print(f"Prompts: {len(PROMPTS)}")
    print(f"Total runs: {len(MODELS) * len(PROMPTS)}")
    print()

    results = []
    timestamp = datetime.now().isoformat()

    for model in MODELS:
        print(f"\n--- {model['name']} ---")

        for prompt_info in PROMPTS:
            print(f"  Running: {prompt_info['name']}...", end=" ", flush=True)

            result = run_benchmark(client, model["id"], prompt_info["prompt"])
            cost = calculate_cost(
                result, model["input_price"], model["output_price"]
            )

            if result["success"]:
                print(f"✓ {result['latency_seconds']:.1f}s, ${cost:.6f}")
            else:
                print(f"✗ Error: {result.get('error', 'Unknown')[:50]}")

            results.append(
                {
                    "timestamp": timestamp,
                    "model_id": model["id"],
                    "model_name": model["name"],
                    "prompt_name": prompt_info["name"],
                    "prompt_category": prompt_info["category"],
                    "prompt_text": prompt_info["prompt"],
                    "cost_usd": cost,
                    **result,
                }
            )

    # Save results
    results_dir = Path(__file__).parent / "results"
    results_dir.mkdir(exist_ok=True)

    output_file = results_dir / f"benchmark_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n\nResults saved to: {output_file}")

    # Summary table
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"{'Model':<30} {'Avg Latency':<12} {'Total Cost':<12} {'Success'}")
    print("-" * 70)

    for model in MODELS:
        model_results = [r for r in results if r["model_id"] == model["id"]]
        successful = [r for r in model_results if r["success"]]

        if successful:
            avg_latency = sum(r["latency_seconds"] for r in successful) / len(successful)
            total_cost = sum(r["cost_usd"] or 0 for r in successful)
        else:
            avg_latency = 0
            total_cost = 0

        success_rate = len(successful) / len(model_results) if model_results else 0

        print(
            f"{model['name']:<30} {avg_latency:>8.2f}s    ${total_cost:>8.6f}    {success_rate:>6.0%}"
        )


if __name__ == "__main__":
    main()
