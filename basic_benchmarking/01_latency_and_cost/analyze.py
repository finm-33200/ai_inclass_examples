"""
01_latency_and_cost/analyze.py

Load benchmark results and compute summary statistics.
Demonstrates how to analyze latency/cost tradeoffs across models.
"""

import json
from pathlib import Path


def load_latest_results() -> list[dict]:
    """Find and load the most recent benchmark results."""
    results_dir = Path(__file__).parent / "results"

    if not results_dir.exists():
        raise FileNotFoundError("No results directory. Run benchmark.py first.")

    json_files = sorted(results_dir.glob("benchmark_*.json"))
    if not json_files:
        raise FileNotFoundError("No benchmark files found. Run benchmark.py first.")

    latest = json_files[-1]
    print(f"Loading: {latest.name}\n")

    with open(latest) as f:
        return json.load(f)


def analyze_by_model(results: list[dict]) -> None:
    """Print summary statistics grouped by model."""
    print("=" * 60)
    print("SUMMARY BY MODEL")
    print("=" * 60)
    print(f"{'Model':<20} {'Avg Latency':<12} {'Avg Tokens':<12} {'Total Cost'}")
    print("-" * 60)

    # Group results by model
    models = {}
    for r in results:
        model = r["model_name"]
        if model not in models:
            models[model] = []
        if "error" not in r:
            models[model].append(r)

    for model_name, model_results in models.items():
        if not model_results:
            print(f"{model_name:<20} No successful results")
            continue

        avg_latency = sum(r["latency_seconds"] for r in model_results) / len(model_results)
        avg_tokens = sum(r["total_tokens"] for r in model_results) / len(model_results)
        total_cost = sum(r["cost_usd"] for r in model_results)

        print(f"{model_name:<20} {avg_latency:>8.2f}s    {avg_tokens:>8.0f}      ${total_cost:.6f}")


def analyze_by_prompt(results: list[dict]) -> None:
    """Print latency comparison for each prompt across models."""
    print("\n" + "=" * 60)
    print("LATENCY BY PROMPT")
    print("=" * 60)

    # Get unique prompts and models
    prompts = list(dict.fromkeys(r["prompt_name"] for r in results))
    models = list(dict.fromkeys(r["model_name"] for r in results))

    # Header
    header = f"{'Prompt':<20}"
    for model in models:
        header += f"{model:<15}"
    print(header)
    print("-" * 60)

    # Rows
    for prompt_name in prompts:
        row = f"{prompt_name:<20}"
        for model in models:
            match = next(
                (r for r in results
                 if r["prompt_name"] == prompt_name and r["model_name"] == model),
                None
            )
            if match and "latency_seconds" in match:
                row += f"{match['latency_seconds']:>8.2f}s      "
            else:
                row += f"{'error':<15}"
        print(row)


def show_cost_comparison(results: list[dict]) -> None:
    """Compare cost between models."""
    print("\n" + "=" * 60)
    print("COST COMPARISON")
    print("=" * 60)

    # Calculate total cost per model
    model_costs = {}
    for r in results:
        if "cost_usd" in r:
            model = r["model_name"]
            model_costs[model] = model_costs.get(model, 0) + r["cost_usd"]

    if len(model_costs) >= 2:
        costs = list(model_costs.items())
        cheapest = min(costs, key=lambda x: x[1])

        print(f"Cheapest: {cheapest[0]} (${cheapest[1]:.6f})")
        print("\nCost ratios:")
        for model, cost in costs:
            ratio = cost / cheapest[1] if cheapest[1] > 0 else 0
            print(f"  {model}: {ratio:.1f}x")


def main():
    results = load_latest_results()

    analyze_by_model(results)
    analyze_by_prompt(results)
    show_cost_comparison(results)

    print("\n" + "=" * 60)
    print("KEY TAKEAWAYS")
    print("=" * 60)
    print("1. Latency varies by model and prompt complexity")
    print("2. Token counts are similar across models for same prompt")
    print("3. Cost differences can be 10-50x between models")
    print("4. Choose based on your quality/cost/latency requirements")


if __name__ == "__main__":
    main()
