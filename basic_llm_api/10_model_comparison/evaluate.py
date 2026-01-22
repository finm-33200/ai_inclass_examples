"""
10_model_comparison/evaluate.py

Analyze benchmark results and display model comparison.
Loads the most recent benchmark results and provides analysis.
"""

import json
from pathlib import Path


def load_latest_results() -> list[dict]:
    """Load the most recent benchmark results."""
    results_dir = Path(__file__).parent / "results"
    result_files = sorted(results_dir.glob("benchmark_*.json"), reverse=True)

    if not result_files:
        raise FileNotFoundError(
            "No benchmark results found. Run benchmark.py first."
        )

    with open(result_files[0]) as f:
        return json.load(f)


def analyze_results(results: list[dict]) -> None:
    """Analyze and display benchmark results."""
    # Group by model
    models = {}
    for r in results:
        model = r["model_name"]
        if model not in models:
            models[model] = []
        models[model].append(r)

    # Group by prompt category
    categories = {}
    for r in results:
        cat = r["prompt_category"]
        if cat not in categories:
            categories[cat] = []
        categories[cat].append(r)

    print("=" * 70)
    print("BENCHMARK ANALYSIS")
    print("=" * 70)

    # Per-model analysis
    print("\n📊 MODEL PERFORMANCE")
    print("-" * 70)
    print(f"{'Model':<30} {'Latency':<10} {'Cost':<10} {'Tokens':<10}")
    print("-" * 70)

    for model_name, model_results in models.items():
        successful = [r for r in model_results if r["success"]]
        if successful:
            avg_latency = sum(r["latency_seconds"] for r in successful) / len(successful)
            total_cost = sum(r["cost_usd"] or 0 for r in successful)
            total_tokens = sum(r.get("total_tokens", 0) for r in successful)
            print(f"{model_name:<30} {avg_latency:>6.2f}s   ${total_cost:>7.5f}   {total_tokens:>6}")

    # Per-category analysis
    print("\n📁 CATEGORY BREAKDOWN")
    print("-" * 70)
    print(f"{'Category':<20} {'Best Model':<25} {'Avg Latency':<12}")
    print("-" * 70)

    for category, cat_results in categories.items():
        successful = [r for r in cat_results if r["success"]]
        if successful:
            # Find fastest model for this category
            by_model = {}
            for r in successful:
                model = r["model_name"]
                if model not in by_model:
                    by_model[model] = []
                by_model[model].append(r["latency_seconds"])

            fastest = min(
                by_model.items(),
                key=lambda x: sum(x[1]) / len(x[1])
            )[0]

            avg_latency = sum(r["latency_seconds"] for r in successful) / len(successful)
            print(f"{category:<20} {fastest:<25} {avg_latency:>8.2f}s")

    # Sample outputs comparison
    print("\n📝 SAMPLE OUTPUT COMPARISON")
    print("-" * 70)

    # Pick one prompt to compare
    sample_prompt = "concept_explanation"
    print(f"Prompt: '{sample_prompt}'")
    print()

    sample_results = [r for r in results if r["prompt_name"] == sample_prompt and r["success"]]
    for r in sample_results:
        print(f"--- {r['model_name']} ({r['latency_seconds']:.1f}s) ---")
        output = r["output"]
        # Truncate long outputs
        if len(output) > 300:
            output = output[:300] + "..."
        print(output)
        print()

    # Cost analysis
    print("💰 COST ANALYSIS")
    print("-" * 70)

    total_by_model = {}
    for r in results:
        if r["success"]:
            model = r["model_name"]
            cost = r["cost_usd"] or 0
            total_by_model[model] = total_by_model.get(model, 0) + cost

    if total_by_model:
        min_cost_model = min(total_by_model.items(), key=lambda x: x[1])
        max_cost_model = max(total_by_model.items(), key=lambda x: x[1])

        print(f"Cheapest: {min_cost_model[0]} (${min_cost_model[1]:.6f})")
        print(f"Most expensive: {max_cost_model[0]} (${max_cost_model[1]:.6f})")

        if min_cost_model[1] > 0:
            ratio = max_cost_model[1] / min_cost_model[1]
            print(f"Cost ratio: {ratio:.1f}x")

    print("\n" + "=" * 70)
    print("TIP: Use cheaper models for simple tasks, premium models for complex ones.")
    print("=" * 70)


def main():
    try:
        results = load_latest_results()
        print(f"Loaded {len(results)} benchmark results\n")
        analyze_results(results)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("\nRun `python benchmark.py` first to generate results.")


if __name__ == "__main__":
    main()
