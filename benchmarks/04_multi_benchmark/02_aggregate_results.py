"""
Aggregate results from multi-benchmark suite.

Creates:
1. Bar chart comparing models across dimensions
2. Summary table with rankings
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_results(results_dir: Path) -> dict:
    """Load lm-eval results from a directory."""
    json_files = list(results_dir.glob("**/results*.json"))
    if not json_files:
        return None

    latest = max(json_files, key=lambda p: p.stat().st_mtime)

    with open(latest) as f:
        return json.load(f)


def extract_scores(results: dict) -> dict[str, float]:
    """Extract scores for each benchmark from lm-eval results."""
    if results is None:
        return {}

    scores = {}

    try:
        all_results = results.get("results", {})

        # GSM8K - prefer strict-match (requires "#### answer" format) for accurate scoring
        if "gsm8k" in all_results:
            for metric in ["exact_match,strict-match", "exact_match,flexible-extract", "exact_match,none", "acc,none"]:
                if metric in all_results["gsm8k"]:
                    scores["GSM8K"] = all_results["gsm8k"][metric]
                    break

        # TruthfulQA (generation version)
        if "truthfulqa_gen" in all_results:
            # Use rougeL_acc as the primary metric for gen version
            # Note: metrics have ",none" suffix in lm-eval output
            for metric in ["rougeL_acc,none", "bleu_acc,none", "rouge1_acc,none"]:
                if metric in all_results["truthfulqa_gen"]:
                    scores["TruthfulQA"] = all_results["truthfulqa_gen"][metric]
                    break

        # MMLU (average across CoT subjects)
        mmlu_scores = []
        for task_name, task_results in all_results.items():
            if task_name.startswith("mmlu_cot_llama_") or (task_name.startswith("mmlu_") and "cot" not in task_name):
                for metric in ["exact_match,strict_match", "exact_match,none", "acc,none"]:
                    if metric in task_results:
                        mmlu_scores.append(task_results[metric])
                        break
        if mmlu_scores:
            scores["MMLU"] = np.mean(mmlu_scores)

    except (KeyError, TypeError) as e:
        print(f"Error extracting scores: {e}")

    return scores


def create_bar_chart(all_scores: dict, output_path: Path):
    """Create grouped bar chart comparing models across benchmarks."""
    benchmarks = ["GSM8K", "MMLU", "TruthfulQA"]

    fig, ax = plt.subplots(figsize=(12, 6))

    x = np.arange(len(benchmarks))
    width = 0.25
    colors = ["#3498db", "#e74c3c", "#2ecc71"]

    for i, (model_name, scores) in enumerate(all_scores.items()):
        values = [scores.get(b, 0) * 100 for b in benchmarks]
        offset = (i - 1) * width
        bars = ax.bar(x + offset, values, width, label=model_name, color=colors[i])

        # Add value labels on bars
        for bar, val in zip(bars, values):
            if val > 0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 1,
                    f"{val:.1f}%",
                    ha="center",
                    va="bottom",
                    fontsize=9,
                )

    ax.set_ylabel("Accuracy (%)", fontsize=12)
    ax.set_xlabel("Benchmark", fontsize=12)
    ax.set_title(
        "Multi-Benchmark Model Comparison", fontsize=14, fontweight="bold"
    )
    ax.set_xticks(x)
    ax.set_xticklabels(benchmarks, fontsize=11)
    ax.legend()
    ax.set_ylim(0, 100)
    ax.yaxis.grid(True, linestyle="--", alpha=0.7)
    ax.set_axisbelow(True)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Bar chart saved to: {output_path}")


def print_summary_table(all_scores: dict):
    """Print a summary table of results."""
    benchmarks = ["GSM8K", "MMLU", "TruthfulQA"]

    print("\n" + "=" * 60)
    print("BENCHMARK SUMMARY")
    print("=" * 60)

    # Header
    header = f"{'Model':<20}"
    for b in benchmarks:
        header += f"{b:>12}"
    header += f"{'Average':>12}"
    print(header)
    print("-" * 60)

    # Data rows
    for model_name, scores in all_scores.items():
        row = f"{model_name:<20}"
        for b in benchmarks:
            score = scores.get(b, 0)
            row += f"{score * 100:>11.1f}%"
        # Average over benchmarks that were actually run (present in scores dict)
        attempted_scores = [scores[b] for b in benchmarks if b in scores]
        avg = np.mean(attempted_scores) * 100 if attempted_scores else 0
        row += f"{avg:>11.1f}%"
        print(row)

    print("=" * 60)

    # Rankings
    print("\nRANKINGS BY BENCHMARK:")
    print("-" * 40)
    for benchmark in benchmarks:
        ranked = sorted(
            all_scores.items(),
            key=lambda x: x[1].get(benchmark, 0),
            reverse=True,
        )
        winner = ranked[0][0] if ranked and ranked[0][1].get(benchmark, 0) > 0 else "N/A"
        print(f"{benchmark:<15}: {winner}")

    # Overall winner - average over benchmarks actually attempted
    avg_scores = {
        model: np.mean([scores[b] for b in benchmarks if b in scores])
        for model, scores in all_scores.items()
    }
    overall_winner = max(avg_scores, key=avg_scores.get) if avg_scores else "N/A"
    print(f"\n{'OVERALL':<15}: {overall_winner}")


def main():
    script_dir = Path(__file__).parent
    results_dir = script_dir / "results"

    if not results_dir.exists():
        print("Error: results/ directory not found")
        print("Run ./01_run_suite.sh first")
        return

    # Model directories and display names
    models = {
        "llama-3.1-8b": "Llama 3.1 8B",
        "gemma-2-9b": "Gemma 2 9B",
        "qwen-2.5-7b": "Qwen 2.5 7B",
    }

    # Load results for each model
    all_scores = {}
    print("Loading results...")

    for model_dir, display_name in models.items():
        model_path = results_dir / model_dir
        if model_path.exists():
            results = load_results(model_path)
            scores = extract_scores(results)
            if scores:
                all_scores[display_name] = scores
                print(f"  {display_name}: {len(scores)} benchmarks loaded")
                for bench, score in scores.items():
                    print(f"    {bench}: {score:.1%}")
            else:
                print(f"  {display_name}: No results found")
        else:
            print(f"  {display_name}: Directory not found")

    if not all_scores:
        print("\nNo results to visualize")
        return

    # Create bar chart
    chart_path = script_dir / "benchmark_comparison.png"
    create_bar_chart(all_scores, chart_path)

    # Print summary table
    print_summary_table(all_scores)


if __name__ == "__main__":
    main()
