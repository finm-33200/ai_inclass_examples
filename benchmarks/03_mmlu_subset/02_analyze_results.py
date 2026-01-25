"""
Analyze MMLU benchmark results across models and subjects.

Creates visualizations showing per-subject performance for each model.
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


def extract_mmlu_scores(results: dict) -> dict[str, float]:
    """Extract per-subject MMLU accuracy from lm-eval results."""
    if results is None:
        return {}

    scores = {}
    try:
        for task_name, task_results in results["results"].items():
            # Handle both mmlu_* and mmlu_cot_llama_* task names
            if task_name.startswith("mmlu_cot_llama_") or task_name.startswith("mmlu_"):
                # Extract subject name (remove prefix)
                subject = task_name.replace("mmlu_cot_llama_", "").replace("mmlu_", "")
                subject = subject.replace("_", " ").title()

                # Try different metric names
                acc = None
                for metric in ["exact_match,strict_match", "acc,none", "exact_match,none"]:
                    if metric in task_results:
                        acc = task_results[metric]
                        break

                if acc is not None:
                    scores[subject] = acc
    except (KeyError, TypeError):
        pass

    return scores


def main():
    script_dir = Path(__file__).parent
    results_dir = script_dir / "results"

    if not results_dir.exists():
        print("Error: results/ directory not found")
        print("Run ./01_run_mmlu.sh first")
        return

    # Model directories and display names
    models = {
        "llama-3.1-8b": "Llama 3.1 8B",
        "gemma-2-9b": "Gemma 2 9B",
        "qwen-2.5-7b": "Qwen 2.5 7B",
    }

    # Load results for each model
    all_scores = {}
    all_subjects = set()

    for model_dir, display_name in models.items():
        model_path = results_dir / model_dir
        if model_path.exists():
            results = load_results(model_path)
            scores = extract_mmlu_scores(results)
            if scores:
                all_scores[display_name] = scores
                all_subjects.update(scores.keys())
                print(f"\n{display_name}:")
                for subject, acc in sorted(scores.items()):
                    print(f"  {subject}: {acc:.1%}")
            else:
                print(f"{display_name}: No results found")
        else:
            print(f"{display_name}: Directory not found")

    if not all_scores:
        print("\nNo results to plot")
        return

    # Sort subjects alphabetically
    subjects = sorted(all_subjects)

    # Create grouped bar chart
    fig, ax = plt.subplots(figsize=(14, 8))

    x = np.arange(len(subjects))
    width = 0.25
    colors = ["#3498db", "#e74c3c", "#2ecc71"]

    for i, (model_name, scores) in enumerate(all_scores.items()):
        accs = [scores.get(s, 0) * 100 for s in subjects]
        offset = (i - 1) * width
        bars = ax.bar(x + offset, accs, width, label=model_name, color=colors[i])

    ax.set_ylabel("Accuracy (%)", fontsize=12)
    ax.set_xlabel("Subject", fontsize=12)
    ax.set_title(
        "MMLU Benchmark: Per-Subject Model Comparison", fontsize=14, fontweight="bold"
    )
    ax.set_xticks(x)
    ax.set_xticklabels(subjects, rotation=45, ha="right")
    ax.legend()
    ax.set_ylim(0, 100)
    ax.yaxis.grid(True, linestyle="--", alpha=0.7)
    ax.set_axisbelow(True)

    plt.tight_layout()

    # Save figure
    output_path = script_dir / "mmlu_comparison.png"
    plt.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"\nChart saved to: {output_path}")

    # Print summary table
    print("\n" + "=" * 60)
    print("Summary: Average Accuracy Across Subjects")
    print("=" * 60)
    for model_name, scores in all_scores.items():
        avg = np.mean(list(scores.values())) * 100
        print(f"{model_name}: {avg:.1f}%")


if __name__ == "__main__":
    main()
