"""
Compare GSM8K benchmark results across models.

Reads lm-eval JSON output and creates a bar chart comparing accuracy.
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt


def load_results(results_dir: Path) -> dict:
    """Load lm-eval results from a directory."""
    # lm-eval creates timestamped subdirectories
    # Find the most recent results file
    json_files = list(results_dir.glob("**/results*.json"))
    if not json_files:
        return None

    # Sort by modification time, get most recent
    latest = max(json_files, key=lambda p: p.stat().st_mtime)

    with open(latest) as f:
        return json.load(f)


def extract_gsm8k_accuracy(results: dict) -> float | None:
    """Extract GSM8K accuracy from lm-eval results."""
    if results is None:
        return None

    # Navigate the results structure
    # Structure: results -> results -> gsm8k -> exact_match,strict-match
    try:
        gsm8k_results = results["results"]["gsm8k"]
        # Try different metric names (lm-eval versions vary)
        for metric in ["exact_match,strict-match", "exact_match,none", "acc,none"]:
            if metric in gsm8k_results:
                return gsm8k_results[metric]
        # If none found, try to get any accuracy-like metric
        for key, value in gsm8k_results.items():
            if "match" in key or "acc" in key:
                if isinstance(value, (int, float)):
                    return value
    except (KeyError, TypeError):
        pass

    return None


def main():
    script_dir = Path(__file__).parent
    results_dir = script_dir / "results"

    if not results_dir.exists():
        print("Error: results/ directory not found")
        print("Run ./01_run_gsm8k.sh first")
        return

    # Model directories and display names
    models = {
        "llama-3.1-8b": "Llama 3.1 8B",
        "gemma-2-9b": "Gemma 2 9B",
        "qwen-2.5-7b": "Qwen 2.5 7B",
    }

    # Load results
    accuracies = {}
    for model_dir, display_name in models.items():
        model_path = results_dir / model_dir
        if model_path.exists():
            results = load_results(model_path)
            acc = extract_gsm8k_accuracy(results)
            if acc is not None:
                accuracies[display_name] = acc
                print(f"{display_name}: {acc:.1%}")
            else:
                print(f"{display_name}: No results found")
        else:
            print(f"{display_name}: Directory not found")

    if not accuracies:
        print("\nNo results to plot")
        return

    # Create bar chart
    fig, ax = plt.subplots(figsize=(10, 6))

    models_list = list(accuracies.keys())
    accs = [accuracies[m] * 100 for m in models_list]  # Convert to percentage

    bars = ax.bar(models_list, accs, color=["#3498db", "#e74c3c", "#2ecc71"])

    # Add value labels on bars
    for bar, acc in zip(bars, accs):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1,
            f"{acc:.1f}%",
            ha="center",
            va="bottom",
            fontsize=12,
            fontweight="bold",
        )

    ax.set_ylabel("Accuracy (%)", fontsize=12)
    ax.set_xlabel("Model", fontsize=12)
    ax.set_title("GSM8K Benchmark: Model Comparison", fontsize=14, fontweight="bold")
    ax.set_ylim(0, 100)

    # Add grid for readability
    ax.yaxis.grid(True, linestyle="--", alpha=0.7)
    ax.set_axisbelow(True)

    plt.tight_layout()

    # Save figure
    output_path = script_dir / "gsm8k_comparison.png"
    plt.savefig(output_path, dpi=150)
    print(f"\nChart saved to: {output_path}")

    plt.show()


if __name__ == "__main__":
    main()
