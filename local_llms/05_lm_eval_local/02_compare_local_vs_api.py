"""
02_compare_local_vs_api.py

Compare local model benchmark results to known API model performance.
This helps contextualize what the local results mean.
"""

import json
from pathlib import Path


# Known benchmark results for reference (approximate values from public benchmarks)
REFERENCE_SCORES = {
    "gsm8k": {
        "GPT-4": 92.0,
        "GPT-3.5-turbo": 57.1,
        "Llama-3.1-70B": 83.7,
        "Llama-3.1-8B": 56.4,
        "Llama-3.2-3B": 44.4,
        "Mistral-7B": 37.8,
        "Phi-2 (2.7B)": 57.2,
        "TinyLlama (1.1B)": 2.3,  # Expected low score for small model
    },
    "hellaswag": {
        "GPT-4": 95.3,
        "GPT-3.5-turbo": 85.5,
        "Llama-3.1-70B": 88.0,
        "Llama-3.1-8B": 82.0,
        "Mistral-7B": 81.0,
        "Phi-2 (2.7B)": 75.1,
        "TinyLlama (1.1B)": 59.2,
    },
}


def find_latest_results(results_dir):
    """Find the most recent results JSON file."""
    results_path = Path(results_dir)
    if not results_path.exists():
        return None

    # Look for results files
    json_files = list(results_path.glob("**/*results*.json"))
    if not json_files:
        return None

    # Return the most recently modified
    return max(json_files, key=lambda p: p.stat().st_mtime)


def load_results(results_file):
    """Load and parse results JSON."""
    with open(results_file) as f:
        data = json.load(f)
    return data


def display_comparison(local_results, task_name):
    """Display local results alongside reference scores."""
    print(f"\n{'='*60}")
    print(f"Comparison: {task_name.upper()}")
    print(f"{'='*60}\n")

    # Get local accuracy
    local_score = None
    if "results" in local_results:
        task_results = local_results["results"].get(task_name, {})
        # Try different metric names
        for metric in ["acc,none", "exact_match,strict-match", "acc_norm,none"]:
            if metric in task_results:
                local_score = task_results[metric] * 100  # Convert to percentage
                break

    print(f"{'Model':<25} {'Score':>10}")
    print("-" * 40)

    # Reference scores
    if task_name in REFERENCE_SCORES:
        for model, score in sorted(
            REFERENCE_SCORES[task_name].items(), key=lambda x: -x[1]
        ):
            marker = ""
            print(f"{model:<25} {score:>10.1f}%{marker}")

    print("-" * 40)

    # Local result
    if local_score is not None:
        print(f"{'Your Local Model':<25} {local_score:>10.1f}% <-- Your result")
    else:
        print("Could not find local score in results file.")

    print()


def main():
    print("Local vs API Model Comparison")
    print("=" * 60)

    script_dir = Path(__file__).parent
    results_dir = script_dir / "results"

    results_file = find_latest_results(results_dir)

    if results_file is None:
        print("\nNo results found!")
        print("\nTo generate results, run:")
        print("  ./01_tinyllama_gsm8k.sh")
        print()
        print("Showing reference scores for context:\n")

        for task_name, scores in REFERENCE_SCORES.items():
            print(f"\n{task_name.upper()}:")
            print(f"{'Model':<25} {'Score':>10}")
            print("-" * 40)
            for model, score in sorted(scores.items(), key=lambda x: -x[1]):
                print(f"{model:<25} {score:>10.1f}%")
        return

    print(f"\nLoading results from: {results_file}")

    results = load_results(results_file)

    # Determine which tasks were run
    if "results" in results:
        for task_name in results["results"].keys():
            display_comparison(results, task_name)

    # Model info
    if "config" in results:
        config = results["config"]
        print(f"\nModel evaluated: {config.get('model_args', 'Unknown')}")
        print(f"Device: {config.get('device', 'Unknown')}")
        if "limit" in config:
            print(f"Examples evaluated: {config['limit']}")
            print("\nNote: Increase --limit for more statistically significant results.")


if __name__ == "__main__":
    main()
