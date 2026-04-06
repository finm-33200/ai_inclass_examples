"""
02_analyze_results.py

Analyze and display benchmark results from ChronoGPT evaluation.
"""

import json
from pathlib import Path


def load_results(results_file):
    """Load results from JSON file."""
    with open(results_file) as f:
        return json.load(f)


def display_results(results):
    """Display benchmark results in a formatted table."""
    print("=" * 70)
    print("ChronoGPT Benchmark Results")
    print("=" * 70)
    print()

    if "models" in results:
        for model_name, model_results in results["models"].items():
            print(f"Model: {model_name}")
            print("-" * 50)

            if "tasks" in model_results:
                print(f"{'Task':<20} {'Score':>10} {'Correct':>10}")
                print("-" * 50)

                for task_name, task_results in model_results["tasks"].items():
                    score = task_results.get("score", 0) * 100
                    correct = task_results.get("correct", 0)
                    total = task_results.get("total", 0)
                    print(f"{task_name:<20} {score:>10.1f}% {correct:>6}/{total:<3}")

            print()

    # Summary statistics
    if "summary" in results:
        print("=" * 70)
        print("Summary")
        print("=" * 70)
        for key, value in results["summary"].items():
            print(f"  {key}: {value}")


def main():
    script_dir = Path(__file__).parent
    results_file = script_dir / "results" / "benchmark_results.json"

    if not results_file.exists():
        print("No results found!")
        print()
        print("To generate results, run:")
        print("  ./01_run_benchmark_suite.sh")
        print("  or")
        print("  python 03_temporal_benchmark.py")
        print()
        print("Example output format:")
        print()

        # Show example format
        example = {
            "models": {
                "chrono-gpt-2024": {
                    "tasks": {
                        "math_reasoning": {"score": 0.15, "correct": 3, "total": 20},
                        "general_knowledge": {"score": 0.60, "correct": 12, "total": 20},
                    }
                },
                "chrono-gpt-2000": {
                    "tasks": {
                        "math_reasoning": {"score": 0.10, "correct": 2, "total": 20},
                        "general_knowledge": {"score": 0.45, "correct": 9, "total": 20},
                    }
                },
            },
            "summary": {
                "total_evaluations": 80,
                "models_evaluated": 2,
            },
        }
        display_results(example)
        return

    results = load_results(results_file)
    display_results(results)


if __name__ == "__main__":
    main()
