#!/bin/bash
# Run benchmark suite on ChronoGPT
#
# This script evaluates ChronoGPT on a small subset of standard benchmarks.
# Results are saved to the results/ directory.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=== ChronoGPT Benchmark Suite ==="
echo ""
echo "Model: manelalab/chrono-gpt-instruct-v1-20241231"
echo "Tasks: Custom evaluation (GSM8K-style questions)"
echo ""

# Create results directory
mkdir -p "$SCRIPT_DIR/results"

# Run the Python benchmark script
python "$SCRIPT_DIR/03_temporal_benchmark.py" --limit 10 --output "$SCRIPT_DIR/results/benchmark_results.json"

echo ""
echo "=== Benchmark Complete ==="
echo "Results saved to: $SCRIPT_DIR/results/"
