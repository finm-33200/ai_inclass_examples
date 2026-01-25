#!/bin/bash
# Run full benchmark suite on multiple models via OpenRouter
#
# Runs: GSM8K, TruthfulQA, MMLU (CoT subset)
#
# Usage:
#   ./01_run_suite.sh        # Run with 50 samples per benchmark (default)
#   ./01_run_suite.sh 100    # Run with 100 samples per benchmark

set -e

# Number of samples per benchmark (default 5 for quick demo)
LIMIT=${1:-5}

# Get script directory and repo root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Load environment variables
if [ -f "$REPO_ROOT/.env" ]; then
    set -a
    source "$REPO_ROOT/.env"
    set +a
else
    echo "Error: .env file not found at $REPO_ROOT/.env"
    exit 1
fi

export OPENAI_API_KEY="$OPENROUTER_API_KEY"

if [ -z "$OPENAI_API_KEY" ]; then
    echo "Error: OPENROUTER_API_KEY not set in .env"
    exit 1
fi

# Tasks to run (3 benchmarks for quick demo)
# Note: Using CoT/generation-based versions because standard versions require logprobs
# HellaSwag requires logprobs so we use truthfulqa_gen instead
TASKS="gsm8k,truthfulqa_gen,mmlu_cot_llama_econometrics"

# Models to benchmark
MODELS=(
    "meta-llama/llama-3.1-8b-instruct:llama-3.1-8b"
    "google/gemma-2-9b-it:gemma-2-9b"
    "qwen/qwen-2.5-7b-instruct:qwen-2.5-7b"
)

echo "Multi-Benchmark Suite (Quick Demo)"
echo "==================================="
echo "Benchmarks: GSM8K, TruthfulQA, MMLU CoT (econometrics)"
echo "Samples per benchmark: $LIMIT"
echo "Models: ${#MODELS[@]}"
echo ""
echo "Expected runtime: ~2 minutes"
echo ""

# Create results directory
mkdir -p "$SCRIPT_DIR/results"

# Run benchmark for each model
for MODEL_ENTRY in "${MODELS[@]}"; do
    MODEL="${MODEL_ENTRY%%:*}"
    SHORTNAME="${MODEL_ENTRY##*:}"

    echo "========================================"
    echo "Running: $MODEL"
    echo "========================================"

    lm_eval --model openai-chat-completions \
        --model_args model="$MODEL",base_url=https://openrouter.ai/api/v1/chat/completions,num_concurrent=50,timeout=120,max_retries=3,tokenizer_backend=None \
        --tasks "$TASKS" \
        --limit "$LIMIT" \
        --apply_chat_template \
        --output_path "$SCRIPT_DIR/results/$SHORTNAME/"

    echo ""
done

echo "========================================"
echo "All benchmarks complete!"
echo "========================================"
echo ""
echo "Results saved to: $SCRIPT_DIR/results/"
echo ""
echo "Run 'python 02_aggregate_results.py' to create visualizations"
