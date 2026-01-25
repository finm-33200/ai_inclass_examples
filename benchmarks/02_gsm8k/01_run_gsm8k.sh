#!/bin/bash
# Run GSM8K benchmark on multiple models via OpenRouter
#
# Usage:
#   ./01_run_gsm8k.sh        # Run with 100 samples (default)
#   ./01_run_gsm8k.sh 20     # Run with 20 samples (quick test)

set -e

# Number of samples (default 100)
LIMIT=${1:-100}

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

# Models to benchmark
MODELS=(
    "meta-llama/llama-3.1-8b-instruct:llama-3.1-8b"
    "google/gemma-2-9b-it:gemma-2-9b"
    "qwen/qwen-2.5-7b-instruct:qwen-2.5-7b"
)

echo "GSM8K Benchmark"
echo "==============="
echo "Samples per model: $LIMIT"
echo "Models: ${#MODELS[@]}"
echo ""

# Create results directory
mkdir -p "$SCRIPT_DIR/results"

# Run benchmark for each model
for MODEL_ENTRY in "${MODELS[@]}"; do
    # Split model:shortname
    MODEL="${MODEL_ENTRY%%:*}"
    SHORTNAME="${MODEL_ENTRY##*:}"

    echo "----------------------------------------"
    echo "Running: $MODEL"
    echo "----------------------------------------"

    lm_eval --model openai-chat-completions \
        --model_args model="$MODEL",base_url=https://openrouter.ai/api/v1/chat/completions,tokenizer_backend=None \
        --tasks gsm8k \
        --limit "$LIMIT" \
        --apply_chat_template \
        --output_path "$SCRIPT_DIR/results/$SHORTNAME/"

    echo ""
done

echo "========================================"
echo "All benchmarks complete!"
echo "Results saved to: $SCRIPT_DIR/results/"
echo ""
echo "Run 'python 02_compare_models.py' to visualize results"
