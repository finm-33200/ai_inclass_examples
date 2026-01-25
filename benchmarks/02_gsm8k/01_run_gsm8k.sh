#!/bin/bash
# Run GSM8K benchmark on multiple models via OpenRouter
#
# Usage:
#   ./01_run_gsm8k.sh           # Quick demo (~20 sec, 5 samples)
#   ./01_run_gsm8k.sh 100       # Fuller run (100 samples)
#   ./01_run_gsm8k.sh all       # Full benchmark (all 1319 samples)

set -e

# Parse argument: number, "all", or default to 5 for quick demo
if [ -z "$1" ]; then
    LIMIT=5
elif [ "$1" = "all" ]; then
    LIMIT=""  # Empty means no limit in lm_eval
else
    LIMIT=$1
fi

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
if [ -z "$LIMIT" ]; then
    echo "Samples per model: ALL (1319)"
else
    echo "Samples per model: $LIMIT"
fi
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

    # Build limit argument (empty LIMIT means no --limit flag = run all)
    LIMIT_ARG=""
    if [ -n "$LIMIT" ]; then
        LIMIT_ARG="--limit $LIMIT"
    fi

    lm_eval --model openai-chat-completions \
        --model_args model="$MODEL",base_url=https://openrouter.ai/api/v1/chat/completions,num_concurrent=50,timeout=120,max_retries=3,tokenizer_backend=None \
        --tasks gsm8k \
        $LIMIT_ARG \
        --apply_chat_template \
        --output_path "$SCRIPT_DIR/results/$SHORTNAME/"

    echo ""
done

echo "========================================"
echo "All benchmarks complete!"
echo "Results saved to: $SCRIPT_DIR/results/"
echo ""
echo "Run 'python 02_compare_models.py' to visualize results"
