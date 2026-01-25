#!/bin/bash
# Run full benchmark suite on multiple models via OpenRouter
#
# Runs: GSM8K, MMLU (CoT subset), TruthfulQA (gen)
#
# Note: HellaSwag and TruthfulQA MC require logprobs which API models don't support
#
# Usage:
#   ./01_run_suite.sh        # Run with 50 samples per benchmark (default)
#   ./01_run_suite.sh 100    # Run with 100 samples per benchmark

set -e

# Number of samples per benchmark (default 50)
LIMIT=${1:-50}

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

# All tasks to run (API-compatible only - generation-based tasks)
# Note: Using CoT versions of MMLU and generation version of TruthfulQA
# because standard versions require logprobs which API models don't support
TASKS="gsm8k,truthfulqa_gen,mmlu_cot_llama_econometrics,mmlu_cot_llama_high_school_macroeconomics,mmlu_cot_llama_college_mathematics"

# Models to benchmark
MODELS=(
    "meta-llama/llama-3.1-8b-instruct:llama-3.1-8b"
    "google/gemma-2-9b-it:gemma-2-9b"
    "qwen/qwen-2.5-7b-instruct:qwen-2.5-7b"
)

echo "Full Benchmark Suite"
echo "===================="
echo "Benchmarks: GSM8K, TruthfulQA (gen), MMLU CoT (3 subjects)"
echo "Samples per benchmark: $LIMIT"
echo "Models: ${#MODELS[@]}"
echo ""
echo "Estimated time: 5-15 minutes depending on API speed"
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
        --model_args model="$MODEL",base_url=https://openrouter.ai/api/v1/chat/completions,tokenizer_backend=None \
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
