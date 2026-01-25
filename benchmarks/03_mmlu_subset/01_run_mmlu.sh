#!/bin/bash
# Run MMLU benchmark on finance-relevant subjects via OpenRouter
#
# Usage:
#   ./01_run_mmlu.sh        # Run with 50 samples per subject (default)
#   ./01_run_mmlu.sh 100    # Run with 100 samples per subject

set -e

# Number of samples per subject (default 50)
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

# MMLU subjects to test (finance-relevant)
# Using mmlu_cot_llama_* for generation-based evaluation (API compatible)
# Standard mmlu_* tasks require logprobs which API models don't support
TASKS="mmlu_cot_llama_econometrics,mmlu_cot_llama_high_school_macroeconomics,mmlu_cot_llama_high_school_microeconomics,mmlu_cot_llama_college_mathematics,mmlu_cot_llama_college_computer_science,mmlu_cot_llama_professional_accounting"

# Models to benchmark
MODELS=(
    "meta-llama/llama-3.1-8b-instruct:llama-3.1-8b"
    "google/gemma-2-9b-it:gemma-2-9b"
    "qwen/qwen-2.5-7b-instruct:qwen-2.5-7b"
)

echo "MMLU Benchmark (Finance-Relevant Subjects)"
echo "==========================================="
echo "Subjects: econometrics, macroeconomics, microeconomics,"
echo "          college_mathematics, college_computer_science,"
echo "          professional_accounting"
echo "Samples per subject: $LIMIT"
echo "Models: ${#MODELS[@]}"
echo ""

# Create results directory
mkdir -p "$SCRIPT_DIR/results"

# Run benchmark for each model
for MODEL_ENTRY in "${MODELS[@]}"; do
    MODEL="${MODEL_ENTRY%%:*}"
    SHORTNAME="${MODEL_ENTRY##*:}"

    echo "----------------------------------------"
    echo "Running: $MODEL"
    echo "----------------------------------------"

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
echo "Results saved to: $SCRIPT_DIR/results/"
echo ""
echo "Run 'python 02_analyze_results.py' to visualize results"
