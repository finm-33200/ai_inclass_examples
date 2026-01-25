#!/bin/bash
# First benchmark with lm-evaluation-harness
#
# This script runs a minimal GSM8K benchmark to verify your setup.
# Cost: ~$0.001-0.002 for 10 examples

set -e

# Get script directory and repo root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Load environment variables from repo root .env
if [ -f "$REPO_ROOT/.env" ]; then
    set -a
    source "$REPO_ROOT/.env"
    set +a
else
    echo "Error: .env file not found at $REPO_ROOT/.env"
    exit 1
fi

# lm-eval uses OPENAI_API_KEY for OpenAI-compatible endpoints
export OPENAI_API_KEY="$OPENROUTER_API_KEY"

if [ -z "$OPENAI_API_KEY" ]; then
    echo "Error: OPENROUTER_API_KEY not set in .env"
    exit 1
fi

echo "Running first benchmark: GSM8K with 10 examples"
echo "Model: meta-llama/llama-3.1-8b-instruct via OpenRouter"
echo ""

# Run the benchmark
# Note: OpenRouter requires the full chat/completions URL
lm_eval --model openai-chat-completions \
    --model_args model=meta-llama/llama-3.1-8b-instruct,base_url=https://openrouter.ai/api/v1/chat/completions,tokenizer_backend=None \
    --tasks gsm8k \
    --limit 10 \
    --apply_chat_template \
    --output_path "$SCRIPT_DIR/results/"

echo ""
echo "Results saved to $SCRIPT_DIR/results/"
