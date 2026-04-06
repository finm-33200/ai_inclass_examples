#!/bin/bash
# Benchmark TinyLlama on GSM8K using lm-evaluation-harness
#
# This runs entirely locally - no API calls or costs.
# Uses a small subset (--limit 20) for quick testing.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=== Local Model Benchmark: TinyLlama on GSM8K ==="
echo ""
echo "Model: TinyLlama/TinyLlama-1.1B-Chat-v1.0"
echo "Task: gsm8k (grade school math)"
echo "Limit: 20 examples (increase for more accurate results)"
echo ""

# Create results directory
mkdir -p "$SCRIPT_DIR/results"

# Determine device
if python -c "import torch; print(torch.cuda.is_available())" 2>/dev/null | grep -q "True"; then
    DEVICE="cuda"
elif python -c "import torch; print(torch.backends.mps.is_available())" 2>/dev/null | grep -q "True"; then
    DEVICE="mps"
else
    DEVICE="cpu"
fi

echo "Device: $DEVICE"
echo ""
echo "Running benchmark (this may take a few minutes)..."
echo ""

# Run the benchmark
lm_eval --model hf \
    --model_args pretrained=TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
    --tasks gsm8k \
    --limit 20 \
    --device "$DEVICE" \
    --batch_size 1 \
    --output_path "$SCRIPT_DIR/results/"

echo ""
echo "=== Benchmark Complete ==="
echo "Results saved to: $SCRIPT_DIR/results/"
echo ""
echo "Note: TinyLlama (1.1B params) will score lower than larger models."
echo "This is expected - the goal is demonstrating the evaluation process."
