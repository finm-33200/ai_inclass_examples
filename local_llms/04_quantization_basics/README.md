# 04_quantization_basics - Run Larger Models with Less Memory

Quantization reduces model precision (e.g., from 32-bit to 4-bit), dramatically cutting memory usage while preserving most quality. This enables running larger models on consumer hardware.

## What You'll Learn

- What quantization is and why it matters
- Different quantization approaches (float16, int8, int4)
- Trade-offs between memory, speed, and quality

## Key Concepts

- **Precision**: Number of bits used to represent each weight (32-bit, 16-bit, 8-bit, 4-bit)
- **Memory Savings**: 4-bit uses ~8x less memory than 32-bit
- **Quality Trade-off**: Lower precision slightly reduces output quality

## Memory Comparison

| Precision | Bits | 7B Model Size | Relative Quality |
|-----------|------|---------------|------------------|
| float32 | 32 | ~28 GB | 100% (baseline) |
| float16 | 16 | ~14 GB | ~99.9% |
| int8 | 8 | ~7 GB | ~99% |
| int4 | 4 | ~3.5 GB | ~95-98% |

## Prerequisites

- Python 3.10+
- For bitsandbytes: Linux with CUDA GPU (not supported on macOS)
- For precision comparison: Any platform

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Running the Examples

```bash
# Compare float32 vs float16 (works everywhere)
python 01_precision_comparison.py

# Bitsandbytes 4-bit quantization (Linux + CUDA only)
python 02_bitsandbytes_quant.py
```

## Platform Notes

- **macOS (Apple Silicon)**: Use float16 via MPS. Bitsandbytes not supported.
- **Linux + CUDA**: Full support for bitsandbytes int4/int8 quantization.
- **CPU only**: float32/float16 work but inference is slower.

## Try It

1. Compare memory usage between precision levels
2. Evaluate output quality differences
3. Try loading a larger model (7B) with 4-bit quantization

## Why Quantization Matters for Finance

- Run larger, more capable models on local hardware
- Keep sensitive financial data on-premises
- Reduce cloud compute costs for batch processing
- Enable real-time inference on consumer GPUs
