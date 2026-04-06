# Local LLM Examples

This directory contains progressive examples for running Large Language Models locally. Examples progress from simple Ollama setup to research-grade benchmarking with ChronoGPT.

## Why Local LLMs?

- **Privacy**: Keep sensitive data on your machine
- **Cost**: No API charges after initial setup
- **Research Models**: Access specialized models not available via APIs (e.g., ChronoGPT)
- **No Rate Limits**: Run as many queries as your hardware allows
- **Offline Access**: Work without internet connectivity

## Prerequisites

- Python 3.10+
- 8GB RAM recommended (16GB+ for larger models)
- [Ollama](https://ollama.ai/) (for examples 01-02)
- ~10GB disk space for model weights

## Examples

| # | Directory | Description | RAM Needed |
|---|-----------|-------------|------------|
| 01 | `01_ollama_hello/` | First local LLM - lowest barrier entry | ~2GB |
| 02 | `02_ollama_openai_api/` | Use OpenAI SDK with local models | ~2GB |
| 03 | `03_huggingface_transformers/` | Load models directly from HuggingFace | ~4GB |
| 04 | `04_quantization_basics/` | Run larger models via quantization | ~4GB |
| 05 | `05_lm_eval_local/` | Benchmark local models with lm-eval | ~4GB |
| 06 | `06_chronogpt_exploration/` | Explore temporally-aware ChronoGPT | ~4GB |
| 07 | `07_chronogpt_benchmark/` | Research-grade ChronoGPT evaluation | ~4GB |

## Rationale for Ordering

1. **Ollama first (01-02)**: One-command installation, manages model downloads, easiest path to running a local LLM. OpenAI SDK compatibility makes transition from cloud APIs seamless.

2. **HuggingFace Transformers (03-04)**: Direct model loading provides maximum flexibility. Required for models not available in Ollama (like ChronoGPT). Quantization enables running larger models on constrained hardware.

3. **Benchmarking (05)**: Use lm-evaluation-harness with local models. Bridges to the `benchmarks/` directory examples but runs entirely locally.

4. **ChronoGPT capstone (06-07)**: Finance-relevant research models demonstrating temporal consistency. Shows how to work with cutting-edge models not available through commercial APIs.

## Running Examples

Each example is self-contained. Navigate to the subdirectory and follow its README:

```bash
cd 01_ollama_hello
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
python 02_interactive_chat.py
```

## Hardware Notes

- **CPU-only**: All examples work on CPU, though inference will be slower
- **Apple Silicon**: Native support via Metal acceleration in Ollama and PyTorch
- **NVIDIA GPU**: CUDA acceleration available for faster inference
- **Memory**: Models are loaded into RAM/VRAM; close other applications if needed

## Resources

- [Ollama](https://ollama.ai/) - Easy local LLM runner
- [HuggingFace Hub](https://huggingface.co/models) - Model repository
- [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) - Benchmarking framework
- [ChronoGPT Collection](https://huggingface.co/collections/manelalab/chrono-gpt-67afefc7c97c4b80de0e5bdc) - Temporally-consistent models
- [ChronoGPT Paper](https://arxiv.org/abs/2502.01029) - Research paper on temporal consistency
