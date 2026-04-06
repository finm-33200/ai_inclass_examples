# 03_huggingface_transformers - Direct HuggingFace Loading

Load models directly from HuggingFace Hub using the transformers library. This provides maximum flexibility and is required for models not available in Ollama (like ChronoGPT).

## What You'll Learn

- Loading models with `AutoModelForCausalLM` and `AutoTokenizer`
- Running inference with transformers pipelines
- Working with different model architectures

## Key Concepts

- **AutoModel**: Automatically detects and loads the correct model architecture
- **Tokenizer**: Converts text to tokens the model understands
- **Model Caching**: Downloaded models are cached in `~/.cache/huggingface/`

## Prerequisites

- Python 3.10+
- ~4GB disk space for model weights
- ~4GB RAM

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Note: First run will download model weights (~2GB for TinyLlama).

## Running the Examples

```bash
# Basic loading and generation
python 01_load_and_generate.py

# Compare with Phi-2 (larger model)
python 02_phi2_generation.py  # Requires ~6GB RAM
```

## Try It

1. Experiment with `max_new_tokens` and `temperature` parameters
2. Try different prompts and observe model behavior
3. Check `~/.cache/huggingface/` to see downloaded models

## Models Used

| Model | Parameters | Size | Notes |
|-------|------------|------|-------|
| TinyLlama-1.1B-Chat | 1.1B | ~2GB | Fast, chat-tuned |
| Phi-2 | 2.7B | ~6GB | Microsoft's compact reasoning model |

## Why HuggingFace Direct?

- **Access to all models**: 100,000+ models on the Hub
- **Fine-tuned variants**: Custom models for specific tasks
- **Research models**: Papers often release weights on HuggingFace
- **ChronoGPT**: Required for the capstone examples
