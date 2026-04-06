# 01_ollama_hello - First Local LLM

The lowest barrier entry point to running LLMs locally. One install, one command to pull a model, and you're running inference on your own machine.

## What You'll Learn

- Installing Ollama
- Pulling and running models from the command line
- Using the Python `ollama` package for programmatic access

## Key Concepts

- **Ollama**: A tool that makes running local LLMs as easy as Docker makes running containers
- **Model pulling**: Downloading pre-quantized model weights
- **Inference**: Running the model to generate text

## Prerequisites

- macOS, Linux, or Windows (with WSL2)
- ~2GB disk space for TinyLlama model
- ~2GB RAM available

## Setup

### 1. Install Ollama

**macOS/Linux:**
```bash
curl -fsSL https://ollama.ai/install.sh | sh
```

**macOS (Homebrew):**
```bash
brew install ollama
```

**Windows:**
Download from [ollama.ai](https://ollama.ai/)

### 2. Start Ollama (if not auto-started)

```bash
ollama serve
```

### 3. Pull TinyLlama

```bash
ollama pull tinyllama:1.1b
```

## Running the Examples

### Shell script (no Python needed)

```bash
chmod +x 01_install_and_run.sh
./01_install_and_run.sh
```

### Python script

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python 02_interactive_chat.py
```

## Try It

1. Try different prompts in the Python script
2. Pull a different model: `ollama pull phi3:mini`
3. List available models: `ollama list`
4. Chat interactively: `ollama run tinyllama:1.1b`

## Troubleshooting

- **"connection refused"**: Make sure `ollama serve` is running
- **Out of memory**: Close other applications or try a smaller model
- **Slow generation**: Normal for CPU inference; GPU acceleration helps significantly
