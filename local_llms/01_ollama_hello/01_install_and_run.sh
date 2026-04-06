#!/bin/bash
# First Local LLM with Ollama
#
# This script checks for Ollama, pulls TinyLlama, and runs a simple prompt.
# No Python required - just shell and Ollama.

set -e

echo "=== Local LLM with Ollama ==="
echo ""

# Check if Ollama is installed
if ! command -v ollama &> /dev/null; then
    echo "Ollama not found. Installing..."
    echo ""

    # Detect OS and install
    if [[ "$OSTYPE" == "darwin"* ]]; then
        # macOS
        if command -v brew &> /dev/null; then
            brew install ollama
        else
            curl -fsSL https://ollama.ai/install.sh | sh
        fi
    else
        # Linux
        curl -fsSL https://ollama.ai/install.sh | sh
    fi

    echo ""
    echo "Ollama installed successfully!"
fi

echo "Ollama version: $(ollama --version)"
echo ""

# Check if Ollama server is running
if ! curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
    echo "Starting Ollama server in background..."
    ollama serve &
    sleep 3
fi

# Pull TinyLlama if not already present
echo "Checking for TinyLlama model..."
if ! ollama list | grep -q "tinyllama:1.1b"; then
    echo "Pulling tinyllama:1.1b (this may take a few minutes)..."
    ollama pull tinyllama:1.1b
else
    echo "TinyLlama already available."
fi

echo ""
echo "=== Running your first local LLM prompt ==="
echo ""
echo "Prompt: What is machine learning? Explain in 2 sentences."
echo ""
echo "Response:"

# Run a simple prompt
ollama run tinyllama:1.1b "What is machine learning? Explain in 2 sentences."

echo ""
echo "=== Success! ==="
echo "You just ran an LLM entirely on your local machine."
echo ""
echo "Try interactive chat: ollama run tinyllama:1.1b"
