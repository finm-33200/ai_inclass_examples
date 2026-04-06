"""
01_load_chronogpt.py

Load and run inference with ChronoGPT, a temporally-consistent language model.
This model only "knows" information available up to its training cutoff date.

ChronoGPT uses a custom model architecture, so we need to download and use
their custom model class from HuggingFace Hub.
"""

import sys
import torch
from huggingface_hub import hf_hub_download
import tiktoken


def download_model_code(model_name):
    """Download the custom ChronoGPT model code."""
    # Download model code
    model_code_path = hf_hub_download(model_name, "ChronoGPT_instruct.py")

    # Add directory to path so we can import
    import importlib.util

    spec = importlib.util.spec_from_file_location("chronogpt", model_code_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    return module


def main():
    print("Loading ChronoGPT from HuggingFace Hub")
    print("=" * 60)
    print()

    model_name = "manelalab/chrono-gpt-instruct-v1-20241231"

    print(f"Model: {model_name}")
    print("Cutoff date: December 31, 2024")
    print("(First run downloads model weights)")
    print()

    # Download and import the custom model code
    print("Downloading model code...")
    chronogpt = download_model_code(model_name)

    # Load the model using the custom class
    print("Loading model weights...")
    model = chronogpt.ChronoGPT.from_pretrained(model_name)

    # Determine device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    model = model.to(device)
    model.eval()

    # Count parameters
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model loaded on device: {device}")
    print(f"Parameters: {num_params:,}")
    print()

    # Load tokenizer (ChronoGPT uses GPT-2 tokenizer)
    tokenizer = tiktoken.get_encoding("gpt2")

    # Test prompt
    prompt = "What is quantitative easing and when has it been used?"
    print(f"Prompt: {prompt}")
    print()

    # Generate response using ChronoGPT's custom function
    print("Generating response...")
    print()

    response = chronogpt.extract_response(
        input_text=prompt,
        model=model,
        tokenizer=tokenizer,
        device=device,
        max_tokens=200,
        temperature=0.7,
    )

    print(f"Response:\n{response}")
    print()
    print("=" * 60)
    print("Note: This response only includes information available up to Dec 2024.")


if __name__ == "__main__":
    main()
