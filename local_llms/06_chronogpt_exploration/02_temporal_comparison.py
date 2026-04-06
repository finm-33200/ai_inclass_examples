"""
02_temporal_comparison.py

Compare responses from ChronoGPT models with different cutoff dates.
This demonstrates temporal consistency - each model only knows information
available up to its training date.
"""

import gc
import torch
from huggingface_hub import hf_hub_download
import tiktoken
import importlib.util


def download_model_code(model_name):
    """Download the custom ChronoGPT model code."""
    model_code_path = hf_hub_download(model_name, "ChronoGPT_instruct.py")
    spec = importlib.util.spec_from_file_location("chronogpt", model_code_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_model(model_name, device):
    """Load a ChronoGPT model."""
    print(f"Loading {model_name}...")
    chronogpt = download_model_code(model_name)
    model = chronogpt.ChronoGPT.from_pretrained(model_name)
    model = model.to(device)
    model.eval()
    return model, chronogpt


def generate_response(model, chronogpt, tokenizer, device, prompt, max_tokens=150):
    """Generate a response from the model."""
    return chronogpt.extract_response(
        input_text=prompt,
        model=model,
        tokenizer=tokenizer,
        device=device,
        max_tokens=max_tokens,
        temperature=0.7,
    )


def main():
    print("ChronoGPT Temporal Comparison")
    print("=" * 70)
    print()
    print("Comparing responses from models with different knowledge cutoffs.")
    print("The 2000 model should NOT know about events after December 2000.")
    print()

    # Determine device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print(f"Device: {device}")
    print()

    # Load tokenizer (shared across all models)
    tokenizer = tiktoken.get_encoding("gpt2")

    # Models to compare
    models_to_compare = [
        ("manelalab/chrono-gpt-instruct-v1-20001231", "December 2000"),
        ("manelalab/chrono-gpt-instruct-v1-20241231", "December 2024"),
    ]

    # Questions that reveal temporal differences
    questions = [
        "What is Bitcoin and how does it work?",
        "Describe the current state of artificial intelligence research.",
    ]

    for question in questions:
        print("=" * 70)
        print(f"QUESTION: {question}")
        print("=" * 70)
        print()

        for model_name, cutoff_date in models_to_compare:
            print(f"--- Model cutoff: {cutoff_date} ---")

            model, chronogpt = load_model(model_name, device)
            response = generate_response(
                model, chronogpt, tokenizer, device, question
            )

            # Truncate response if too long
            response_lines = response.split("\n")[:5]
            response = "\n".join(response_lines)

            print(f"\n{response}\n")

            # Clean up memory
            del model, chronogpt
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        print()

    print("=" * 70)
    print("ANALYSIS")
    print("=" * 70)
    print()
    print("Notice how the 2000 model:")
    print("- Cannot describe Bitcoin (invented 2008)")
    print("- Describes AI as 'expert systems' or 'neural networks' (pre-deep learning)")
    print()
    print("This temporal consistency is crucial for:")
    print("- Backtesting trading strategies without look-ahead bias")
    print("- Historical analysis with period-appropriate context")
    print("- Simulating decisions based on information available at the time")


if __name__ == "__main__":
    main()
