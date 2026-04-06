"""
03_finance_questions.py

Test ChronoGPT with finance-specific questions that reveal temporal knowledge.
These questions are designed to show how financial knowledge evolves over time.
"""

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
    chronogpt = download_model_code(model_name)
    model = chronogpt.ChronoGPT.from_pretrained(model_name)
    model = model.to(device)
    model.eval()
    return model, chronogpt


def generate_response(model, chronogpt, tokenizer, device, prompt, max_tokens=200):
    """Generate a response from the model."""
    return chronogpt.extract_response(
        input_text=prompt,
        model=model,
        tokenizer=tokenizer,
        device=device,
        max_tokens=max_tokens,
        temperature=0.7,
    )


# Finance questions designed to reveal temporal differences
FINANCE_QUESTIONS = [
    {
        "question": "What are the largest companies by market capitalization?",
        "insight": "Compare tech giants across eras (Microsoft in 2000 vs Apple/NVIDIA in 2024)",
    },
    {
        "question": "What is high-frequency trading and how prevalent is it?",
        "insight": "HFT became dominant after 2005; 2000 model may not know it well",
    },
    {
        "question": "Describe the role of the Federal Reserve in monetary policy.",
        "insight": "Post-2008 QE policies would be unknown to 2000 model",
    },
]


def main():
    print("ChronoGPT Finance Knowledge Test")
    print("=" * 70)
    print()

    # Determine device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    # Use the 2024 model for this demonstration
    model_name = "manelalab/chrono-gpt-instruct-v1-20241231"

    print(f"Model: {model_name}")
    print("Cutoff: December 2024")
    print(f"Device: {device}")
    print()

    print("Loading model...")
    model, chronogpt = load_model(model_name, device)

    # Count parameters
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {num_params:,}")
    print()

    # Load tokenizer
    tokenizer = tiktoken.get_encoding("gpt2")

    for i, item in enumerate(FINANCE_QUESTIONS, 1):
        print("=" * 70)
        print(f"Question {i}: {item['question']}")
        print(f"Insight: {item['insight']}")
        print("-" * 70)

        response = generate_response(
            model, chronogpt, tokenizer, device, item["question"]
        )

        # Truncate if too long
        response_lines = response.split("\n")[:8]
        response = "\n".join(response_lines)

        print(f"\nResponse:\n{response}")
        print()

    print("=" * 70)
    print("EXERCISE: Run this with chrono-gpt-instruct-v1-20001231")
    print("Compare how financial knowledge and advice differs across 24 years!")
    print("=" * 70)


if __name__ == "__main__":
    main()
