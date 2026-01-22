"""
03_openrouter_hello/compare_simple.py

Run the same prompt across multiple models to compare responses.
Useful for understanding model differences and selecting the right one.
"""

import time
from pathlib import Path

from decouple import Config, RepositoryEnv
from openai import OpenAI


# Models to compare (mix of free and paid, different providers)
MODELS = [
    "google/gemini-2.0-flash-exp:free",  # Free Google model
    "qwen/qwen3-coder:free",  # Free Qwen model
    "openai/gpt-4o-mini",  # Paid OpenAI model (cheap)
]


def call_model(client: OpenAI, model: str, prompt: str) -> tuple[str, float]:
    """Call a model and return the response with elapsed time."""
    messages = [
        {"role": "system", "content": "You are a helpful assistant. Be concise."},
        {"role": "user", "content": prompt},
    ]

    start_time = time.time()
    response = client.chat.completions.create(
        model=model,
        messages=messages,
    )
    elapsed = time.time() - start_time

    return response.choices[0].message.content, elapsed


def main():
    # Load API key from .env at repository root
    repo_root = Path(__file__).resolve().parents[2]
    config = Config(RepositoryEnv(repo_root / ".env"))

    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=config("OPENROUTER_API_KEY"),
    )

    prompt = "What is the difference between stocks and bonds? Answer in 2 sentences."

    print("=" * 80)
    print(f"Prompt: {prompt}")
    print("=" * 80)

    for model in MODELS:
        print(f"\n--- {model} ---")
        try:
            response, elapsed = call_model(client, model, prompt)
            print(f"Time: {elapsed:.2f}s")
            print(f"Response:\n{response}")
        except Exception as e:
            print(f"Error: {e}")

    print("\n" + "=" * 80)
    print("Compare: response quality, length, and latency across models")


if __name__ == "__main__":
    main()
