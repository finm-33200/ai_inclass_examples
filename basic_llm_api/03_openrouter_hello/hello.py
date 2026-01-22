"""
03_openrouter_hello/hello.py

Basic OpenRouter API call using the OpenAI SDK.
OpenRouter provides access to many models through a unified API.
"""

from pathlib import Path

from decouple import Config, RepositoryEnv
from openai import OpenAI


def main():
    # Load API key from .env at repository root
    repo_root = Path(__file__).resolve().parents[2]
    config = Config(RepositoryEnv(repo_root / ".env"))

    # Create client with OpenRouter base URL
    # OpenRouter uses the same API format as OpenAI
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=config("OPENROUTER_API_KEY"),
    )

    # Define the conversation (same format as OpenAI)
    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant. Keep responses concise.",
        },
        {
            "role": "user",
            "content": "What is an LLM? Explain in 2-3 sentences.",
        },
    ]

    print(f"User: {messages[-1]['content']}\n")

    # Make the API call
    # Model format: provider/model-name
    # Using a free model for this example
    response = client.chat.completions.create(
        model="google/gemini-2.0-flash-exp:free",
        messages=messages,
    )

    # Extract and print the response
    assistant_message = response.choices[0].message.content
    print(f"Assistant:\n{assistant_message}")

    # Show model used and token usage
    print(f"\n--- Details ---")
    print(f"Model: {response.model}")
    if response.usage:
        print(f"Total tokens: {response.usage.total_tokens}")


if __name__ == "__main__":
    main()
