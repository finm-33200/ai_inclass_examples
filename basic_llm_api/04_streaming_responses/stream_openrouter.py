"""
04_streaming_responses/stream_openrouter.py

Stream responses via OpenRouter.
Same pattern works across all OpenRouter models.
"""

from pathlib import Path

from decouple import Config, RepositoryEnv
from openai import OpenAI


def main():
    # Load API key from .env at repository root
    repo_root = Path(__file__).resolve().parents[2]
    config = Config(RepositoryEnv(repo_root / ".env"))

    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=config("OPENROUTER_API_KEY"),
    )

    messages = [
        {
            "role": "system",
            "content": "You are a finance expert. Provide clear, informative responses.",
        },
        {
            "role": "user",
            "content": "Explain what a yield curve inversion means and why investors watch it.",
        },
    ]

    print(f"User: {messages[-1]['content']}\n")
    print("Assistant: ", end="", flush=True)

    # Make streaming API call with OpenRouter
    response = client.chat.completions.create(
        model="google/gemini-2.0-flash-exp:free",
        messages=messages,
        stream=True,
    )

    # Process each chunk as it arrives
    for chunk in response:
        if chunk.choices[0].delta.content:
            print(chunk.choices[0].delta.content, end="", flush=True)

    print("\n")


if __name__ == "__main__":
    main()
