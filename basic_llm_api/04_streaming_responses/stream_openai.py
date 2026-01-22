"""
04_streaming_responses/stream_openai.py

Stream responses from OpenAI token-by-token.
Watch the response appear in real-time as the model generates it.
"""

from pathlib import Path

from decouple import Config, RepositoryEnv
from openai import OpenAI


def main():
    # Load API key from .env at repository root
    repo_root = Path(__file__).resolve().parents[2]
    config = Config(RepositoryEnv(repo_root / ".env"))

    client = OpenAI(api_key=config("OPENAI_API_KEY"))

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

    # Make streaming API call
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        stream=True,  # Enable streaming
    )

    # Process each chunk as it arrives
    for chunk in response:
        # Each chunk contains a delta (incremental content)
        if chunk.choices[0].delta.content:
            print(chunk.choices[0].delta.content, end="", flush=True)

    print("\n")  # Newline after streaming completes


if __name__ == "__main__":
    main()
