"""
08_web_search/openrouter_web_search.py

Use OpenRouter's built-in web search to ground responses with current information.
Append :online to any model slug or use the web plugin for real-time data.
"""

from pathlib import Path

from decouple import Config, RepositoryEnv
from openai import OpenAI


def main():
    # Load API key from .env at repository root
    repo_root = Path(__file__).resolve().parents[2]
    config = Config(RepositoryEnv(repo_root / ".env"))

    # OpenRouter uses OpenAI-compatible API
    client = OpenAI(
        api_key=config("OPENROUTER_API_KEY"),
        base_url="https://openrouter.ai/api/v1",
    )

    # Finance question that benefits from current information
    question = "What is the current Federal Reserve interest rate target, and what are market expectations for the next Fed meeting?"

    print(f"Question: {question}\n")
    print("Searching the web and generating response...\n")

    # Method 1: Use :online suffix on model name
    # This automatically enables web search for any model
    response = client.chat.completions.create(
        model="openai/gpt-4o-mini:online",
        messages=[{"role": "user", "content": question}],
        extra_body={
            # Optional: customize web search behavior
            "plugins": [
                {
                    "id": "web",
                    "max_results": 5,
                }
            ]
        },
    )

    print("Answer:")
    print(response.choices[0].message.content)
    print()

    # Show any annotations/citations if available
    message = response.choices[0].message
    if hasattr(message, "annotations") and message.annotations:
        print("--- Sources ---")
        for annotation in message.annotations:
            if hasattr(annotation, "url"):
                print(f"- {annotation.url}")


def example_with_anthropic():
    """Example using Claude with web search via OpenRouter."""
    repo_root = Path(__file__).resolve().parents[2]
    config = Config(RepositoryEnv(repo_root / ".env"))

    client = OpenAI(
        api_key=config("OPENROUTER_API_KEY"),
        base_url="https://openrouter.ai/api/v1",
    )

    question = "What were the key announcements from the most recent Federal Reserve meeting?"

    print("\n" + "=" * 60)
    print("Using Claude with web search")
    print("=" * 60 + "\n")

    response = client.chat.completions.create(
        model="anthropic/claude-sonnet-4:online",
        messages=[{"role": "user", "content": question}],
    )

    print("Answer:")
    print(response.choices[0].message.content)


if __name__ == "__main__":
    main()
    # Uncomment to try with Claude:
    # example_with_anthropic()
