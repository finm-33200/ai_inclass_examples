"""
08_web_search/openai_web_search.py

Use OpenAI's built-in web search to ground responses with current information.
Uses the Responses API which supports web search natively.
"""

from pathlib import Path

from decouple import Config, RepositoryEnv
from openai import OpenAI


def main():
    # Load API key from .env at repository root
    repo_root = Path(__file__).resolve().parents[2]
    config = Config(RepositoryEnv(repo_root / ".env"))

    client = OpenAI(api_key=config("OPENAI_API_KEY"))

    # Finance question that benefits from current information
    question = "What is the current Federal Reserve interest rate target, and what are market expectations for the next Fed meeting?"

    print(f"Question: {question}\n")
    print("Searching the web and generating response...\n")

    # Use the Responses API with web search
    response = client.responses.create(
        model="gpt-4o-mini",
        tools=[{"type": "web_search_preview"}],
        input=question,
    )

    # Print the main response
    print("Answer:")
    print(response.output_text)
    print()

    # Show what was searched and sources
    print("--- Search Details ---")
    for item in response.output:
        if hasattr(item, "type"):
            if item.type == "web_search_call":
                print(f"Query: {item.query if hasattr(item, 'query') else 'N/A'}")


def example_without_search():
    """Compare: what happens without web search."""
    repo_root = Path(__file__).resolve().parents[2]
    config = Config(RepositoryEnv(repo_root / ".env"))
    client = OpenAI(api_key=config("OPENAI_API_KEY"))

    question = "What is the current Federal Reserve interest rate target?"

    print("\n" + "=" * 60)
    print("COMPARISON: Without web search")
    print("=" * 60 + "\n")

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": question}],
    )

    print("Answer (may be outdated):")
    print(response.choices[0].message.content)


if __name__ == "__main__":
    main()
    # Uncomment to see the difference without search:
    # example_without_search()
