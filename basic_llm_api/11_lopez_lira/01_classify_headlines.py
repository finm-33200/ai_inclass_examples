"""
11_lopez_lira/01_classify_headlines.py

Basic sentiment classification following Lopez-Lira (2023).
Classifies financial headlines as "good news", "bad news", or "uncertain".

This is the simplest version - no structured outputs, just raw text responses.
"""

from pathlib import Path

from decouple import Config, RepositoryEnv
from openai import OpenAI


def main():
    # Load API key from .env at repository root
    repo_root = Path(__file__).resolve().parents[2]
    config = Config(RepositoryEnv(repo_root / ".env"))

    client = OpenAI(api_key=config("OPENAI_API_KEY"))

    # Sample headlines to classify
    headlines = [
        "Apple Reports Record Q4 Earnings, Beats Analyst Expectations",
        "Fed Signals More Rate Hikes Ahead, Markets Tumble",
        "Tesla Stock Plunges After Disappointing Delivery Numbers",
        "Microsoft Cloud Revenue Grows 27%, Exceeding Forecasts",
        "Bank of America Warns of Recession Risk in 2024",
    ]

    # System prompt following Lopez-Lira methodology
    system_prompt = """You are a financial expert. For each headline, determine whether
it represents 'good news', 'bad news', or 'uncertain' for the mentioned company's
stock price.

Consider:
- Market expectations (beating vs missing estimates)
- Typical investor reactions to similar news
- The broader market context implied

Respond with ONLY one of: 'good news', 'bad news', or 'uncertain'"""

    print("Lopez-Lira Sentiment Classification")
    print("=" * 60)
    print()

    for headline in headlines:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Headline: {headline}"},
            ],
            temperature=0,  # Deterministic output for reproducibility
        )

        classification = response.choices[0].message.content.strip().lower()
        print(f"{classification:12} | {headline[:55]}...")
        print()

    # Show token usage for the last request
    print("-" * 60)
    print(f"Last request tokens: {response.usage.total_tokens}")


if __name__ == "__main__":
    main()
