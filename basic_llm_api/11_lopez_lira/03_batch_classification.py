"""
11_lopez_lira/03_batch_classification.py

Batch processing of headlines from CSV with error handling and cost tracking.
Suitable for processing larger datasets as required for HW1.
"""

import csv
import time
from pathlib import Path
from typing import Literal

from decouple import Config, RepositoryEnv
from openai import OpenAI, RateLimitError
from pydantic import BaseModel, Field


class LopezLiraSentiment(BaseModel):
    """Structured sentiment analysis following Lopez-Lira (2023)."""

    classification: Literal["good news", "bad news", "uncertain"] = Field(
        description="Sentiment classification for stock price impact"
    )
    confidence: float = Field(
        description="Confidence in the classification from 0.0 to 1.0",
        ge=0.0,
        le=1.0,
    )


def pydantic_to_strict_schema(model: type[BaseModel]) -> dict:
    """Convert Pydantic model to OpenAI strict-compatible schema."""
    schema = model.model_json_schema()
    schema["additionalProperties"] = False
    if "properties" in schema:
        schema["required"] = list(schema["properties"].keys())
    if "$defs" in schema:
        del schema["$defs"]
    return schema


def classify_with_retry(
    client: OpenAI,
    headline: str,
    max_retries: int = 3,
) -> tuple[LopezLiraSentiment, int]:
    """
    Classify a headline with exponential backoff retry on rate limits.
    Returns (result, tokens_used).
    """
    system_prompt = """You are a financial expert. Classify the headline as
'good news', 'bad news', or 'uncertain' for the company's stock price.
Consider market expectations and typical investor reactions."""

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Headline: {headline}"},
                ],
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "LopezLiraSentiment",
                        "strict": True,
                        "schema": pydantic_to_strict_schema(LopezLiraSentiment),
                    },
                },
                temperature=0,
            )

            result = LopezLiraSentiment.model_validate_json(
                response.choices[0].message.content
            )
            return result, response.usage.total_tokens

        except RateLimitError:
            if attempt < max_retries - 1:
                wait_time = 2 ** (attempt + 1)
                print(f"  Rate limited, waiting {wait_time}s...")
                time.sleep(wait_time)
            else:
                raise

    raise RuntimeError("Max retries exceeded")


def load_headlines(csv_path: Path) -> list[dict]:
    """Load headlines from CSV file."""
    headlines = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            headlines.append(row)
    return headlines


def save_results(results: list[dict], output_path: Path):
    """Save classification results to CSV."""
    if not results:
        return

    fieldnames = list(results[0].keys())
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)


def main():
    # Load API key from .env at repository root
    repo_root = Path(__file__).resolve().parents[2]
    config = Config(RepositoryEnv(repo_root / ".env"))

    client = OpenAI(api_key=config("OPENAI_API_KEY"))

    # Load headlines from CSV
    data_dir = Path(__file__).parent / "data"
    headlines_path = data_dir / "sample_headlines.csv"

    if not headlines_path.exists():
        print(f"Error: {headlines_path} not found")
        print("Please create the sample headlines file first.")
        return

    headlines = load_headlines(headlines_path)
    print(f"Loaded {len(headlines)} headlines from {headlines_path.name}")
    print()

    # Process each headline
    results = []
    total_tokens = 0
    classification_counts = {"good news": 0, "bad news": 0, "uncertain": 0}

    for i, row in enumerate(headlines, 1):
        headline = row["headline"]
        print(f"[{i}/{len(headlines)}] {headline[:50]}...", end=" ")

        try:
            result, tokens = classify_with_retry(client, headline)
            total_tokens += tokens

            classification_counts[result.classification] += 1
            print(f"→ {result.classification} ({result.confidence:.0%})")

            results.append({
                "headline": headline,
                "source": row.get("source", ""),
                "date": row.get("date", ""),
                "ticker": row.get("ticker", ""),
                "classification": result.classification,
                "confidence": result.confidence,
            })

        except Exception as e:
            print(f"→ ERROR: {e}")
            results.append({
                "headline": headline,
                "source": row.get("source", ""),
                "date": row.get("date", ""),
                "ticker": row.get("ticker", ""),
                "classification": "error",
                "confidence": 0.0,
            })

    # Save results
    output_path = data_dir / "classifications.csv"
    save_results(results, output_path)
    print()
    print(f"Results saved to {output_path}")

    # Summary statistics
    print()
    print("=" * 50)
    print("Summary")
    print("=" * 50)
    print(f"Total headlines processed: {len(headlines)}")
    print()
    print("Classification distribution:")
    for classification, count in classification_counts.items():
        pct = count / len(headlines) * 100 if headlines else 0
        print(f"  {classification:12}: {count:3} ({pct:.1f}%)")

    print()
    print(f"Total tokens used: {total_tokens:,}")

    # Estimate cost (GPT-4o-mini pricing as of 2024)
    # Input: $0.15/1M tokens, Output: $0.60/1M tokens
    # Assuming ~60% input, 40% output
    estimated_cost = (total_tokens * 0.6 * 0.15 + total_tokens * 0.4 * 0.60) / 1_000_000
    print(f"Estimated cost: ${estimated_cost:.4f}")


if __name__ == "__main__":
    main()
