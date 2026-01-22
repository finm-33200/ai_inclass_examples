"""
03_openrouter_hello/list_models.py

List available models on OpenRouter with pricing information.
Useful for finding the right model for your use case and budget.
"""

from pathlib import Path

from decouple import Config, RepositoryEnv
import requests


def main():
    # Load API key from .env at repository root
    repo_root = Path(__file__).resolve().parents[2]
    config = Config(RepositoryEnv(repo_root / ".env"))
    api_key = config("OPENROUTER_API_KEY")

    # Fetch models from OpenRouter API
    response = requests.get(
        "https://openrouter.ai/api/v1/models",
        headers={"Authorization": f"Bearer {api_key}"},
    )
    response.raise_for_status()
    models = response.json()["data"]

    # Filter and sort by price (cheapest first)
    # Price is per million tokens
    models_with_price = [
        m for m in models if m.get("pricing", {}).get("prompt") is not None
    ]
    models_with_price.sort(key=lambda m: float(m["pricing"]["prompt"]))

    print("=" * 80)
    print("Available Models on OpenRouter (sorted by input price)")
    print("=" * 80)
    print(f"{'Model ID':<45} {'Input $/M':<12} {'Output $/M':<12} {'Context':<10}")
    print("-" * 80)

    # Show top 20 cheapest models
    for model in models_with_price[:20]:
        model_id = model["id"]
        input_price = float(model["pricing"]["prompt"]) * 1_000_000
        output_price = float(model["pricing"]["completion"]) * 1_000_000
        context = model.get("context_length", "N/A")

        # Mark free models
        price_str = f"${input_price:.4f}" if input_price > 0 else "FREE"
        output_str = f"${output_price:.4f}" if output_price > 0 else "FREE"

        print(f"{model_id:<45} {price_str:<12} {output_str:<12} {context:<10}")

    print("-" * 80)
    print(f"Total models available: {len(models)}")
    print("\nTip: Add ':free' suffix to model IDs for free tier access where available")


if __name__ == "__main__":
    main()
