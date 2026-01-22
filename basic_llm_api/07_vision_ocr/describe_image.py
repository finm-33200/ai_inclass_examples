"""
07_vision_ocr/describe_image.py

Basic image description using vision-capable LLMs.
Generates a sample chart and asks the model to describe it.
"""

import base64
import io
from pathlib import Path

from decouple import Config, RepositoryEnv
from openai import OpenAI

# Import matplotlib for generating sample images
import matplotlib.pyplot as plt
import numpy as np


def generate_sample_chart() -> bytes:
    """Generate a sample stock price chart as PNG bytes."""
    # Create sample data
    np.random.seed(42)
    dates = np.arange(100)
    prices = 100 + np.cumsum(np.random.randn(100) * 2)

    # Create the chart
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(dates, prices, "b-", linewidth=2)
    ax.fill_between(dates, prices, alpha=0.3)
    ax.set_title("ACME Corp Stock Price (2024)", fontsize=14)
    ax.set_xlabel("Trading Days")
    ax.set_ylabel("Price ($)")
    ax.grid(True, alpha=0.3)

    # Add some annotations
    max_idx = np.argmax(prices)
    min_idx = np.argmin(prices)
    ax.annotate(
        f"High: ${prices[max_idx]:.2f}",
        xy=(max_idx, prices[max_idx]),
        xytext=(max_idx + 5, prices[max_idx] + 5),
        fontsize=10,
    )
    ax.annotate(
        f"Low: ${prices[min_idx]:.2f}",
        xy=(min_idx, prices[min_idx]),
        xytext=(min_idx + 5, prices[min_idx] - 5),
        fontsize=10,
    )

    # Save to bytes
    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=100, bbox_inches="tight")
    plt.close()
    buf.seek(0)
    return buf.read()


def main():
    # Load API key from .env at repository root
    repo_root = Path(__file__).resolve().parents[2]
    config = Config(RepositoryEnv(repo_root / ".env"))

    client = OpenAI(api_key=config("OPENAI_API_KEY"))

    # Generate a sample chart
    print("Generating sample stock chart...")
    image_bytes = generate_sample_chart()

    # Save it for reference
    sample_path = Path(__file__).parent / "sample_images" / "stock_chart.png"
    sample_path.write_bytes(image_bytes)
    print(f"Saved to: {sample_path}\n")

    # Encode as base64
    b64_image = base64.b64encode(image_bytes).decode("utf-8")

    # Send to vision model
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "Describe this stock chart. What trends do you see? Is the stock trending up or down overall?",
                },
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{b64_image}"},
                },
            ],
        }
    ]

    print("Asking GPT-4o to analyze the chart...\n")

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        max_tokens=500,
    )

    print("Analysis:")
    print(response.choices[0].message.content)


if __name__ == "__main__":
    main()
