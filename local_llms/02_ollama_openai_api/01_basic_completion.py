"""
01_basic_completion.py

Use the OpenAI SDK to interact with local Ollama models.
This demonstrates the compatibility layer that makes switching between
cloud and local models seamless.
"""

from openai import OpenAI


def main():
    # Create client pointing to Ollama's OpenAI-compatible endpoint
    # No API key needed for local models
    client = OpenAI(
        base_url="http://localhost:11434/v1",
        api_key="ollama",  # Required by SDK but not used by Ollama
    )

    print("OpenAI SDK with Local Ollama Model")
    print("=" * 50)
    print()

    # Standard chat completion - identical to OpenAI API
    prompt = "What is quantitative finance? Explain in 2-3 sentences."
    print(f"Prompt: {prompt}\n")

    response = client.chat.completions.create(
        model="tinyllama:1.1b",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.7,
        max_tokens=200,
    )

    print(f"Response: {response.choices[0].message.content}")
    print()

    # Show the response metadata
    print("Response Metadata:")
    print(f"  Model: {response.model}")
    print(f"  Finish reason: {response.choices[0].finish_reason}")
    if response.usage:
        print(f"  Tokens - Prompt: {response.usage.prompt_tokens}, "
              f"Completion: {response.usage.completion_tokens}")


if __name__ == "__main__":
    main()
