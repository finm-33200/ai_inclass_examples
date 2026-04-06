"""
02_streaming.py

Demonstrate streaming responses with the OpenAI SDK and local Ollama.
Streaming shows tokens as they're generated, providing a more responsive experience.
"""

from openai import OpenAI


def main():
    client = OpenAI(
        base_url="http://localhost:11434/v1",
        api_key="ollama",
    )

    print("Streaming Tokens from Local LLM")
    print("=" * 50)
    print()

    prompt = "Explain the Black-Scholes model in simple terms."
    print(f"Prompt: {prompt}\n")
    print("Response (streaming): ", end="", flush=True)

    # Create streaming completion
    stream = client.chat.completions.create(
        model="tinyllama:1.1b",
        messages=[
            {"role": "system", "content": "You are a finance expert who explains concepts clearly."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.7,
        max_tokens=300,
        stream=True,  # Enable streaming
    )

    # Process tokens as they arrive
    for chunk in stream:
        if chunk.choices[0].delta.content:
            print(chunk.choices[0].delta.content, end="", flush=True)

    print("\n")
    print("=" * 50)
    print("Streaming complete!")


if __name__ == "__main__":
    main()
