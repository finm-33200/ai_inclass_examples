"""
02_openai_hello/hello.py

Basic OpenAI API call using the chat completions endpoint.
This demonstrates the standard pattern used across most LLM tooling.
"""

from pathlib import Path

from decouple import Config, RepositoryEnv
from openai import OpenAI


def main():
    # Load API key from .env at repository root
    # Path: hello.py -> 02_openai_hello -> basic_llm_api -> ai_inclass_examples
    repo_root = Path(__file__).resolve().parents[2]
    config = Config(RepositoryEnv(repo_root / ".env"))

    # Create the OpenAI client
    client = OpenAI(api_key=config("OPENAI_API_KEY"))

    # Define the conversation
    # OpenAI uses a messages format with roles: system, user, assistant
    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant. Keep responses concise.",
        },
        {
            "role": "user",
            "content": "What is an LLM? Explain in 2-3 sentences.",
        },
    ]

    print(f"User: {messages[-1]['content']}\n")

    # Make the API call
    # gpt-4o-mini is fast and cost-effective for most tasks
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
    )

    # Extract and print the response
    assistant_message = response.choices[0].message.content
    print(f"Assistant:\n{assistant_message}")

    # Show token usage (helpful for understanding costs)
    print(f"\n--- Token Usage ---")
    print(f"Prompt tokens: {response.usage.prompt_tokens}")
    print(f"Completion tokens: {response.usage.completion_tokens}")
    print(f"Total tokens: {response.usage.total_tokens}")


if __name__ == "__main__":
    main()
