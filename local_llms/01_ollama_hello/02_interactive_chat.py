"""
02_interactive_chat.py

Demonstrates using the Python ollama package to interact with local models.
This provides programmatic access to the same functionality as `ollama run`.
"""

import ollama


def single_prompt():
    """Send a single prompt and get a response."""
    print("=== Single Prompt Example ===\n")

    prompt = "What is the capital of France? Answer in one sentence."
    print(f"Prompt: {prompt}\n")

    response = ollama.chat(
        model="tinyllama:1.1b",
        messages=[{"role": "user", "content": prompt}],
    )

    print(f"Response: {response['message']['content']}\n")


def multi_turn_conversation():
    """Demonstrate a multi-turn conversation with context."""
    print("=== Multi-Turn Conversation ===\n")

    messages = []

    # First turn
    user_msg1 = "My name is Alex. What's a good programming language for beginners?"
    messages.append({"role": "user", "content": user_msg1})
    print(f"User: {user_msg1}")

    response1 = ollama.chat(model="tinyllama:1.1b", messages=messages)
    assistant_msg1 = response1["message"]["content"]
    messages.append({"role": "assistant", "content": assistant_msg1})
    print(f"Assistant: {assistant_msg1}\n")

    # Second turn - model should remember the name
    user_msg2 = "What's my name? And why would that language be good for finance?"
    messages.append({"role": "user", "content": user_msg2})
    print(f"User: {user_msg2}")

    response2 = ollama.chat(model="tinyllama:1.1b", messages=messages)
    assistant_msg2 = response2["message"]["content"]
    print(f"Assistant: {assistant_msg2}\n")


def streaming_response():
    """Demonstrate streaming tokens as they're generated."""
    print("=== Streaming Response ===\n")

    prompt = "Write a haiku about coding."
    print(f"Prompt: {prompt}\n")
    print("Response: ", end="", flush=True)

    stream = ollama.chat(
        model="tinyllama:1.1b",
        messages=[{"role": "user", "content": prompt}],
        stream=True,
    )

    for chunk in stream:
        print(chunk["message"]["content"], end="", flush=True)

    print("\n")


def main():
    print("Local LLM Chat with Ollama Python Package")
    print("=" * 50)
    print()

    # Check if model is available
    try:
        models = ollama.list()
        # Handle both old dict format and new object format
        if hasattr(models, "models"):
            model_names = [m.model for m in models.models]
        else:
            model_names = [m.get("name", m.get("model", "")) for m in models.get("models", [])]

        if not any("tinyllama" in name for name in model_names):
            print("TinyLlama not found. Pulling model...")
            ollama.pull("tinyllama:1.1b")
            print("Model pulled successfully!\n")
    except Exception as e:
        print(f"Error connecting to Ollama: {e}")
        print("Make sure Ollama is running: ollama serve")
        return

    single_prompt()
    multi_turn_conversation()
    streaming_response()

    print("=" * 50)
    print("All examples completed successfully!")


if __name__ == "__main__":
    main()
