"""
01_gemini_hello/hello.py

The simplest possible LLM API call using Google's Gemini model.
This demonstrates the basic pattern: configure client, send prompt, get response.
"""

from pathlib import Path

from decouple import Config, RepositoryEnv
from google import genai


def main():
    # Load API key from .env at repository root
    # Path: hello.py -> 01_gemini_hello -> basic_llm_api -> ai_inclass_examples
    repo_root = Path(__file__).resolve().parents[2]
    config = Config(RepositoryEnv(repo_root / ".env"))
    api_key = config("GOOGLE_API_KEY")

    # Create a Gemini client
    client = genai.Client(api_key=api_key)

    # Define your prompt
    prompt = "What is an LLM? Explain in 2-3 sentences."

    # Make the API call
    # gemini-2.0-flash-lite has the most generous free tier quota
    # Other options: gemini-2.0-flash, gemini-2.5-flash, gemini-2.5-pro
    print(f"Prompt: {prompt}\n")
    response = client.models.generate_content(
        model="gemini-2.0-flash-lite",
        contents=prompt,
    )

    # Print the response
    print(f"Response:\n{response.text}")


if __name__ == "__main__":
    main()
