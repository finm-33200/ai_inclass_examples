"""
Simplest possible web chatbot using Gradio and OpenAI.

Run:
    python app.py

Then open http://127.0.0.1:7860 in your browser.
"""

from pathlib import Path

import gradio as gr
from decouple import Config, RepositoryEnv
from openai import OpenAI

repo_root = Path(__file__).resolve().parents[2]
config = Config(RepositoryEnv(repo_root / ".env"))

client = OpenAI(api_key=config("OPENAI_API_KEY"))


def chat(message, history):
    messages = [{"role": h["role"], "content": h["content"]} for h in history]
    messages.append({"role": "user", "content": message})
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
    )
    return response.choices[0].message.content


gr.ChatInterface(chat, title="Simple Chatbot").launch()
