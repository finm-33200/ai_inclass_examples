"""
06_function_calling/simple_tools.py

Basic function calling example.
The LLM decides when to call tools to answer questions.
"""

import json
from pathlib import Path

from decouple import Config, RepositoryEnv
from openai import OpenAI


# Define the tools the LLM can use
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_current_time",
            "description": "Get the current date and time",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": "Perform a mathematical calculation",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "Math expression to evaluate (e.g., '2 + 2', '100 * 1.05')",
                    }
                },
                "required": ["expression"],
            },
        },
    },
]


def get_current_time() -> str:
    """Return the current date and time."""
    from datetime import datetime

    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def calculate(expression: str) -> str:
    """Safely evaluate a math expression."""
    # Only allow safe characters for math
    allowed = set("0123456789+-*/.(). ")
    if not all(c in allowed for c in expression):
        return "Error: Invalid characters in expression"
    try:
        result = eval(expression)
        return str(result)
    except Exception as e:
        return f"Error: {e}"


def execute_tool(name: str, arguments: dict) -> str:
    """Execute a tool by name with the given arguments."""
    if name == "get_current_time":
        return get_current_time()
    elif name == "calculate":
        return calculate(arguments["expression"])
    else:
        return f"Unknown tool: {name}"


def main():
    # Load API key from .env at repository root
    repo_root = Path(__file__).resolve().parents[2]
    config = Config(RepositoryEnv(repo_root / ".env"))

    client = OpenAI(api_key=config("OPENAI_API_KEY"))

    # User question that requires tool use
    user_question = "What time is it, and what is 15% of 850?"

    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant. Use the available tools to answer questions accurately.",
        },
        {"role": "user", "content": user_question},
    ]

    print(f"User: {user_question}\n")

    # Tool use loop
    while True:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            tools=TOOLS,
        )

        message = response.choices[0].message
        finish_reason = response.choices[0].finish_reason

        # Check if the model wants to call tools
        if message.tool_calls:
            # Add the assistant's message to history
            messages.append(message)

            print("Tool calls requested:")
            for tool_call in message.tool_calls:
                name = tool_call.function.name
                args = json.loads(tool_call.function.arguments)
                print(f"  - {name}({args})")

                # Execute the tool
                result = execute_tool(name, args)
                print(f"    Result: {result}")

                # Add tool result to messages
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": result,
                    }
                )
            print()
        else:
            # No more tool calls - we have the final answer
            print(f"Assistant: {message.content}")
            break


if __name__ == "__main__":
    main()
