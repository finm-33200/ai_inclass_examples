# 02 - Agent with Memory

This example adds **conversation memory** to our agent. The agent remembers what you talked about and can reference earlier context.

## Why Memory Matters

Without memory, every question is isolated. With memory:
- "What's my AAPL position?" → agent looks it up
- "What if it goes up 10%?" → agent knows you mean AAPL

## What You'll Learn

1. **Memory is just messages**: The conversation history IS the memory
2. **Context accumulation**: Each turn adds to what the LLM sees
3. **Stateful agents**: The agent "remembers" by keeping all messages
4. **Memory limits**: Eventually you hit context window limits

## The Key Insight

```python
class Agent:
    def __init__(self):
        self.messages = [system_prompt]  # This IS the memory

    def chat(self, user_message):
        self.messages.append({"role": "user", "content": user_message})
        # LLM sees ALL previous messages
        response = llm.chat(messages=self.messages)
        self.messages.append({"role": "assistant", "content": response})
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run

```bash
python memory_agent.py
```

## Example Conversation

```
You: What's my portfolio worth?
  [Tool] get_portfolio_value({}) -> {"holdings": [...], "total_value": 25256.0}

Agent: Your portfolio is currently worth **$25,256.00**.
  [Memory: 5 messages]

You: How much AAPL do I have?

Agent: You have **50 shares** of Apple (AAPL) in your portfolio.
  [Memory: 7 messages]

You: What if AAPL goes up 10%? How much would my AAPL position be worth?
  [Tool] calculate({'expression': '8925 * 1.1'}) -> {"result": 9817.5}

Agent: If Apple (AAPL) goes up by 10%, your position would be worth **$9,817.50**.
  [Memory: 11 messages]
```

Notice:
- The agent remembered portfolio details from question 1
- It knew "AAPL position" meant the 50 shares worth $8,925
- Memory grew from 5 → 7 → 11 messages

## Memory Growth

```
Turn 1: system + user + tool_calls + tool_results + assistant = 5 messages
Turn 2: + user + assistant = 7 messages
Turn 3: + user + tool_calls + tool_results + assistant = 11 messages
```

Each turn adds to the context. Eventually you hit limits:
- GPT-4o-mini: 128K tokens
- Llama 3.1 8B: 128K tokens
- Claude Haiku: 200K tokens

## Finance Application

Portfolio assistants, research chatbots, and trading copilots all need memory:
- "Buy more of what I bought yesterday"
- "How does this compare to the analysis you did earlier?"
- "Update the position we discussed"

## Try It

1. **Watch memory grow**: Type `memory` to see the conversation history
2. **Test context**: Ask about "the first stock" without naming it
3. **Add more tools**: Create `add_to_portfolio(ticker, shares)` tool
4. **Break context**: Have a very long conversation - what happens?
