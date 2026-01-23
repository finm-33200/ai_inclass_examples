# 02 - Promptfoo Introduction

Learn declarative LLM evaluation using promptfoo's YAML-based configuration.

## What You'll Learn

- Defining prompts with variable placeholders
- Writing test cases with assertions
- Using built-in assertion types (contains, regex, llm-rubric)
- Comparing models side-by-side

## Why Promptfoo?

| Manual Python | Promptfoo |
|---------------|-----------|
| Write timing/logging code | Built-in metrics |
| Custom assertion logic | Declarative assertions |
| Build your own UI | Web viewer included |
| Manual test management | YAML-based test suites |

## Setup

Promptfoo is a Node.js CLI tool (no Python package needed):

```bash
# Verify Node.js is installed
node --version  # Should be 18+

# No install needed - npx runs it directly
```

## How to Run

Promptfoo needs the `OPENAI_API_KEY` environment variable. Load it from the repo root `.env`:

```bash
# Option 1: Source the env file (bash/zsh)
set -a && source ../../.env && set +a
npx promptfoo@latest eval

# Option 2: Use the helper script
python run_eval.py

# View results in browser
npx promptfoo@latest view
```

## Configuration Explained

### Prompts

```yaml
prompts:
  - |
    You are a financial analyst. Answer: {{question}}
```

The `{{question}}` placeholder is replaced with test variables.

### Providers

```yaml
providers:
  - openai:gpt-4o-mini
  - openai:gpt-4o
```

Promptfoo supports 50+ providers. See [docs](https://www.promptfoo.dev/docs/providers/).

### Assertions

```yaml
tests:
  - vars:
      question: "What is a bond's current yield?"
    assert:
      - type: contains        # Must contain string
        value: "coupon"
      - type: contains-any    # Must contain at least one
        value: ["price", "market value"]
      - type: llm-rubric      # LLM grades the response
        value: "Response correctly explains the formula"
```

## Assertion Types

| Type | Description | Example |
|------|-------------|---------|
| `contains` | String must appear | `"coupon"` |
| `contains-any` | At least one must appear | `["price", "value"]` |
| `regex` | Pattern must match | `"\\d+%"` |
| `llm-rubric` | LLM grades against criteria | `"Response is accurate"` |
| `cost` | Max cost threshold | `0.01` |
| `latency` | Max latency in ms | `5000` |

## Files

| File | Purpose |
|------|---------|
| `promptfooconfig.yaml` | Main configuration |
| `results/` | Output JSON (auto-generated) |

## Try It

1. Add a new test case for "What is beta in CAPM?"
2. Add `gpt-3.5-turbo` as a third provider
3. Try the `--no-cache` flag to force fresh API calls
4. Add a `cost` assertion to fail if cost exceeds $0.01

## Resources

- [Promptfoo Docs](https://www.promptfoo.dev/docs/intro/)
- [Assertion Reference](https://www.promptfoo.dev/docs/configuration/expected-outputs/)
- [Provider List](https://www.promptfoo.dev/docs/providers/)
