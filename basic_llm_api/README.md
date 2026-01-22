# Basic LLM API Examples

This directory contains progressive examples for making API calls to Large Language Models (LLMs). Examples progress from simple "hello world" calls to advanced capabilities like vision, web search, and structured outputs.

## Prerequisites

- Python 3.10+
- API keys (see setup below)

## API Key Setup

Copy the root `.env.example` to `.env` and fill in your keys:

| Key | Provider | How to Get |
|-----|----------|------------|
| `GOOGLE_API_KEY` | Google AI Studio | Free at [aistudio.google.com](https://aistudio.google.com/) |
| `OPENAI_API_KEY` | OpenAI | [platform.openai.com](https://platform.openai.com/) |
| `OPENROUTER_API_KEY` | OpenRouter | [openrouter.ai](https://openrouter.ai/) |

## Examples

| # | Directory | Description | Key Concepts |
|---|-----------|-------------|--------------|
| 01 | `01_gemini_hello/` | Simplest LLM call (FREE tier) | API keys, basic prompts |
| 02 | `02_openai_hello/` | OpenAI API basics | Chat completions, message roles |
| 03 | `03_openrouter_hello/` | Model flexibility via OpenRouter | Model selection, cost awareness |
| 04 | `04_streaming_responses/` | Stream tokens for responsive UX | Generators, incremental output |
| 05 | `05_structured_outputs/` | Get reliable JSON from LLMs | JSON mode, Pydantic schemas |
| 06 | `06_function_calling/` | LLM-driven tool use | Tool definitions, execution |
| 07 | `07_vision_ocr/` | Image understanding | Multimodal inputs, base64 |
| 08 | `08_web_search/` | Ground responses with web data | Citations, current information |
| 09 | `09_finance_analysis/` | Realistic finance workflow | Combined techniques |
| 10 | `10_model_comparison/` | Benchmark models systematically | Evaluation, cost/quality |

## Rationale for Ordering

1. **Gemini first**: Free API tier via Google AI Studio - no credit card needed, lowest barrier to entry for students to start experimenting immediately.

2. **OpenAI second**: Industry standard API that most tooling and documentation is built around. Required for the course.

3. **OpenRouter third**: Aggregator service that provides access to many models (OpenAI, Anthropic, open-source) through a unified API. Enables model comparison and cost optimization.

4. **Streaming & Structured outputs**: Essential patterns for production applications - responsive UX and reliable data extraction.

5. **Function calling**: Foundation for agentic AI - letting the LLM decide when to use external tools.

6. **Vision & Web search**: Multimodal capabilities that extend beyond text.

7. **Finance analysis & Model comparison**: Capstone examples combining multiple techniques for realistic finance use cases.

## Running Examples

Each example is self-contained. Navigate to the subdirectory and follow its README:

```bash
cd 01_gemini_hello
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
python hello.py
```

## Resources

- [OpenAI Platform Docs](https://platform.openai.com/docs/overview)
- [OpenAI Cookbook](https://github.com/openai/openai-cookbook)
- [Google AI Studio](https://aistudio.google.com/)
- [OpenRouter](https://openrouter.ai/)
