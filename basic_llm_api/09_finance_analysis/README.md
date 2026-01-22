# 09 - Finance Analysis

Combine multiple LLM techniques for realistic finance workflows.

## Overview

This example brings together:
- Structured outputs for reliable data extraction
- Text analysis and summarization
- Financial metric extraction

Applied to real-world tasks like earnings call analysis and market briefings.

## What You'll Learn

- Analyzing earnings call transcripts
- Extracting structured financial metrics
- Generating investment summaries
- Combining LLM capabilities for complex workflows

## Setup

```bash
cd 09_finance_analysis
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Examples

### Earnings Call Analyzer
```bash
python earnings_analyzer.py
```

### Market Briefing Generator
```bash
python market_briefing.py
```

## Key Concepts

### Earnings Call Analysis

Earnings calls contain valuable information but are lengthy and unstructured. LLMs can:
- Extract key metrics (revenue, EPS, guidance)
- Identify management tone and confidence
- Highlight risks and opportunities
- Summarize Q&A highlights

### Structured Extraction

Use Pydantic schemas to ensure consistent output:

```python
class EarningsAnalysis(BaseModel):
    ticker: str
    quarter: str
    revenue: float | None
    eps: float | None
    guidance: str | None
    sentiment: Literal["positive", "negative", "neutral"]
    key_themes: list[str]
    risks: list[str]
```

### Multi-Pass Analysis

For complex documents:
1. First pass: Extract structured data
2. Second pass: Generate narrative summary
3. Third pass: Create actionable insights

## Sample Data

The `data/` directory contains sample earnings call excerpts from public sources.

## Try It

1. Add your own earnings transcript excerpt
2. Modify the extraction schema for your use case
3. Try chaining analyses (e.g., compare across quarters)
4. Generate different output formats (bullet points vs. narrative)
