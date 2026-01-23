# 03 - LLM as Judge

Learn automated quality evaluation using the LLM-as-judge pattern.

## What You'll Learn

- How LLM-as-judge works (one model grades another)
- Writing effective grading rubrics
- Trade-offs: cost, consistency, and correlation with human judgment
- Using llm-rubric in promptfoo vs manual implementation

## The LLM-as-Judge Pattern

```
┌─────────────┐     question      ┌─────────────┐
│   Test      │ ───────────────── │   Model     │
│   Input     │                   │   (GPT-4o   │
└─────────────┘                   │   Mini)     │
                                  └──────┬──────┘
                                         │ response
                                         ▼
                              ┌─────────────────────┐
                              │   Judge Model       │
                              │   (GPT-4o)          │
                              │                     │
                              │   "Grade this       │
                              │    response 1-5"    │
                              └──────────┬──────────┘
                                         │
                                         ▼
                              ┌─────────────────────┐
                              │   Score + Reasoning │
                              │   {"score": 4,      │
                              │    "reasoning":...} │
                              └─────────────────────┘
```

## Files

| File | Purpose |
|------|---------|
| `judge.py` | Manual implementation showing the mechanics |
| `promptfooconfig.yaml` | Same evaluation using promptfoo's llm-rubric |

## How to Run

### Manual Implementation

```bash
pip install -r requirements.txt
python judge.py
```

### Promptfoo Version

```bash
npx promptfoo@latest eval
npx promptfoo@latest view
```

## Writing Good Rubrics

### Bad Rubric

```
"Is this response good?"
```

### Good Rubric

```
ACCURACY: Is the formula correct (current yield = annual coupon / current price)?
COMPLETENESS: Does it explain what each term means?
Score 1-5 where 5 is completely accurate and complete.
```

### Finance-Specific Rubrics

| Dimension | What to Check |
|-----------|---------------|
| **Accuracy** | Formulas correct? Concepts right? |
| **Completeness** | All parts answered? Caveats mentioned? |
| **Risk Disclosure** | Risks mentioned? Disclaimers included? |
| **Calculation** | Math correct? Steps shown? |

## Trade-offs

### Cost

Each judgment = 1 additional API call

```
Total cost = (model calls) + (judge calls)
           = N tests × M models + N tests × M models × R rubrics
```

For 10 tests, 2 models, 3 rubrics:
- Model calls: 20
- Judge calls: 60
- **Total: 80 API calls**

### Consistency

- LLM judges have variance (~0.5-1 point on 5-point scale)
- Use specific rubrics to reduce variance
- Consider multiple judge runs for high-stakes evals

### Human Correlation

Research shows LLM-as-judge correlates ~0.7-0.9 with human judgment for:
- Factual accuracy
- Helpfulness
- Harmlessness

Lower correlation for:
- Creativity
- Domain expertise (without fine-tuning)

## Try It

1. Modify `judge.py` to use a cheaper judge model (gpt-4o-mini)
2. Compare judge scores between gpt-4o and gpt-4o-mini as judge
3. Add a new rubric for "clarity" or "conciseness"
4. Run the same test 3 times and measure judge variance

## Resources

- [LLM-as-Judge Paper](https://arxiv.org/abs/2306.05685) (MT-Bench)
- [Promptfoo LLM Rubric Docs](https://www.promptfoo.dev/docs/configuration/expected-outputs/model-graded/)
- [G-Eval Paper](https://arxiv.org/abs/2303.16634)
