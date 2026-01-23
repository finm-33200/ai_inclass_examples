"""
03_llm_as_judge/judge.py

Manual implementation of the LLM-as-judge pattern for automated quality evaluation.
This shows how automated grading works before using it via promptfoo.
"""

import json
from pathlib import Path

from decouple import Config, RepositoryEnv
from openai import OpenAI


# Finance-specific grading rubrics
RUBRICS = {
    "accuracy": """
    Evaluate the ACCURACY of this financial explanation.
    - Are the formulas and calculations correct?
    - Are the concepts explained without errors?
    - Would a finance professional agree with this answer?
    """,
    "completeness": """
    Evaluate the COMPLETENESS of this response.
    - Does it answer all parts of the question?
    - Are key concepts and caveats mentioned?
    - Is anything important missing?
    """,
    "risk_disclosure": """
    Evaluate the RISK DISCLOSURE in this financial advice.
    - Does it mention relevant risks?
    - Are appropriate disclaimers included?
    - Would this be considered responsible financial communication?
    """,
}


def judge_response(
    client: OpenAI,
    question: str,
    response: str,
    rubric: str,
    judge_model: str = "gpt-4o",
) -> dict:
    """
    Use an LLM to grade another LLM's response.

    Returns a dict with:
    - score: 1-5 integer
    - reasoning: explanation for the score
    """
    prompt = f"""You are an expert financial analyst grading an AI assistant's response.

QUESTION ASKED:
{question}

AI RESPONSE TO GRADE:
{response}

GRADING CRITERIA:
{rubric}

Grade the response on a scale of 1-5:
- 1: Completely incorrect or missing
- 2: Major errors or significant gaps
- 3: Partially correct, some issues
- 4: Mostly correct, minor issues
- 5: Excellent, accurate and complete

Respond in JSON format:
{{"score": <1-5>, "reasoning": "<brief explanation>"}}
"""

    judge_response = client.chat.completions.create(
        model=judge_model,
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
        max_tokens=200,
    )

    return json.loads(judge_response.choices[0].message.content)


def get_model_response(client: OpenAI, model: str, question: str) -> str:
    """Get a response from the model being evaluated."""
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a financial analyst. Be concise."},
            {"role": "user", "content": question},
        ],
        max_tokens=300,
    )
    return response.choices[0].message.content


def main():
    # Load API key
    repo_root = Path(__file__).resolve().parents[2]
    config = Config(RepositoryEnv(repo_root / ".env"))
    client = OpenAI(api_key=config("OPENAI_API_KEY"))

    # Test questions
    questions = [
        "What is the current yield formula for a bond?",
        "Should I invest all my savings in a single stock?",
        "Explain the difference between systematic and unsystematic risk.",
    ]

    # Models to evaluate
    models_to_test = ["gpt-4o-mini", "gpt-4o"]

    print("=" * 70)
    print("LLM-AS-JUDGE EVALUATION")
    print("=" * 70)
    print(f"Judge model: gpt-4o")
    print(f"Models being evaluated: {models_to_test}")
    print(f"Rubrics: {list(RUBRICS.keys())}")
    print()

    results = []

    for question in questions:
        print(f"\n{'='*70}")
        print(f"QUESTION: {question[:60]}...")
        print("=" * 70)

        for model in models_to_test:
            print(f"\n--- {model} ---")

            # Get the model's response
            response = get_model_response(client, model, question)
            print(f"Response: {response[:100]}...")

            # Judge on each rubric
            model_scores = {}
            for rubric_name, rubric_text in RUBRICS.items():
                judgment = judge_response(
                    client, question, response, rubric_text
                )
                model_scores[rubric_name] = judgment["score"]
                print(f"  {rubric_name}: {judgment['score']}/5 - {judgment['reasoning'][:50]}...")

            results.append({
                "question": question,
                "model": model,
                "response": response,
                "scores": model_scores,
                "avg_score": sum(model_scores.values()) / len(model_scores),
            })

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"{'Model':<20} {'Avg Score':<12} {'Accuracy':<12} {'Completeness':<12} {'Risk'}")
    print("-" * 70)

    for model in models_to_test:
        model_results = [r for r in results if r["model"] == model]
        avg_accuracy = sum(r["scores"]["accuracy"] for r in model_results) / len(model_results)
        avg_complete = sum(r["scores"]["completeness"] for r in model_results) / len(model_results)
        avg_risk = sum(r["scores"]["risk_disclosure"] for r in model_results) / len(model_results)
        avg_total = sum(r["avg_score"] for r in model_results) / len(model_results)

        print(f"{model:<20} {avg_total:>8.2f}      {avg_accuracy:>8.2f}      {avg_complete:>8.2f}      {avg_risk:.2f}")

    # Save results
    results_dir = Path(__file__).parent / "results"
    results_dir.mkdir(exist_ok=True)
    with open(results_dir / "judge_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to results/judge_results.json")


if __name__ == "__main__":
    main()
