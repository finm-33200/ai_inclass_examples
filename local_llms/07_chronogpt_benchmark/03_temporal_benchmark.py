"""
03_temporal_benchmark.py

Benchmark ChronoGPT models across different time periods.
This script evaluates multiple ChronoGPT versions on simple tasks.
"""

import argparse
import gc
import json
import re
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download
import tiktoken
import importlib.util


def download_model_code(model_name):
    """Download the custom ChronoGPT model code."""
    model_code_path = hf_hub_download(model_name, "ChronoGPT_instruct.py")
    spec = importlib.util.spec_from_file_location("chronogpt", model_code_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_model(model_name, device):
    """Load a ChronoGPT model."""
    chronogpt = download_model_code(model_name)
    model = chronogpt.ChronoGPT.from_pretrained(model_name)
    model = model.to(device)
    model.eval()
    return model, chronogpt


def generate_response(model, chronogpt, tokenizer, device, prompt, max_tokens=100):
    """Generate a response from the model."""
    return chronogpt.extract_response(
        input_text=prompt,
        model=model,
        tokenizer=tokenizer,
        device=device,
        max_tokens=max_tokens,
        temperature=0.0,  # Deterministic for benchmarking
    )


# Simple math questions for evaluation
MATH_QUESTIONS = [
    {"question": "What is 15 + 27?", "answer": "42"},
    {"question": "What is 8 x 7?", "answer": "56"},
    {"question": "What is 100 - 37?", "answer": "63"},
    {"question": "What is 144 / 12?", "answer": "12"},
    {"question": "What is 25 x 4?", "answer": "100"},
    {"question": "What is 200 - 85?", "answer": "115"},
    {"question": "What is 9 x 9?", "answer": "81"},
    {"question": "What is 1000 / 8?", "answer": "125"},
    {"question": "What is 33 + 67?", "answer": "100"},
    {"question": "What is 7 x 13?", "answer": "91"},
]

# General knowledge questions
KNOWLEDGE_QUESTIONS = [
    {"question": "What is the capital of France?", "answer": "Paris"},
    {"question": "What planet is known as the Red Planet?", "answer": "Mars"},
    {"question": "What is the chemical symbol for gold?", "answer": "Au"},
    {"question": "Who wrote Romeo and Juliet?", "answer": "Shakespeare"},
    {"question": "What is the largest ocean on Earth?", "answer": "Pacific"},
    {"question": "What is the speed of light in vacuum (approximately in km/s)?", "answer": "300000"},
    {"question": "What gas do plants absorb from the atmosphere?", "answer": "carbon dioxide"},
    {"question": "What is the square root of 144?", "answer": "12"},
    {"question": "In what year did World War II end?", "answer": "1945"},
    {"question": "What is the chemical formula for water?", "answer": "H2O"},
]


def extract_answer(response, expected):
    """Try to extract and match the answer from the response."""
    response_lower = response.lower()
    expected_lower = expected.lower()

    # Direct match
    if expected_lower in response_lower:
        return True

    # Try to find numbers for numeric answers
    if expected.isdigit():
        numbers = re.findall(r"\b" + expected + r"\b", response)
        if numbers:
            return True

    return False


def evaluate_task(model, chronogpt, tokenizer, device, questions, task_name, limit=None):
    """Evaluate model on a set of questions."""
    if limit:
        questions = questions[:limit]

    correct = 0
    total = len(questions)

    print(f"  Evaluating {task_name} ({total} questions)...")

    for q in questions:
        prompt = f"{q['question']} Answer with just the answer, nothing else."
        response = generate_response(model, chronogpt, tokenizer, device, prompt)

        if extract_answer(response, q["answer"]):
            correct += 1

    score = correct / total if total > 0 else 0
    print(f"    Score: {correct}/{total} ({score*100:.1f}%)")

    return {"score": score, "correct": correct, "total": total}


def main():
    parser = argparse.ArgumentParser(description="Benchmark ChronoGPT models")
    parser.add_argument("--limit", type=int, default=10, help="Limit questions per task")
    parser.add_argument("--output", type=str, help="Output JSON file")
    args = parser.parse_args()

    print("ChronoGPT Temporal Benchmark")
    print("=" * 70)
    print()

    # Determine device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print(f"Device: {device}")
    print(f"Questions per task: {args.limit}")
    print()

    # Load tokenizer
    tokenizer = tiktoken.get_encoding("gpt2")

    # Models to benchmark
    models_to_test = [
        ("manelalab/chrono-gpt-instruct-v1-20241231", "ChronoGPT-2024"),
        ("manelalab/chrono-gpt-instruct-v1-20001231", "ChronoGPT-2000"),
    ]

    results = {"models": {}, "summary": {}}
    total_evaluations = 0

    for model_name, display_name in models_to_test:
        print("-" * 70)
        print(f"Evaluating: {display_name}")
        print("-" * 70)

        model, chronogpt = load_model(model_name, device)

        model_results = {"tasks": {}}

        # Evaluate on math
        math_results = evaluate_task(
            model, chronogpt, tokenizer, device,
            MATH_QUESTIONS, "math_reasoning", args.limit
        )
        model_results["tasks"]["math_reasoning"] = math_results
        total_evaluations += math_results["total"]

        # Evaluate on knowledge
        knowledge_results = evaluate_task(
            model, chronogpt, tokenizer, device,
            KNOWLEDGE_QUESTIONS, "general_knowledge", args.limit
        )
        model_results["tasks"]["general_knowledge"] = knowledge_results
        total_evaluations += knowledge_results["total"]

        results["models"][display_name] = model_results

        # Clean up
        del model, chronogpt
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        print()

    results["summary"] = {
        "total_evaluations": total_evaluations,
        "models_evaluated": len(models_to_test),
        "questions_per_task": args.limit,
    }

    # Save results
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to: {args.output}")

    # Display summary
    print("=" * 70)
    print("Summary")
    print("=" * 70)
    print()
    print(f"{'Model':<20} {'Math':>15} {'Knowledge':>15}")
    print("-" * 50)

    for model_name, model_results in results["models"].items():
        math_score = model_results["tasks"]["math_reasoning"]["score"] * 100
        knowledge_score = model_results["tasks"]["general_knowledge"]["score"] * 100
        print(f"{model_name:<20} {math_score:>14.1f}% {knowledge_score:>14.1f}%")

    print()
    print("Note: ChronoGPT is optimized for temporal consistency, not benchmark scores.")
    print("The 2000 model should show different knowledge patterns than the 2024 model.")


if __name__ == "__main__":
    main()
