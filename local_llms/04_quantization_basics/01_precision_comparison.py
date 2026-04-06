"""
01_precision_comparison.py

Compare loading models at different precisions (float32 vs float16).
This demonstrates the memory-quality trade-off without requiring special libraries.
Works on all platforms (macOS, Linux, Windows).
"""

import gc
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def get_model_memory_mb(model):
    """Estimate model memory usage in MB."""
    param_size = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
    return (param_size + buffer_size) / (1024 * 1024)


def generate_response(model, tokenizer, prompt):
    """Generate a response from the model."""
    # Format with chat template for TinyLlama
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt},
    ]
    formatted_prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    inputs = tokenizer(formatted_prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=100,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # Extract just the assistant's response
    if "<|assistant|>" in response:
        response = response.split("<|assistant|>")[-1].strip()
    return response


def main():
    print("Precision Comparison: float32 vs float16")
    print("=" * 60)
    print()

    model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    test_prompt = "What is inflation? Answer in one sentence."

    # Determine available device
    if torch.cuda.is_available():
        device = "cuda"
        supports_fp16 = True
    elif torch.backends.mps.is_available():
        device = "mps"
        supports_fp16 = True
    else:
        device = "cpu"
        supports_fp16 = True  # CPU supports float16 but may be slower

    print(f"Device: {device}")
    print(f"Model: {model_name}")
    print(f"Test prompt: {test_prompt}")
    print()

    # Load tokenizer (shared across both)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    results = []

    # Test float32 (only on CPU, as GPU prefers float16)
    if device == "cpu":
        print("-" * 60)
        print("Loading with float32 precision...")
        model_fp32 = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float32,
        )
        memory_fp32 = get_model_memory_mb(model_fp32)
        response_fp32 = generate_response(model_fp32, tokenizer, test_prompt)

        results.append(("float32", memory_fp32, response_fp32))
        print(f"Memory usage: {memory_fp32:.1f} MB")
        print(f"Response: {response_fp32}")

        # Clean up
        del model_fp32
        gc.collect()
        print()

    # Test float16
    print("-" * 60)
    print("Loading with float16 precision...")
    model_fp16 = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    memory_fp16 = get_model_memory_mb(model_fp16)
    response_fp16 = generate_response(model_fp16, tokenizer, test_prompt)

    results.append(("float16", memory_fp16, response_fp16))
    print(f"Memory usage: {memory_fp16:.1f} MB")
    print(f"Response: {response_fp16}")

    # Clean up
    del model_fp16
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Summary
    print()
    print("=" * 60)
    print("Summary")
    print("=" * 60)
    print()
    print(f"{'Precision':<12} {'Memory (MB)':<15} {'Relative':<10}")
    print("-" * 40)

    baseline = results[0][1]
    for precision, memory, _ in results:
        relative = memory / baseline * 100
        print(f"{precision:<12} {memory:<15.1f} {relative:.1f}%")

    print()
    print("Note: float16 uses ~50% of float32 memory with minimal quality loss.")
    print("For 4-bit quantization (87.5% savings), see 02_bitsandbytes_quant.py")
    print("(requires Linux + CUDA)")


if __name__ == "__main__":
    main()
