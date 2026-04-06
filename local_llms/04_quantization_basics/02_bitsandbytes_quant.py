"""
02_bitsandbytes_quant.py

Load a model with 4-bit quantization using bitsandbytes.
This dramatically reduces memory usage, enabling larger models on consumer GPUs.

NOTE: Requires Linux with CUDA GPU. Not supported on macOS or Windows.
"""

import sys
import torch


def check_requirements():
    """Check if the system supports bitsandbytes."""
    if not torch.cuda.is_available():
        print("=" * 60)
        print("CUDA GPU Required")
        print("=" * 60)
        print()
        print("bitsandbytes requires a CUDA GPU (Linux only).")
        print()
        print("Your system:")
        print(f"  - CUDA available: {torch.cuda.is_available()}")
        if torch.backends.mps.is_available():
            print("  - MPS available: True (Apple Silicon)")
            print()
            print("For macOS, use float16 precision instead (see 01_precision_comparison.py)")
        print()
        print("Alternatives:")
        print("  1. Use Google Colab (free GPU): colab.research.google.com")
        print("  2. Use Ollama with pre-quantized models: ollama pull llama3.2:3b")
        print("  3. Use GGUF models with llama-cpp-python")
        return False

    return True


def main():
    if not check_requirements():
        sys.exit(0)

    # Import here to avoid errors on unsupported platforms
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    print("4-bit Quantization with bitsandbytes")
    print("=" * 60)
    print()

    model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

    # Configure 4-bit quantization
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",  # Normalized float 4-bit
        bnb_4bit_use_double_quant=True,  # Nested quantization for more savings
    )

    print(f"Loading {model_name} with 4-bit quantization...")
    print()

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quantization_config,
        device_map="auto",
    )

    # Calculate memory usage
    param_bytes = sum(
        p.numel() * p.element_size() for p in model.parameters()
    )
    memory_mb = param_bytes / (1024 * 1024)

    print(f"Model loaded!")
    print(f"Memory usage: ~{memory_mb:.1f} MB (vs ~2200 MB for float16)")
    print(f"Device: {model.device}")
    print()

    # Test generation
    messages = [
        {"role": "user", "content": "What is the P/E ratio? Explain briefly."},
    ]

    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    print("Generating response...")
    print()

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=150,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    if "<|assistant|>" in response:
        response = response.split("<|assistant|>")[-1].strip()

    print(f"Response:\n{response}")
    print()
    print("=" * 60)
    print("4-bit quantization enables running much larger models!")
    print("Try: meta-llama/Llama-3.2-3B or mistralai/Mistral-7B-v0.1")


if __name__ == "__main__":
    main()
