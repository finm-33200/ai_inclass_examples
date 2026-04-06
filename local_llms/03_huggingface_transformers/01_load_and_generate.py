"""
01_load_and_generate.py

Load a model directly from HuggingFace Hub and generate text.
This is the foundation for working with models not available in Ollama.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def main():
    print("Loading TinyLlama from HuggingFace Hub")
    print("=" * 50)
    print()

    model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

    # Load tokenizer and model
    print(f"Loading model: {model_name}")
    print("(First run downloads ~2GB of weights)\n")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
    )

    print(f"Model loaded on device: {model.device}")
    print(f"Model parameters: {model.num_parameters():,}")
    print()

    # Format prompt using chat template
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is compound interest? Explain briefly."},
    ]

    # Apply chat template
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    print(f"Formatted prompt:\n{prompt}\n")

    # Tokenize
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    # Generate
    print("Generating response...")
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=150,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )

    # Decode and print response
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Extract just the assistant's response
    if "<|assistant|>" in response:
        response = response.split("<|assistant|>")[-1].strip()

    print(f"Response:\n{response}")


if __name__ == "__main__":
    main()
