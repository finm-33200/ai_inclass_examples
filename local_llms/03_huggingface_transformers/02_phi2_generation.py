"""
02_phi2_generation.py

Load Microsoft's Phi-2 model for comparison.
Phi-2 is larger (2.7B parameters) but known for strong reasoning.

Note: Requires ~6GB RAM. Skip this example if memory-constrained.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def main():
    print("Loading Microsoft Phi-2 from HuggingFace Hub")
    print("=" * 50)
    print()

    model_name = "microsoft/phi-2"

    print(f"Loading model: {model_name}")
    print("(First run downloads ~6GB of weights)")
    print("(Requires ~6GB RAM - skip if memory-constrained)\n")

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto",
            trust_remote_code=True,
        )
    except Exception as e:
        print(f"Error loading model: {e}")
        print("\nThis model requires ~6GB RAM. Try:")
        print("1. Close other applications")
        print("2. Use the TinyLlama example instead")
        print("3. Try quantization in example 04")
        return

    print(f"Model loaded on device: {model.device}")
    print(f"Model parameters: {model.num_parameters():,}")
    print()

    # Phi-2 uses a simple instruction format
    prompt = """Instruct: Calculate the future value of $1000 invested at 5% annual interest for 10 years with monthly compounding. Show your reasoning step by step.
Output:"""

    print(f"Prompt:\n{prompt}\n")

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    print("Generating response...")
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=300,
            temperature=0.3,  # Lower temperature for more focused reasoning
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Extract just the output part
    if "Output:" in response:
        response = response.split("Output:")[-1].strip()

    print(f"Response:\n{response}")


if __name__ == "__main__":
    main()
