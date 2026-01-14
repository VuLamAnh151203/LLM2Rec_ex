
import argparse
import torch
import os
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

def merge_lora(base_model_path, adapter_path, output_path):
    print(f"Loading base model from: {base_model_path}")
    # Load base model
    # We load in float16 for efficiency, change to float32 if needed for precision
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )

    print(f"Loading LoRA adapter from: {adapter_path}")
    # Load adapter
    model = PeftModel.from_pretrained(base_model, adapter_path)

    print("Merging adapter into base model...")
    # Merge weights
    model = model.merge_and_unload()

    print(f"Saving merged model to: {output_path}")
    # Save model
    model.save_pretrained(output_path)

    # Save tokenizer
    print("Saving tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
    tokenizer.save_pretrained(output_path)
    
    print("Merge complete!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge LoRA adapter into base model.")
    parser.add_argument("--base_model", type=str, required=True, help="Path or HF ID of the base model (e.g., Qwen/Qwen2-0.5B)")
    parser.add_argument("--adapter", type=str, required=True, help="Path to the trained LoRA adapter directory (Stage 1 output)")
    parser.add_argument("--output", type=str, required=True, help="Directory to save the merged full model")

    args = parser.parse_args()

    merge_lora(args.base_model, args.adapter, args.output)
