
import os
import sys
import torch
import numpy as np
import argparse
from tqdm import tqdm
from llm2vec import LLM2Vec
from transformers import AutoTokenizer, AutoConfig, AutoModel
from peft import PeftModel

def extract_embeddings(model_path, data_path, output_path, batch_size=32, max_length=128):
    print(f"Loading model from {model_path}...")
    
    # Load LLM2Vec model
    # Note: LLM2Vec.from_pretrained handles loading base model + adapters if needed
    # But usually we need to be careful with how it was saved.
    # In Stage 2 SimCSE, we saved the model.
    
    # Simple strategy: Load base model, then apply adapters if present, or just load as LLM2Vec
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Use LLM2Vec wrapper as in run_unsupervised_SimCSE.py
    # We assume the model at model_path is the full SimCSE trained model
    
    model = LLM2Vec.from_pretrained(
        base_model_name_or_path=model_path,
        enable_bidirectional=True,
        merge_peft=True,
        pooling_mode="mean",
        max_length=max_length,
        torch_dtype=torch.bfloat16,
    )
    
    model.to(device)
    model.eval()
    
    # Load Item Titles
    print(f"Loading items from {data_path}...")
    with open(data_path, 'r', encoding='utf-8') as f:
        titles = [line.strip() for line in f if line.strip()]
        
    print(f"Found {len(titles)} items.")
    
    all_embeddings = []
    
    # Batch extraction
    for i in tqdm(range(0, len(titles), batch_size)):
        batch_texts = titles[i : i + batch_size]
        
        # Tokenize (handled by model.encode usually, but let's check LLM2Vec API)
        # LLM2Vec has an 'encode' method
        
        with torch.no_grad():
            # encode returns tensor on device
            embeddings = model.encode(batch_texts, batch_size=len(batch_texts), show_progress_bar=False)
            
        all_embeddings.append(embeddings.cpu().numpy())
        
    all_embeddings = np.concatenate(all_embeddings, axis=0)
    print(f"Extracted shape: {all_embeddings.shape}")
    
    # Save
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    np.save(output_path, all_embeddings)
    print(f"Saved embeddings to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True, help="Path to Stage 2 checkpoint")
    parser.add_argument("--data_path", type=str, required=True, help="Path to item_titles.txt")
    parser.add_argument("--output_path", type=str, required=True, help="Path to save .npy file")
    parser.add_argument("--batch_size", type=int, default=128)
    args = parser.parse_args()
    
    extract_embeddings(args.model_path, args.data_path, args.output_path, args.batch_size)
