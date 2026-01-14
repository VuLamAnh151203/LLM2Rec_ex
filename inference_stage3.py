import os
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import argparse
from tqdm import tqdm
import pickle

# --- Model Definitions (Must match run_stage3.py) ---
class StudentModel(nn.Module):
    def __init__(self, item_embeddings, hidden_dim=512, output_dim=128):
        super(StudentModel, self).__init__()
        num_items, input_dim = item_embeddings.shape
        self.item_embedding = nn.Embedding.from_pretrained(torch.FloatTensor(item_embeddings), freeze=True)
        
        self.user_mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )
        
    def forward(self, history_indices):
        # Optimized for inference: only compute user vectors
        num_embeddings = self.item_embedding.num_embeddings
        history_indices = history_indices.long()
        history_indices_for_lookup = torch.where(history_indices == -1, torch.zeros_like(history_indices), history_indices)
        
        hist_embs = self.item_embedding(history_indices_for_lookup) 
        mask = (history_indices != -1).unsqueeze(-1).float()
        sum_embs = (hist_embs * mask).sum(dim=1)
        count = mask.sum(dim=1).clamp(min=1)
        mean_embs = sum_embs / count
        
        user_vec = self.user_mlp(mean_embs)
        return user_vec

def run_inference(
    train_file,
    item_emb_file,
    item_titles_file,
    checkpoint_path,
    test_file,
    output_file,
    batch_size=256,
    top_k=20,
    max_hist_len=20
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Load Embeddings & Titles
    print("Loading Item Embeddings & Titles...")
    item_embeddings = np.load(item_emb_file)
    with open(item_titles_file, 'r', encoding='utf-8') as f:
        titles = [line.strip() for line in f if line.strip()]
    title_to_emb_idx = {t: i for i, t in enumerate(titles)}
    num_item_embeddings = item_embeddings.shape[0]

    # 2. Load Student Model
    print(f"Loading Student Model from {checkpoint_path}...")
    student_model = StudentModel(item_embeddings).to(device)
    student_model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    student_model.eval()

    # 3. Build User History Map from train_file
    print(f"Building User History from {train_file}...")
    df_train = pd.read_csv(train_file)
    
    # We need a unique list of users and their latest history
    user_histories = {}
    
    # In CiteULike, 'history_item_title' is a string representation of a list
    # We'll take the latest interaction for each user to represent their current profile
    for _, row in tqdm(df_train.iterrows(), total=len(df_train), desc="Processing Users"):
        u_id = row['user_id']
        if u_id not in user_histories:
            try:
                hist_titles = eval(row['history_item_title'])
                # Pre-process history to indices
                hist_idxs = [title_to_emb_idx[t] for t in hist_titles if t in title_to_emb_idx]
                if hist_idxs:
                    # Validate and pad
                    hist_idxs = [idx for idx in hist_idxs if 0 <= idx < num_item_embeddings]
                    hist_idxs = hist_idxs[-max_hist_len:]
                    pad_len = max_hist_len - len(hist_idxs)
                    hist_idxs = hist_idxs + [-1] * pad_len
                    user_histories[u_id] = hist_idxs
            except:
                continue

    user_ids = sorted(list(user_histories.keys()))
    user_hist_matrix = np.array([user_histories[uid] for uid in user_ids], dtype=np.int64)
    
    # 4. Compute All User Embeddings
    print("Computing User Embeddings...")
    all_user_vecs = []
    with torch.no_grad():
        for i in tqdm(range(0, len(user_hist_matrix), batch_size), desc="User Embedding Batches"):
            batch_hist = torch.tensor(user_hist_matrix[i:i+batch_size], device=device)
            user_vecs = student_model(batch_hist) # [B, Dim]
            # Normalize for cosine similarity (optional, depends on Stage 3 loss, but good for ranking)
            user_vecs = torch.nn.functional.normalize(user_vecs, dim=-1)
            all_user_vecs.append(user_vecs.cpu())
    
    all_user_vecs = torch.cat(all_user_vecs, dim=0) # [NumUsers, Dim]
    print(f"Computed embeddings for {len(user_ids)} users.")

    # 5. Process Cold Items from test_file
    print(f"Processing Cold Items from {test_file}...")
    df_test = pd.read_csv(test_file)
    
    # Need item_id -> title mapping if not in test_file
    # Assuming test_file has 'item_id' and 'item_title' (standard for these scripts)
    # If not, we might need item.csv. Let's assume it has them or we can lookup by ID.
    
    results = []
    
    # Pre-calculate item embeddings for target cold items
    target_items = df_test[['item_id', 'item_title']].drop_duplicates()
    
    print(f"Ranking users for {len(target_items)} cold items...")
    
    all_user_vecs = all_user_vecs.to(device)
    
    with torch.no_grad():
        for _, row in tqdm(target_items.iterrows(), total=len(target_items), desc="Item-to-User Ranking"):
            item_id = row['item_id']
            item_title = row['item_title']
            
            if item_title not in title_to_emb_idx:
                print(f"Warning: Title '{item_title}' not found in item_titles.txt! Skipping.")
                continue
            
            idx = title_to_emb_idx[item_title]
            if idx >= num_item_embeddings:
                continue
                
            item_vec = student_model.item_embedding(torch.tensor([idx], device=device)) # [1, Dim]
            item_vec = torch.nn.functional.normalize(item_vec, dim=-1)
            
            # Scores: [1, Dim] @ [Dim, NumUsers] -> [1, NumUsers]
            scores = torch.matmul(item_vec, all_user_vecs.t()).squeeze(0)
            
            # Top-K
            top_scores, top_indices = torch.topk(scores, k=min(top_k, len(user_ids)))
            
            # Map indices back to user_ids
            rec_user_ids = [user_ids[idx.item()] for idx in top_indices]
            
            results.append({
                'item_id': item_id,
                'top_user_ids': rec_user_ids
            })

    # 6. Save Results
    df_results = pd.DataFrame(results)
    df_results.to_csv(output_file, index=False)
    print(f"Inference complete! Results saved to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_file", required=True)
    parser.add_argument("--item_emb_file", required=True)
    parser.add_argument("--item_titles_file", required=True)
    parser.add_argument("--checkpoint_path", required=True)
    parser.add_argument("--test_file", required=True)
    parser.add_argument("--output_file", default="cold_item_recommendations.csv")
    parser.add_argument("--batch_size", default=256, type=int)
    parser.add_argument("--top_k", default=20, type=int)
    args = parser.parse_args()
    
    run_inference(
        args.train_file,
        args.item_emb_file,
        args.item_titles_file,
        args.checkpoint_path,
        args.test_file,
        args.output_file,
        args.batch_size,
        args.top_k
    )
