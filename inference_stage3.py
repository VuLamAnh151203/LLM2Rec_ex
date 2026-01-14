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
        
    def forward(self, history_indices, use_mlp=True):
        # Optimized for inference: only compute user vectors
        num_embeddings = self.item_embedding.num_embeddings
        history_indices = history_indices.long()
        history_indices_for_lookup = torch.where(history_indices == -1, torch.zeros_like(history_indices), history_indices)
        
        hist_embs = self.item_embedding(history_indices_for_lookup) 
        mask = (history_indices != -1).unsqueeze(-1).float()
        sum_embs = (hist_embs * mask).sum(dim=1)
        count = mask.sum(dim=1).clamp(min=1)
        mean_embs = sum_embs / count
        
        if use_mlp:
            user_vec = self.user_mlp(mean_embs) + mean_embs # [batch, dim]

            user_vec = user_vec / user_vec.norm(dim=-1, keepdim=True)
            return user_vec
        else:
            # "Weighted-only" mode: skip MLP alignment
            return mean_embs

class AttentiveStudentModel(nn.Module):
    """
    Attentive Collaborative Filtering User Encoder

    - Learns CF signal by reweighting interacted item embeddings
    - Preserves embedding geometry (no MLP, no nonlinearity)
    - Uses global trainable queries with temperature scaling
    """

    def __init__(
        self,
        item_embeddings,        # numpy array or tensor [num_items, dim]
        num_heads=2,
        temperature=0.1
    ):
        super().__init__()

        num_items, dim = item_embeddings.shape

        # Frozen CF item embeddings
        self.item_embedding = nn.Embedding.from_pretrained(
            torch.tensor(item_embeddings, dtype=torch.float32),
            freeze=True
        )

        self.num_heads = num_heads
        self.temperature = temperature
        self.dim = dim

        # Global attention queries (CF-safe)
        self.queries = nn.Parameter(torch.empty(num_heads, dim))
        nn.init.normal_(self.queries, mean=0.0, std=0.1)

    def forward(self, history_indices):
        """
        history_indices: LongTensor [batch, max_len]
        padding index = -1
        """

        batch_size, max_len = history_indices.shape
        num_items = self.item_embedding.num_embeddings

        # Safety clamp
        history_indices = history_indices.clamp(min=-1, max=num_items - 1)

        # Replace padding index for lookup
        lookup_indices = torch.where(
            history_indices == -1,
            torch.zeros_like(history_indices),
            history_indices
        )

        # Item embeddings: [B, L, D]
        hist_embs = self.item_embedding(lookup_indices)

        # Attention scores: [B, K, L]
        scores = torch.einsum(
            "bld,kd->bkl",
            hist_embs,
            self.queries
        )

        # Padding mask
        mask = (history_indices != -1).unsqueeze(1)  # [B, 1, L]
        scores = scores.masked_fill(~mask, -1e9)

        # Temperature-scaled softmax
        attn = torch.softmax(scores / self.temperature, dim=-1)

        # Head-wise aggregation: [B, K, D]
        head_vecs = torch.einsum(
            "bkl,bld->bkd",
            attn,
            hist_embs
        )

        # Combine heads (scale to preserve gradient magnitude)
        user_vec = head_vecs.mean(dim=1) * self.num_heads  # [B, D]

        return user_vec

def run_inference(
    train_file,
    item_emb_file,
    checkpoint_path,
    test_file,
    output_file,
    batch_size=256,
    top_k=20,
    max_hist_len=20,
    use_mlp=True,
    model_type="MLP"
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Load Embeddings
    print("Loading Item Embeddings...")
    item_embeddings = np.load(item_emb_file)
    num_item_embeddings = item_embeddings.shape[0]

    # 2. Load Student Model
    print(f"Loading Student Model ({model_type}) from {checkpoint_path}...")
    if model_type == "Attentive":
        student_model = AttentiveStudentModel(item_embeddings).to(device)
    else:
        student_model = StudentModel(item_embeddings).to(device)
        
    student_model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    student_model.eval()

    # 3. Build User History Map from train_file
    print(f"Building User History from {train_file}...")
    df_train = pd.read_csv(train_file)
    
    # Robust column detection for train file
    user_col = 'user_id' if 'user_id' in df_train.columns else 'user'
    hist_col = 'history_item_id' if 'history_item_id' in df_train.columns else ('history' if 'history' in df_train.columns else None)
    
    if user_col not in df_train.columns:
        raise KeyError(f"Could not find user column in {train_file}. Columns: {df_train.columns.tolist()}")
    if hist_col is None:
        raise KeyError(f"Could not find history ID column in {train_file}. Columns: {df_train.columns.tolist()}")

    # We need a unique list of users and their latest history
    user_histories = {}
    
    for _, row in tqdm(df_train.iterrows(), total=len(df_train), desc="Processing Users"):
        u_id = row[user_col]
        if u_id not in user_histories:
            try:
                hist_val = row[hist_col]
                # Usually history_item_id is "[123, 456, ...]"
                if isinstance(hist_val, str) and hist_val.startswith('['):
                    hist_idxs = eval(hist_val)
                else:
                    # Maybe it's a single ID or already a list
                    hist_idxs = hist_val if isinstance(hist_val, list) else [hist_val]
                
                if hist_idxs:
                    # Filter and pad
                    hist_idxs = [int(idx) for idx in hist_idxs if 0 <= int(idx) < num_item_embeddings]
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
            user_vecs = student_model(batch_hist, use_mlp=use_mlp) # [B, Dim]
            # Normalize for cosine similarity (optional, depends on Stage 3 loss, but good for ranking)
            user_vecs = torch.nn.functional.normalize(user_vecs, dim=-1)
            all_user_vecs.append(user_vecs.cpu())
    
    all_user_vecs = torch.cat(all_user_vecs, dim=0) # [NumUsers, Dim]
    print(f"Computed embeddings for {len(user_ids)} users.")

    # 5. Process Cold Items from test_file
    print(f"Processing Cold Items from {test_file}...")
    df_test = pd.read_csv(test_file)
    
    # Robust column detection for test file
    item_col = 'item_id' if 'item_id' in df_test.columns else 'item'
    
    if item_col not in df_test.columns:
        raise KeyError(f"Could not find item column in {test_file}. Columns: {df_test.columns.tolist()}")

    # For each cold item_id, we just get its embedding from the matrix
    results = []
    
    # Get unique cold item IDs
    cold_item_ids = sorted(df_test[item_col].unique())
    
    print(f"Ranking users for {len(cold_item_ids)} cold items...")
    
    all_user_vecs = all_user_vecs.to(device)
    
    with torch.no_grad():
        for item_id in tqdm(cold_item_ids, desc="Item-to-User Ranking"):
            idx = int(item_id)
            if not (0 <= idx < num_item_embeddings):
                print(f"Warning: Item ID {idx} is out of embedding bounds (0-{num_item_embeddings-1})! Skipping.")
                continue
                
            # Take item embedding directly from StudentModel's static table
            item_vec = student_model.item_embedding(torch.tensor([idx], device=device)) # [1, Dim]
            item_vec = torch.nn.functional.normalize(item_vec, dim=-1)
            
            # Scores: [1, Dim] @ [Dim, NumUsers] -> [1, NumUsers]
            scores = torch.matmul(item_vec, all_user_vecs.t()).squeeze(0)
            
            # Top-K
            top_scores, top_indices = torch.topk(scores, k=min(top_k, len(user_ids)))
            
            # Map indices back to user_ids
            rec_user_ids = [user_ids[idx_val.item()] for idx_val in top_indices]
            
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
    parser.add_argument("--checkpoint_path", required=True)
    parser.add_argument("--test_file", required=True)
    parser.add_argument("--output_file", default="cold_item_recommendations.csv")
    parser.add_argument("--batch_size", default=256, type=int)
    parser.add_argument("--top_k", default=20, type=int)
    parser.add_argument("--no_mlp", action="store_true", help="Use weighted-only mode (skip MLP alignment)")
    parser.add_argument("--model_type", default="MLP", choices=["MLP", "Attentive"])
    args = parser.parse_args()
    
    run_inference(
        args.train_file,
        args.item_emb_file,
        args.checkpoint_path,
        args.test_file,
        args.output_file,
        args.batch_size,
        args.top_k,
        use_mlp=not args.no_mlp,
        model_type=args.model_type
    )
