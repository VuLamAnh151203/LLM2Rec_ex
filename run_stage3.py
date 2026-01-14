
import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import argparse
from tqdm import tqdm
import pickle
import random

# --- Model Definitions ---
class CFTeacher(nn.Module):
    """ Simple BPR-MF for scoring, reusing structure from train_cf_teacher.py """
    def __init__(self, num_users, num_items, embedding_dim=2048):
        super(CFTeacher, self).__init__()
        self.user_emb = nn.Embedding(num_users, embedding_dim)
        self.item_emb = nn.Embedding(num_items, embedding_dim)

    def forward(self, u_idxs):
        # Return scores over ALL items for these users
        # u_idxs: [batch_size]
        # output: [batch_size, num_items]
        u_e = self.user_emb(u_idxs)
        scores = torch.matmul(u_e, self.item_emb.weight.t())
        return scores

class StudentModel(nn.Module):
    """ 
    User Encoder: MLP(Mean(History_Embeddings)) 
    Item Encoder: Identity (Static Embeddings)
    """
    def __init__(self, item_embeddings, hidden_dim=512, output_dim=128):
        super(StudentModel, self).__init__()
        # Static Item Embeddings (Frozen)
        # item_embeddings: numpy array [num_items, input_dim]
        # We assume item_embeddings are aligned with our internal item IDs used in mapping
        
        num_items, input_dim = item_embeddings.shape
        self.item_embedding = nn.Embedding.from_pretrained(
            torch.FloatTensor(item_embeddings), 
            freeze=True
        )
        
        # User Encoder
        # input: Mean of History Item Embeddings (dim = input_dim)
        # output: User Vector (dim = input_dim to match item space? Or project both?)
        # Step 5 says: s(u, i) = g(H_u, e_i)
        # Usually we want dot product similarity in same space.
        
        # User Projector
        self.user_mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim) # Project back to embedding space for dot product
        )
        
    def forward(self, history_indices, target_item_indices):
        # history_indices: [batch, max_len] (padded with padding_idx?)
        # We need to handle padding for mean.
        
        # CRITICAL: Validate indices before embedding lookup
        num_embeddings = self.item_embedding.num_embeddings
        
        # DEFENSIVE: Ensure correct dtype and clamp ALL indices
        # Convert to long (int64) first to ensure compatibility
        history_indices = history_indices.long()
        target_item_indices = target_item_indices.long()
        
        # Debug: Check for invalid indices BEFORE clamping
        if history_indices.max() >= num_embeddings or history_indices.min() < -1:
            print(f"WARNING: Clamping history indices from [{history_indices.min().item()}, {history_indices.max().item()}] to [-1, {num_embeddings-1}]")
        if target_item_indices.max() >= num_embeddings or target_item_indices.min() < 0:
            print(f"WARNING: Clamping target indices from [{target_item_indices.min().item()}, {target_item_indices.max().item()}] to [0, {num_embeddings-1}]")
        
        # Clamp to valid range
        history_indices = torch.clamp(history_indices, -1, num_embeddings - 1)
        target_item_indices = torch.clamp(target_item_indices, 0, num_embeddings - 1)
        
        # CRITICAL FIX: Replace -1 padding with 0 before embedding lookup
        # PyTorch embedding doesn't support negative indices
        # We'll use the mask later to ignore these positions anyway
        history_indices_for_lookup = torch.where(history_indices == -1, torch.zeros_like(history_indices), history_indices)
        
        # Get embeddings
        # [batch, max_len, dim]
        hist_embs = self.item_embedding(history_indices_for_lookup) 
        
        # Mean pooling (ignoring padding if 0 is pad? Assume 0 is valid item? We need a pad idx)
        # Simple mean for now - let's assume we handle lengths or padding mask
        mask = (history_indices != -1).unsqueeze(-1).float() # [batch, max_len, 1]
        sum_embs = (hist_embs * mask).sum(dim=1)
        count = mask.sum(dim=1).clamp(min=1)
        mean_embs = sum_embs / count
        
        user_vec = self.user_mlp(mean_embs) # [batch, dim]
        
        target_vec = self.item_embedding(target_item_indices) # [batch, dim]
        
        # Scores
        scores = (user_vec * target_vec).sum(dim=-1)
        return scores, user_vec

# --- Data ---
class AlignmentDataset(Dataset):
    def __init__(self, data_df, title_to_emb_idx, user_map, num_embeddings, max_hist_len=20):
        self.data = data_df
        self.title_to_emb_idx = title_to_emb_idx
        self.user_map = user_map
        self.max_hist_len = max_hist_len
        self.num_embeddings = num_embeddings  # Actual size of embedding table
        
        # Precompute indices
        self.samples = []
        for idx, row in tqdm(data_df.iterrows(), total=len(data_df), desc="Precomputing Dataset"):
            if 'user_id' not in row or row['user_id'] not in user_map:
                continue
                
            u_idx = user_map[row['user_id']]
            
            # Target
            if row['item_title'] not in title_to_emb_idx:
                continue
            target_idx = title_to_emb_idx[row['item_title']]
            
            # History
            # row['history_item_title'] is string rep of list: "['Title A', 'Title B']"
            try:
                hist_titles = eval(row['history_item_title'])
                hist_idxs = [title_to_emb_idx[t] for t in hist_titles if t in title_to_emb_idx]
            except:
                hist_idxs = []
                
            if not hist_idxs:
                continue
            
            # CRITICAL: Validate all indices are within bounds
            # This prevents CUDA assert errors during training
            # Use ACTUAL embedding table size, not dictionary size
            max_valid_idx = self.num_embeddings - 1
            hist_idxs = [idx for idx in hist_idxs if 0 <= idx <= max_valid_idx]
            
            if not hist_idxs:  # All history indices were invalid
                continue
            
            # Also validate target
            if target_idx < 0 or target_idx > max_valid_idx:
                continue
                
            # Pad/Truncate
            hist_idxs = hist_idxs[-max_hist_len:]
            pad_len = max_hist_len - len(hist_idxs)
            hist_idxs = hist_idxs + [-1] * pad_len # -1 as pad
            
            self.samples.append({
                'u_idx': u_idx,
                'target_idx': int(target_idx),
                'hist_idxs': np.array(hist_idxs, dtype=np.int64)
            })
            
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        return (
            torch.tensor(s['u_idx'], dtype=torch.long),
            torch.tensor(s['hist_idxs'], dtype=torch.long),
            torch.tensor(s['target_idx'], dtype=torch.long)
        )

# --- Training ---
def train_alignment(
    train_file, 
    item_emb_file, 
    item_titles_file,
    cf_teacher_dir, 
    output_dir,
    epochs=20,
    batch_size=128,
    lr=1e-3,
    loss_type="BPR"
):
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")
    
    # 1. Load Maps & Embeddings
    print("Loading Embeddings & Maps...")
    item_embeddings = np.load(item_emb_file)
    with open(item_titles_file, 'r', encoding='utf-8') as f:
        titles = [line.strip() for line in f if line.strip()]
    title_to_emb_idx = {t: i for i, t in enumerate(titles)}
    
    
    # Check if map files exist
    user_map_path = os.path.join(cf_teacher_dir, 'user_map.pkl')
    item_map_path = os.path.join(cf_teacher_dir, 'item_map.pkl')
    
    if os.path.exists(user_map_path):
        with open(user_map_path, 'rb') as f:
            teacher_user_map = pickle.load(f)
    else:
        print(f"Warning: {user_map_path} not found. Assuming Identity Mapping (User ID = Embedding Index).")
        df_temp = pd.read_csv(train_file)
        
        # Handle both 'user_id' and 'user' column names
        if 'user_id' in df_temp.columns:
            unique_users = df_temp['user_id'].unique()
        elif 'user' in df_temp.columns:
            unique_users = df_temp['user'].unique()
        else:
            raise KeyError(f"train.csv must have either 'user_id' or 'user' column. Found: {df_temp.columns.tolist()}")
            
        teacher_user_map = {u: int(u) for u in unique_users} # Identity map
        
    if os.path.exists(item_map_path):
        with open(item_map_path, 'rb') as f:
            teacher_item_map = pickle.load(f)
    else:
         print(f"Warning: {item_map_path} not found. Assuming Identity Mapping (Item ID = Embedding Index).")
         df_temp = pd.read_csv(train_file)
         
         # Handle both 'item_id' and 'item' column names
         if 'item_id' in df_temp.columns:
             unique_items = df_temp['item_id'].unique()
         elif 'item' in df_temp.columns:
             unique_items = df_temp['item'].unique()
         else:
             raise KeyError(f"train.csv must have either 'item_id' or 'item' column. Found: {df_temp.columns.tolist()}")
             
         teacher_item_map = {i: int(i) for i in unique_items}

        
    # We need to map Student Item Indices (Embedding Idx) -> Teacher Item Indices (CF Idx)
    # This is tricky without 'item_id' available in title map or vice versa.
    # Assumption: `item.csv` maps item_id <-> item_title 1:1.
    # We should rebuild this map.
    # Or rely on `train.csv` having both.
    
    # Let's verify mapping on the fly during dataset prep? Too slow.
    # Solution: Load `item.csv` (since we know path structure relative to train_file?)
    # Assume `train_file` is in `llm2rec_processed`, neighbor to `item.csv`? No, neighbor to `item_titles.txt`.
    # `item.csv` is in parent dir usually.
    # Let's just pass `item_map_file`... or rely on `train.csv` rows having both ID and Title.
    # Correct: `train.csv` has `item_id` and `item_title`.
    # We can build `title -> item_id` map from `train.csv` (unique).
    
    print("Building Maps from Train Data...")
    df = pd.read_csv(train_file)
    
    # Title -> ItemID -> TeacherIdx
    # Note: `item_map` from CF Teacher maps `item_id` (int) -> `teacher_idx` (0..M).
    title_to_item_id = dict(zip(df['item_title'], df['item_id']))
    
    # Create mapping array: student_idx -> teacher_idx
    # items in `item_embeddings` correspond to `titles` list order.
    # `student_idx` i corresponds to `titles[i]`.
    # `titles[i]` -> `item_id` -> `teacher_idx`.
    
    student_to_teacher = np.zeros(len(titles), dtype=int)
    for i, t in enumerate(titles):
        if t in title_to_item_id and title_to_item_id[t] in teacher_item_map:
            student_to_teacher[i] = teacher_item_map[title_to_item_id[t]]
        else:
            student_to_teacher[i] = -1 # Missing in teacher (shouldn't happen for train items)
            
    # Build Teacher -> Student Map
    # student_to_teacher[s_idx] = t_idx. Inverse needed.
    # teacher_to_student[t_idx] = s_idx.
    
    teacher_to_student = np.zeros(len(teacher_item_map), dtype=int)
    # Be careful: multiple students mapping to same teacher? Unlikely given unique titles/IDs.
    # Multiple students might imply different titles for same ID? No. One-to-one.
    
    found_count = 0
    for s_idx, t_idx in enumerate(student_to_teacher):
        if t_idx != -1 and t_idx < len(teacher_to_student):
            teacher_to_student[t_idx] = s_idx
            found_count += 1
            
    print(f"Mapped {found_count} items between Teacher and Student.")
    teacher_to_student_tensor = torch.tensor(teacher_to_student, device=device)

    # 2. Dataset
    # CRITICAL: Pass actual embedding size, not dictionary size
    num_item_embeddings = item_embeddings.shape[0]
    print(f"Item embeddings shape: {item_embeddings.shape}")
    print(f"Title dictionary size: {len(title_to_emb_idx)}")
    
    if num_item_embeddings != len(title_to_emb_idx):
        print(f"WARNING: Mismatch between embeddings ({num_item_embeddings}) and titles ({len(title_to_emb_idx)})")
        print(f"Will use embedding size ({num_item_embeddings}) for validation")
    
    dataset = AlignmentDataset(df, title_to_emb_idx, teacher_user_map, num_item_embeddings)
    print(f"Dataset created with {len(dataset)} samples")
    if len(dataset) > 0:
        # Test first sample
        u, h, t = dataset[0]
        print(f"First sample - user: {u}, hist shape: {h.shape}, target: {t}")
        print(f"  Hist indices: {h}")
        print(f"  Target index: {t.item()}")
        print(f"  Max hist: {h.max().item()}, Min hist: {h.min().item()}")
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # 3. Models
    student_model = StudentModel(item_embeddings).to(device)
    optimizer = optim.Adam(student_model.parameters(), lr=lr)
    
    # Load Teacher
    # Use dummy counts from map to init
    teacher_model = CFTeacher(len(teacher_user_map), len(teacher_item_map))
    
    teacher_ckpt_path = os.path.join(cf_teacher_dir, 'cf_teacher.pt')
    if os.path.exists(teacher_ckpt_path):
        print(f"Loading Teacher from {teacher_ckpt_path}...")
        teacher_model.load_state_dict(torch.load(teacher_ckpt_path))
    else:
        # Fallback: Load separate PT files (user_embedding.pt, item_embedding.pt)
        # Assuming they are simple tensors [num_users, dim]
        u_emb_path = os.path.join(cf_teacher_dir, 'CiteULike_cold_item_MF_user_emb.pt')
        i_emb_path = os.path.join(cf_teacher_dir, 'CiteULike_cold_item_MF_item_emb.pt')
        
        print(f"Loading Teacher from separate files: {u_emb_path}, {i_emb_path}...")
        
        if not os.path.exists(u_emb_path) or not os.path.exists(i_emb_path):
             raise FileNotFoundError(f"Could not find 'cf_teacher.pt' OR pair ('user_embedding.pt', 'item_embedding.pt') in {cf_teacher_dir}")
             
        u_emb_weights = torch.load(u_emb_path, map_location='cpu')
        i_emb_weights = torch.load(i_emb_path, map_location='cpu')
        
        # Handle size mismatches
        # The pre-trained embeddings might have different counts than current train.csv
        expected_users = len(teacher_user_map)
        expected_items = len(teacher_item_map)
        
        loaded_users = u_emb_weights.shape[0]
        loaded_items = i_emb_weights.shape[0]
        
        print(f"Expected: {expected_users} users, {expected_items} items")
        print(f"Loaded: {loaded_users} users, {loaded_items} items")
        
        if loaded_users != expected_users or loaded_items != expected_items:
            print("WARNING: Size mismatch detected. Adjusting...")
            
            # Option 1: Resize the model to match loaded embeddings (safer)
            # This assumes the loaded embeddings are the "ground truth"
            teacher_model = CFTeacher(loaded_users, loaded_items, embedding_dim=u_emb_weights.shape[1])
            
            # Update maps to match (identity mapping up to loaded size)
            teacher_user_map = {i: i for i in range(loaded_users)}
            teacher_item_map = {i: i for i in range(loaded_items)}
            
            print(f"Resized teacher model to match loaded embeddings: {loaded_users} users, {loaded_items} items")
        
        # Now shapes should match
        teacher_model.user_emb.weight.data.copy_(u_emb_weights)
        teacher_model.item_emb.weight.data.copy_(i_emb_weights)
        
    teacher_model.to(device)
    teacher_model.eval()
    
    print("Starting Alignment Training...")
    
    # Curriculum Schedule
    # 0-25% epochs: Random
    # 25-50%: Mixed
    # 50%+: Hard
    
    criterion_infonce = nn.CrossEntropyLoss()
    
    for epoch in range(epochs):
        student_model.train()
        total_loss = 0
        
        # Determine strategy
        if epoch < epochs * 0.25:
            strategy = "random"
            ratio = 0.0
        elif epoch < epochs * 0.5:
            strategy = "mixed"
            ratio = 0.5
        else:
            strategy = "hard"
            ratio = 1.0
            
        pbar = tqdm(dataloader, desc=f"Ep {epoch+1} [{strategy}] Loss: {loss_type}")
        
        for u_idxs, hist_idxs, target_idxs in pbar:
            u_idxs, hist_idxs, target_idxs = u_idxs.to(device), hist_idxs.to(device), target_idxs.to(device)
            batch_size_curr = u_idxs.size(0)
            
            # --- Negative Sampling ---
            if loss_type == 'BPR':
                # BPR: 1 Neg per Pos
                neg_idxs = torch.randint(0, len(titles), (batch_size_curr,), device=device)
                
                if strategy != "random":
                    num_hard = int(batch_size_curr * ratio)
                    if num_hard > 0:
                        with torch.no_grad():
                            cf_scores = teacher_model(u_idxs[:num_hard])
                            # Restrict to warm items (only those present in mapping)
                            cf_scores = cf_scores[:, :teacher_to_student_tensor.size(0)]
                            
                            _, topk_t_idxs = torch.topk(cf_scores, k=min(50, cf_scores.size(1)), dim=1)
                            rand_select = torch.randint(0, topk_t_idxs.size(1), (num_hard, 1), device=device)
                            selected_t_idxs = topk_t_idxs.gather(1, rand_select).squeeze(1)
                            
                            # Convert to Student Indices
                            hard_neg_idxs = teacher_to_student_tensor[selected_t_idxs]
                            
                            # CRITICAL: Validate indices are within bounds
                            # teacher_to_student might map to indices >= len(titles) if there's mismatch
                            max_valid_idx = len(titles) - 1
                            invalid_mask = hard_neg_idxs > max_valid_idx
                            
                            if invalid_mask.any():
                                # Replace invalid indices with random valid ones
                                num_invalid = invalid_mask.sum().item()
                                hard_neg_idxs[invalid_mask] = torch.randint(0, len(titles), (num_invalid,), device=device)
                                print(f"Warning: {num_invalid} hard negatives out of bounds, replaced with random")
                            
                            
                            neg_idxs[:num_hard] = hard_neg_idxs
                
                optimizer.zero_grad()
                pos_scores, _ = student_model(hist_idxs, target_idxs) # [B]
                neg_scores, _ = student_model(hist_idxs, neg_idxs)    # [B]
                loss = -torch.mean(torch.nn.functional.logsigmoid(pos_scores - neg_scores))

            else: # InfoNCE
                # InfoNCE: 1 Pos vs (Batch Negatives + Hard Negatives)
                # To keep it simple and efficient: we use In-Batch Negatives as "Random"
                # And we explicitly sample "Hard Negatives" and append to the batch
                
                # 1. Compute User & Positive Item Vectors
                optimizer.zero_grad()
                _, user_vec = student_model(hist_idxs, target_idxs) # [B, Dim]
                target_vec = student_model.item_embedding(target_idxs) # [B, Dim]
                
                # Normalization typically good for InfoNCE (cosine sim)
                user_vec = torch.nn.functional.normalize(user_vec, dim=-1)
                target_vec = torch.nn.functional.normalize(target_vec, dim=-1)
                
                # 2. Hard Negatives (if strategy demands)
                hard_negs_vec = None
                if strategy != "random":
                     num_hard_users = int(batch_size_curr * ratio)
                     if num_hard_users > 0:
                        with torch.no_grad():
                            # Teacher Logic (Same as BPR)
                            cf_scores = teacher_model(u_idxs[:num_hard_users])
                            # Restrict to warm items (only those present in mapping)
                            cf_scores = cf_scores[:, :teacher_to_student_tensor.size(0)]
                            
                            _, topk_t_idxs = torch.topk(cf_scores, k=min(50, cf_scores.size(1)), dim=1)
                            rand_select = torch.randint(0, topk_t_idxs.size(1), (num_hard_users, 1), device=device)
                            selected_t_idxs = topk_t_idxs.gather(1, rand_select).squeeze(1)
                            hard_neg_idxs = teacher_to_student_tensor[selected_t_idxs] # [num_hard]
                            
                            # Validate bounds
                            max_valid_idx = len(titles) - 1
                            invalid_mask = hard_neg_idxs > max_valid_idx
                            if invalid_mask.any():
                                num_invalid = invalid_mask.sum().item()
                                hard_neg_idxs[invalid_mask] = torch.randint(0, len(titles), (num_invalid,), device=device)
                                print(f"Warning (InfoNCE): {num_invalid} hard negatives out of bounds, replaced with random")
                        
                        hard_negs_vec = student_model.item_embedding(hard_neg_idxs) # [num_hard, Dim]
                        hard_negs_vec = torch.nn.functional.normalize(hard_negs_vec, dim=-1)

                # 3. Logits Calculation
                # Positive Logits: (B, 1) dot product
                pos_logits = (user_vec * target_vec).sum(dim=1, keepdim=True) 
                
                # In-Batch Negative Logits: (B, B)
                # user[i] vs target[j] where i != j
                # Matmul: [B, Dim] @ [Dim, B] -> [B, B]
                all_neg_logits = torch.matmul(user_vec, target_vec.t())
                
                # Mask out diagonal (positives)
                mask = torch.eye(batch_size_curr, device=device).bool()
                all_neg_logits.masked_fill_(mask, -1e9) # Effectively zero probability
                
                # Append Hard Negatives if any
                if hard_negs_vec is not None:
                    # User[i] vs HardNeg[i] -> Diagonal ONLY?
                    # Hard negatives are specific to user[i].
                    # We compute dot product: (user[i] * hard[i]).sum()
                    # But shape match: user_vec[:num_hard] and hard_negs_vec
                    
                    hard_term_logits = (user_vec[:hard_negs_vec.size(0)] * hard_negs_vec).sum(dim=1, keepdim=True) # [num_hard, 1]
                    
                    # We need to reshape/pad to join with all_neg_logits?
                    # InfoNCE loss expects [B, 1+Negs].
                    # But here Neg count varies per row if we just append hard to specific rows.
                    # Simpler: Just Compute Loss manually or use padded Cat.
                    
                    # Standard Strategy: Positive is Col 0. Negatives are Cols 1..N.
                    
                    # Let's concat efficiently
                    # Logits: [Pos (B,1), Negs (B, B-1), Hard (B, 1 or 0)]
                    pass

                # Let's simplify InfoNCE implementation:
                # Temperature
                temp = 0.07
                
                # Logits: [B, B] (diagonal is pos, off-diagonal is neg)
                logits = torch.matmul(user_vec, target_vec.t()) / temp
                labels = torch.arange(batch_size_curr, device=device)
                
                # If we have hard negatives, we must modify logits?
                # Hard negatives are extra columns.
                # If we add hard negatives, dimension becomes [B, B + K].
                # But hard negatives are 1-per-user (at most).
                # So we can add 1 column of "Hard Negatives".
                
                if hard_negs_vec is not None:
                     # Compute scores for hard negatives [num_hard]
                     # user_vec[:num_hard] dot hard_negs_vec
                     hard_scores = (user_vec[:hard_negs_vec.size(0)] * hard_negs_vec).sum(dim=1) / temp
                     
                     # We need a tensor of shape [B, 1] for hard negatives (pad with -inf)
                     hard_col = torch.full((batch_size_curr, 1), -1e9, device=device)
                     hard_col[:hard_negs_vec.size(0), 0] = hard_scores
                     
                     # Concat to logits [B, B+1]
                     logits = torch.cat([logits, hard_col], dim=1)
                
                loss = criterion_infonce(logits, labels)

            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            pbar.set_postfix({'loss': loss.item()})
            
        print(f"Epoch {epoch+1} Mean Loss: {total_loss / len(dataloader):.4f}")

    # Save
    os.makedirs(output_dir, exist_ok=True)
    save_name = f"student_aligned_{loss_type}.pt"
    torch.save(student_model.state_dict(), os.path.join(output_dir, save_name))
    print(f"Saved Stage 3 Model ({loss_type}).")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_file", required=True)
    parser.add_argument("--item_emb_file", required=True)
    parser.add_argument("--item_titles_file", required=True)
    parser.add_argument("--cf_teacher_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--epochs", default=20, type=int)
    parser.add_argument("--batch_size", default=128, type=int)
    parser.add_argument("--lr", default=1e-3, type=float)
    parser.add_argument("--loss_type", default="BPR", choices=["BPR", "InfoNCE"])
    args = parser.parse_args()
    
    train_alignment(
        args.train_file, 
        args.item_emb_file, 
        args.item_titles_file, 
        args.cf_teacher_dir, 
        args.output_dir,
        loss_type=args.loss_type
    )
