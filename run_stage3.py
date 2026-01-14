
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
    def __init__(self, num_users, num_items, embedding_dim=64):
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
    def __init__(self, item_embeddings, hidden_dim=256, output_dim=128):
        super(StudentModel, self).__init__()
        # Static Item Embeddings (Frozen)
        # item_embeddings: numpy array [num_items, input_dim]
        # We assume item_embeddings are aligned with our internal item IDs used in mapping
        
        num_items, input_dim = item_embeddings.shape
        self.item_embedding = nn.Embedding.from_pretrained(torch.FloatTensor(item_embeddings), freeze=True)
        
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
        
        # Get embeddings
        # [batch, max_len, dim]
        hist_embs = self.item_embedding(history_indices) 
        
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
    def __init__(self, data_df, title_to_emb_idx, user_map, max_hist_len=20):
        self.data = data_df
        self.title_to_emb_idx = title_to_emb_idx
        self.user_map = user_map
        self.max_hist_len = max_hist_len
        
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
                
            # Pad/Truncate
            hist_idxs = hist_idxs[-max_hist_len:]
            pad_len = max_hist_len - len(hist_idxs)
            hist_idxs = hist_idxs + [-1] * pad_len # -1 as pad
            
            self.samples.append({
                'u_idx': u_idx,
                'target_idx': int(target_idx),
                'hist_idxs': np.array(hist_idxs)
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
    lr=1e-3
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Load Maps & Embeddings
    print("Loading Embeddings & Maps...")
    item_embeddings = np.load(item_emb_file)
    with open(item_titles_file, 'r', encoding='utf-8') as f:
        titles = [line.strip() for line in f if line.strip()]
    title_to_emb_idx = {t: i for i, t in enumerate(titles)}
    
    with open(os.path.join(cf_teacher_dir, 'user_map.pkl'), 'rb') as f:
        teacher_user_map = pickle.load(f)
    with open(os.path.join(cf_teacher_dir, 'item_map.pkl'), 'rb') as f:
        teacher_item_map = pickle.load(f) # Logic mapping item_id -> teacher_idx
        
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
    dataset = AlignmentDataset(df, title_to_emb_idx, teacher_user_map)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # 3. Models
    student_model = StudentModel(item_embeddings).to(device)
    optimizer = optim.Adam(student_model.parameters(), lr=lr)
    
    # Load Teacher
    # Use dummy counts from map to init
    teacher_model = CFTeacher(len(teacher_user_map), len(teacher_item_map))
    teacher_model.load_state_dict(torch.load(os.path.join(cf_teacher_dir, 'cf_teacher.pt')))
    teacher_model.to(device)
    teacher_model.eval()
    
    print("Starting Alignment Training...")
    
    # Curriculum Schedule (Example)
    # 0-25% epochs: Random
    # 25-50%: Mixed
    # 50%+: Hard
    
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
            
        pbar = tqdm(dataloader, desc=f"Ep {epoch+1} [{strategy}]")
        
        for u_idxs, hist_idxs, target_idxs in pbar:
            # -1 padding handling in student model
            u_idxs, hist_idxs, target_idxs = u_idxs.to(device), hist_idxs.to(device), target_idxs.to(device)
            
            batch_size_curr = u_idxs.size(0)
            
            # --- Negative Sampling ---
            # Default Random
            neg_idxs = torch.randint(0, len(titles), (batch_size_curr,), device=device) # Init random
            
            if strategy != "random":
                num_hard = int(batch_size_curr * ratio)
                if num_hard > 0:
                    with torch.no_grad():
                        # Score on Teacher [num_hard, K_teacher]
                        cf_scores = teacher_model(u_idxs[:num_hard]) 
                        
                        # Top K (e.g. 50)
                        _, topk_t_idxs = torch.topk(cf_scores, k=min(50, len(teacher_item_map)), dim=1) 
                        
                        # Sample 1 from top K per user
                        rand_select = torch.randint(0, topk_t_idxs.size(1), (num_hard, 1), device=device)
                        selected_t_idxs = topk_t_idxs.gather(1, rand_select).squeeze(1)
                        
                        # Convert to Student Indices
                        hard_neg_idxs = teacher_to_student_tensor[selected_t_idxs]
                        
                        # Replace in neg_idxs
                        neg_idxs[:num_hard] = hard_neg_idxs
            
            optimizer.zero_grad()
            pos_scores, _ = student_model(hist_idxs, target_idxs)
            neg_scores, _ = student_model(hist_idxs, neg_idxs) # Negatives use same history context? No, negs are items.
            # Student model needs (hist, item) to score.
            # neg_idxs is just [batch] item indices.
            # We call student forward with same history but negative target.
            
            loss = -torch.mean(torch.nn.functional.logsigmoid(pos_scores - neg_scores))
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            pbar.set_postfix({'loss': loss.item()})
            
        print(f"Epoch {epoch+1} Mean Loss: {total_loss / len(dataloader):.4f}")

    # Save
    os.makedirs(output_dir, exist_ok=True)
    torch.save(student_model.state_dict(), os.path.join(output_dir, "student_aligned.pt"))
    print("Saved Stage 3 Model.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_file", required=True)
    parser.add_argument("--item_emb_file", required=True)
    parser.add_argument("--item_titles_file", required=True)
    parser.add_argument("--cf_teacher_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()
    
    train_alignment(args.train_file, args.item_emb_file, args.item_titles_file, args.cf_teacher_dir, args.output_dir)
