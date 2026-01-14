
import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import argparse
from tqdm import tqdm
import pickle

class BPRMF(nn.Module):
    def __init__(self, num_users, num_items, embedding_dim=64):
        super(BPRMF, self).__init__()
        self.user_emb = nn.Embedding(num_users, embedding_dim)
        self.item_emb = nn.Embedding(num_items, embedding_dim)
        
        # Init small random weights
        nn.init.normal_(self.user_emb.weight, std=0.01)
        nn.init.normal_(self.item_emb.weight, std=0.01)

    def forward(self, u, i):
        return (self.user_emb(u) * self.item_emb(i)).sum(dim=-1)

class InteractionDataset(Dataset):
    def __init__(self, user_ids, item_ids, num_items, num_users, neg_samples=1):
        self.users = torch.tensor(user_ids, dtype=torch.long)
        self.items = torch.tensor(item_ids, dtype=torch.long)
        self.num_items = num_items
        self.num_users = num_users
        self.neg_samples = neg_samples
        
        # Build interaction set for fast checking
        self.interacted = set(zip(user_ids, item_ids))

    def __len__(self):
        return len(self.users)

    def __getitem__(self, idx):
        u = self.users[idx]
        i = self.items[idx]
        
        # Simple random negative sampling
        neg_i = np.random.randint(0, self.num_items)
        # Fast skipping loop
        while (u.item(), neg_i) in self.interacted:
            neg_i = np.random.randint(0, self.num_items)
            
        return u, i, torch.tensor(neg_i, dtype=torch.long)

def train_cf_teacher(train_file, output_dir, embedding_dim=64, batch_size=1024, epochs=10, lr=1e-3):
    print(f"Loading {train_file}...")
    df = pd.read_csv(train_file)
    
    # Needs user_id and item_id
    if 'user_id' not in df.columns:
        raise ValueError("train.csv missing user_id column")
        
    # Map IDs to 0..N
    unique_users = df['user_id'].unique()
    unique_items = df['item_id'].unique()
    
    # We might need to handle items not in train but in item_titles.txt?
    # Stage 3 uses alignment. Hard negatives must be from the set of items we have embeddings for.
    # We should probably map ALL items found in item_titles.txt or similar to ensure coverage.
    # But for a simple teacher, training on train items is sufficient.
    
    user_map = {u: i for i, u in enumerate(unique_users)}
    item_map = {i: idx for idx, i in enumerate(unique_items)}
    
    print(f"Users: {len(user_map)}, Items: {len(item_map)}")
    
    # Save maps
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, 'user_map.pkl'), 'wb') as f:
        pickle.dump(user_map, f)
    with open(os.path.join(output_dir, 'item_map.pkl'), 'wb') as f:
        pickle.dump(item_map, f)
        
    # Prepare Data
    user_ids = df['user_id'].map(user_map).values
    item_ids = df['item_id'].map(item_map).values
    
    dataset = InteractionDataset(user_ids, item_ids, len(item_map), len(user_map))
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BPRMF(len(user_map), len(item_map), embedding_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    print("Training BPR-MF Teacher...")
    for epoch in range(epochs):
        total_loss = 0
        for u, i_pos, i_neg in tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}"):
            u, i_pos, i_neg = u.to(device), i_pos.to(device), i_neg.to(device)
            
            optimizer.zero_grad()
            pos_scores = model(u, i_pos)
            neg_scores = model(u, i_neg)
            
            # BPR Loss
            loss = -torch.mean(torch.nn.functional.logsigmoid(pos_scores - neg_scores))
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        print(f"Epoch {epoch+1} loss: {total_loss / len(dataloader):.4f}")
        
    # Save Model
    torch.save(model.state_dict(), os.path.join(output_dir, 'cf_teacher.pt'))
    print("Saved CF Teacher.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_file", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=20)
    args = parser.parse_args()
    
    train_cf_teacher(args.train_file, args.output_dir, epochs=args.epochs)
