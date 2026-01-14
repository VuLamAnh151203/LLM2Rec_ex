import pandas as pd
import ast
import argparse
import numpy as np

def calculate_recall_at_k(truth_file, pred_file, ks=[5, 10, 20]):
    print(f"Loading ground truth from {truth_file}...")
    df_truth = pd.read_csv(truth_file)
    
    # Robust column detection for truth file
    user_col_truth = 'user' if 'user' in df_truth.columns else 'user_id'
    item_col_truth = 'item' if 'item' in df_truth.columns else 'item_id'
    
    # Group truth by item: {item_id: {user1, user2, ...}}
    ground_truth = df_truth.groupby(item_col_truth)[user_col_truth].apply(set).to_dict()
    
    print(f"Loading predictions from {pred_file}...")
    df_pred = pd.read_csv(pred_file)
    
    # The output from inference_stage3.py has 'item_id' and 'top_user_ids'
    # 'top_user_ids' is stored as a string representation of a list
    
    results = {k: [] for k in ks}
    
    print(f"Evaluating Recall for {len(df_pred)} items...")
    
    for _, row in df_pred.iterrows():
        item_id = row['item_id']
        
        # Check if we have ground truth for this item
        if item_id not in ground_truth:
            continue
            
        true_users = ground_truth[item_id]
        
        # Parse predicted users
        pred_users_raw = row['top_user_ids']
        if isinstance(pred_users_raw, str):
            pred_users = ast.literal_eval(pred_users_raw)
        else:
            pred_users = pred_users_raw
            
        for k in ks:
            top_k_preds = set(pred_users[:k])
            hits = len(top_k_preds.intersection(true_users))
            recall = hits / len(true_users) if len(true_users) > 0 else 0
            results[k].append(recall)
            
    print("\n=== Evaluation Results ===")
    for k in ks:
        if results[k]:
            avg_recall = np.mean(results[k])
            print(f"Recall@{k}: {avg_recall:.4f}")
        else:
            print(f"Recall@{k}: No data")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--truth_file", required=True, help="Path to cold_item_test.csv")
    parser.add_argument("--pred_file", required=True, help="Path to cold_item_recommendations.csv")
    parser.add_argument("--ks", nargs="+", type=int, default=[5, 10, 20], help="List of K values for Recall@K")
    args = parser.parse_args()
    
    calculate_recall_at_k(args.truth_file, args.pred_file, args.ks)
