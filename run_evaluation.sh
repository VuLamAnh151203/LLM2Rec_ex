#!/bin/bash

# Configuration
TRUTH_FILE="/kaggle/input/llm2rec-data/CiteULike/cold_item_test.csv"
PRED_FILE="./cold_item_recommendations.csv"

echo "=== Evaluation: Calculating Recall@K ==="

python evaluate_recall.py \
    --truth_file "$TRUTH_FILE" \
    --pred_file "$PRED_FILE" \
    --ks 5 10 20

echo "=== Evaluation Complete ==="
