#!/bin/bash

# Configuration
TRAIN_FILE="/kaggle/input/llm2rec-data/CiteULike/llm2rec_processed/train.csv"
ITEM_EMB_FILE="./item_embeddings.npy"
ITEM_TITLES_FILE="/kaggle/input/llm2rec-data/CiteULike/llm2rec_processed/item_titles.txt"
ITEM_INFO_FILE="/kaggle/input/llm2rec-data/CiteULike/item.csv"
CHECKPOINT_PATH="./student_aligned_BPR.pt"
TEST_FILE="/kaggle/input/llm2rec-data/CiteULike/cold_item_test.csv"
OUTPUT_FILE="./cold_item_recommendations.csv"

TOP_K=20
BATCH_SIZE=256
NO_MLP="" # Set to "--no_mlp" for weighted-only mode

echo "=== Stage 4: Inference (User Retrieval for Cold Items) ==="

python inference_stage3.py \
    --train_file "$TRAIN_FILE" \
    --item_emb_file "$ITEM_EMB_FILE" \
    --checkpoint_path "$CHECKPOINT_PATH" \
    --test_file "$TEST_FILE" \
    --output_file "$OUTPUT_FILE" \
    --top_k $TOP_K \
    --batch_size $BATCH_SIZE \
    $NO_MLP

echo "=== Inference Complete ==="
echo "Results saved to: $OUTPUT_FILE"
