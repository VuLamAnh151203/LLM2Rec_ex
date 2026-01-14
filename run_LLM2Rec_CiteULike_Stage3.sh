# Stage 3: User-Item Alignment
# Assumes Stage 2 has completed (SimCSE model in output/iem_stage2/...)

# Paths
INPUT_MODEL="output/iem_stage2/Qwen2-0.5B-CiteULike-CSFT"
DATA_DIR="../ColdLLM-repo/data/CiteULike/llm2rec_processed"
STAGE3_DIR="./output/stage3_alignment"
TEACHER_DIR="${STAGE3_DIR}/cf_teacher"

echo "=== Stage 3.1: Extract Static Embeddings ==="
python extract_embeddings.py \
    --model_path "${INPUT_MODEL}" \
    --data_path "${DATA_DIR}/item_titles.txt" \
    --output_path "${STAGE3_DIR}/item_embeddings.npy" \
    --batch_size 128

echo "=== Stage 3.2: Train CF Teacher (Hard Negatives) ==="
python train_cf_teacher.py \
    --train_file "${DATA_DIR}/train.csv" \
    --output_dir "${TEACHER_DIR}" \
    --epochs 20

echo "=== Stage 3.3: Train Alignment Model (Curriculum Learning) ==="
python run_stage3.py \
    --train_file "${DATA_DIR}/train.csv" \
    --item_emb_file "${STAGE3_DIR}/item_embeddings.npy" \
    --item_titles_file "${DATA_DIR}/item_titles.txt" \
    --cf_teacher_dir "${TEACHER_DIR}" \
    --output_dir "${STAGE3_DIR}/student_model"

echo "Stage 3 Complete!"
