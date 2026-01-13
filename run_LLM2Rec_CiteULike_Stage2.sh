# Stage 2: Item-level Embedding Modeling (MNTP + SimCSE)
# Assumes Stage 1 has completed and saved to ./output/Qwen2-0.5B-CSFT-CiteULike

# 1. Train MNTP
echo "Starting MNTP Training..."
# Torchrun with 1 Node (adjust if needed)
torchrun --nproc_per_node=1 --master_port=29501 \
    ./llm2rec/run_mntp.py \
    ./llm2rec/train_mntp_citeulike.json

# 2. Train SimCSE
# Note: SimCSE config assumes it picks up model from ./output/iem_stage1/...
echo "Starting SimCSE Training..."
torchrun --nproc_per_node=1 --master_port=29502 \
    ./llm2rec/run_unsupervised_SimCSE.py \
    ./llm2rec/train_simcse_citeulike.json
