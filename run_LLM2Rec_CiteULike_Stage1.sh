# Stage 1: Collaborative Supervised Fine-Tuning (CSFT) for CiteULike

# Use a local path or HuggingFace ID
model_path="Qwen/Qwen2-0.5B" 

# Relative path to the processed data we just created
# Assumes this script is run from LLM2Rec directory
train_file="../ColdLLM-repo/data/CiteULike/llm2rec_processed/train.csv"
eval_file="../ColdLLM-repo/data/CiteULike/llm2rec_processed/valid.csv"
output_dir="./output/Qwen2-0.5B-CSFT-CiteULike"

echo "Training with: ${train_file}"

# Run training
# Adjust nproc_per_node based on your GPU count
torchrun --nproc_per_node 1 --master_port=29500 \
    ./llm2rec/run_csft.py \
    --base_model "${model_path}" \
    --train_file "${train_file}" \
    --eval_file "${eval_file}" \
    --output_dir "${output_dir}" \
    --wandb_run_name "Qwen2-0.5B-CSFT-CiteULike" \
    --category "CiteULike" \
    --train_from_scratch False \
    --use_lora False \
    --learning_rate 3e-4 \
    --num_epochs 100 \
    --batch_size 128 \
    --micro_batch_size 4

# Copy tokenizer to output for next stages
echo "Copying tokenizer files..."
# Note: If model_path is a local directory, copy from there. 
# If it's a Hub ID, the tokenizer is saved by the script in output_dir mostly, but let's be safe.
# The script `run_csft.py` saves the tokenizer/model at the end.
