#!/bin/bash
#
# Fast training script for 4x H200 GPUs
# Target: ~1 hour training time
#
# Usage:
#   bash train_fast_4gpu.sh qwen
#   bash train_fast_4gpu.sh mistral

MODEL_CHOICE=${1:-qwen}

if [ "$MODEL_CHOICE" = "qwen" ]; then
    MODEL_NAME="Qwen/Qwen2.5-7B-Instruct"
    TRAIN_DATA="tagged/qwen/qwen_all_tagged.jsonl"
    OUTPUT_DIR="./checkpoints/qwen_confidence_fast"
    HF_REPO="huyxdang/qwen-confidence-lora"
elif [ "$MODEL_CHOICE" = "mistral" ]; then
    MODEL_NAME="mistralai/Mistral-7B-Instruct-v0.2"
    TRAIN_DATA="tagged/mistral/mistral_all_tagged.jsonl"
    OUTPUT_DIR="./checkpoints/mistral_confidence_fast"
    HF_REPO="huyxdang/mistral-confidence-lora"
else
    echo "Usage: bash train_fast_4gpu.sh [qwen|mistral]"
    exit 1
fi

echo "========================================"
echo "Fast 1-Hour Training (4x H200)"
echo "========================================"
echo "Model: $MODEL_NAME"
echo "Data: $TRAIN_DATA"
echo "Output: $OUTPUT_DIR"
echo "========================================"

# Multi-GPU training with accelerate
accelerate launch \
    --config_file train_config_4gpu.yaml \
    --num_processes 4 \
    train_confidence_lora.py \
    --model_name "$MODEL_NAME" \
    --train_data "$TRAIN_DATA" \
    --output_dir "$OUTPUT_DIR" \
    --hf_repo "$HF_REPO" \
    --num_epochs 2 \
    --batch_size 16 \
    --gradient_accumulation_steps 2 \
    --learning_rate 3e-4 \
    --warmup_ratio 0.05 \
    --weight_decay 0.01 \
    --max_grad_norm 1.0 \
    --logging_steps 25 \
    --save_steps 500 \
    --eval_steps 500 \
    --calib_beta 100 \
    --bf16 \
    --seed 42

echo ""
echo "========================================"
echo "Training complete!"
echo "Best models saved in: $OUTPUT_DIR"
echo "========================================"

