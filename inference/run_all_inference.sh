#!/bin/bash
# Run inference on all datasets for both models

set -e  # Exit on error

echo "=================================="
echo "Running inference on all datasets"
echo "=================================="
echo ""

# Configuration
TENSOR_PARALLEL=1
BATCH_SIZE=50

# Models
LLAMA3="meta-llama/Meta-Llama-3-8B-Instruct"
QWEN25="Qwen/Qwen2.5-7B-Instruct"

# Create predictions directory
mkdir -p predictions

echo "Starting inference runs..."
echo ""

# Llama3-8B on all datasets
echo "==================================
"
echo "Llama3-8B-Instruct"
echo "=================================="

echo "
[1/6] Llama3-8B on MATH..."
python run_dataset_inference.py \
  --model_name "$LLAMA3" \
  --dataset math \
  --split train \
  --tensor_parallel_size $TENSOR_PARALLEL \
  --batch_size $BATCH_SIZE

echo "
[2/6] Llama3-8B on MedQA..."
python run_dataset_inference.py \
  --model_name "$LLAMA3" \
  --dataset medqa \
  --split train \
  --tensor_parallel_size $TENSOR_PARALLEL \
  --batch_size $BATCH_SIZE

echo "
[3/6] Llama3-8B on BoolQ..."
python run_dataset_inference.py \
  --model_name "$LLAMA3" \
  --dataset boolq \
  --split train \
  --tensor_parallel_size $TENSOR_PARALLEL \
  --batch_size $BATCH_SIZE

# Qwen2.5-7B on all datasets
echo "
=================================="
echo "Qwen2.5-7B-Instruct"
echo "=================================="

echo "
[4/6] Qwen2.5-7B on MATH..."
python run_dataset_inference.py \
  --model_name "$QWEN25" \
  --dataset math \
  --split train \
  --tensor_parallel_size $TENSOR_PARALLEL \
  --batch_size $BATCH_SIZE

echo "
[5/6] Qwen2.5-7B on MedQA..."
python run_dataset_inference.py \
  --model_name "$QWEN25" \
  --dataset medqa \
  --split train \
  --tensor_parallel_size $TENSOR_PARALLEL \
  --batch_size $BATCH_SIZE

echo "
[6/6] Qwen2.5-7B on BoolQ..."
python run_dataset_inference.py \
  --model_name "$QWEN25" \
  --dataset boolq \
  --split train \
  --tensor_parallel_size $TENSOR_PARALLEL \
  --batch_size $BATCH_SIZE

echo "
=================================="
echo "All inference runs completed!"
echo "=================================="
echo ""
echo "Output files in predictions/"
ls -lh predictions/

