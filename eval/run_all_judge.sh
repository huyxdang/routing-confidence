#!/bin/bash
# Judge all prediction files and add domain-specific confidence tokens

set -e  # Exit on error

echo "======================================="
echo "Judging all predictions"
echo "======================================="
echo ""

# Check for OpenAI API key
if [ -z "$OPENAI_API_KEY" ]; then
    echo "Error: OPENAI_API_KEY environment variable not set"
    echo "Please set it with: export OPENAI_API_KEY='your-key'"
    exit 1
fi

# Configuration
JUDGE_MODEL="gpt-4o-2024-08-06"
NUM_WORKERS=50

# Create judged directory
mkdir -p judged

echo "Using judge model: $JUDGE_MODEL"
echo "Parallel workers: $NUM_WORKERS"
echo ""

# Prediction files
PRED_DIR="../inference/predictions"

# Check if predictions exist
if [ ! -d "$PRED_DIR" ]; then
    echo "Error: Predictions directory not found: $PRED_DIR"
    echo "Please run inference first using run_all_inference.sh"
    exit 1
fi

# Mistral-7B predictions
echo "==================================
"
echo "Judging Mistral-7B predictions"
echo "=================================="

if [ -f "$PRED_DIR/mistral7binstructv03_math_train.json" ]; then
    echo "
[1/6] Judging Mistral-7B MATH..."
    python run_judge_datasets.py \
      --predictions "$PRED_DIR/mistral7binstructv03_math_train.json" \
      --dataset math \
      --judge "$JUDGE_MODEL" \
      --num_workers $NUM_WORKERS
else
    echo "⚠ Warning: Mistral-7B MATH predictions not found, skipping..."
fi

if [ -f "$PRED_DIR/mistral7binstructv03_medqa_train.json" ]; then
    echo "
[2/6] Judging Mistral-7B MedQA..."
    python run_judge_datasets.py \
      --predictions "$PRED_DIR/mistral7binstructv03_medqa_train.json" \
      --dataset medqa \
      --judge "$JUDGE_MODEL" \
      --num_workers $NUM_WORKERS
else
    echo "⚠ Warning: Mistral-7B MedQA predictions not found, skipping..."
fi

if [ -f "$PRED_DIR/mistral7binstructv03_boolq_train.json" ]; then
    echo "
[3/6] Judging Mistral-7B BoolQ..."
    python run_judge_datasets.py \
      --predictions "$PRED_DIR/mistral7binstructv03_boolq_train.json" \
      --dataset boolq \
      --judge "$JUDGE_MODEL" \
      --num_workers $NUM_WORKERS
else
    echo "⚠ Warning: Mistral-7B BoolQ predictions not found, skipping..."
fi

# Qwen2.5-7B predictions
echo "
=================================="
echo "Judging Qwen2.5-7B predictions"
echo "=================================="

if [ -f "$PRED_DIR/qwen257binstruct_math_train.json" ]; then
    echo "
[4/6] Judging Qwen2.5-7B MATH..."
    python run_judge_datasets.py \
      --predictions "$PRED_DIR/qwen257binstruct_math_train.json" \
      --dataset math \
      --judge "$JUDGE_MODEL" \
      --num_workers $NUM_WORKERS
else
    echo "⚠ Warning: Qwen2.5-7B MATH predictions not found, skipping..."
fi

if [ -f "$PRED_DIR/qwen257binstruct_medqa_train.json" ]; then
    echo "
[5/6] Judging Qwen2.5-7B MedQA..."
    python run_judge_datasets.py \
      --predictions "$PRED_DIR/qwen257binstruct_medqa_train.json" \
      --dataset medqa \
      --judge "$JUDGE_MODEL" \
      --num_workers $NUM_WORKERS
else
    echo "⚠ Warning: Qwen2.5-7B MedQA predictions not found, skipping..."
fi

if [ -f "$PRED_DIR/qwen257binstruct_boolq_train.json" ]; then
    echo "
[6/6] Judging Qwen2.5-7B BoolQ..."
    python run_judge_datasets.py \
      --predictions "$PRED_DIR/qwen257binstruct_boolq_train.json" \
      --dataset boolq \
      --judge "$JUDGE_MODEL" \
      --num_workers $NUM_WORKERS
else
    echo "⚠ Warning: Qwen2.5-7B BoolQ predictions not found, skipping..."
fi

echo "
=================================="
echo "All judging completed!"
echo "=================================="
echo ""
echo "Tagged output files in judged/"
ls -lh judged/

