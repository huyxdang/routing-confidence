#!/bin/bash
# Judge all prediction files using simple pattern matching (no LLM/API needed)
# Much faster and free compared to OpenAI approach!

set -e  # Exit on error

echo "======================================="
echo "Judging all predictions (Pattern-based)"
echo "======================================="
echo ""

# Prediction files
PRED_DIR="../inference/predictions"

# Check if predictions exist
if [ ! -d "$PRED_DIR" ]; then
    echo "Error: Predictions directory not found: $PRED_DIR"
    echo "Please run inference first using run_all_inference.sh"
    exit 1
fi

echo "Using pattern-based extraction (no API calls needed)"
echo "This is MUCH faster than LLM-as-judge!"
echo ""

# Mistral-7B predictions
echo "==================================
"
echo "Judging Mistral-7B predictions"
echo "=================================="

if [ -f "$PRED_DIR/math/mistral_math_train.json" ]; then
    echo "
[1/6] Judging Mistral-7B MATH..."
    python run_judge_datasets_simple.py \
      --predictions "$PRED_DIR/math/mistral_math_train.json" \
      --dataset math
else
    echo "⚠ Warning: Mistral-7B MATH predictions not found, skipping..."
fi

if [ -f "$PRED_DIR/medqa/mistral_medqa_train.json" ]; then
    echo "
[2/6] Judging Mistral-7B MedQA..."
    python run_judge_datasets_simple.py \
      --predictions "$PRED_DIR/medqa/mistral_medqa_train.json" \
      --dataset medqa
else
    echo "⚠ Warning: Mistral-7B MedQA predictions not found, skipping..."
fi

if [ -f "$PRED_DIR/boolq/mistral_boolq_train.json" ]; then
    echo "
[3/6] Judging Mistral-7B BoolQ..."
    python run_judge_datasets_simple.py \
      --predictions "$PRED_DIR/boolq/mistral_boolq_train.json" \
      --dataset boolq
else
    echo "⚠ Warning: Mistral-7B BoolQ predictions not found, skipping..."
fi

# Qwen2.5-7B predictions
echo "
=================================="
echo "Judging Qwen2.5-7B predictions"
echo "=================================="

if [ -f "$PRED_DIR/math/qwen_math_train.json" ]; then
    echo "
[4/6] Judging Qwen2.5-7B MATH..."
    python run_judge_datasets_simple.py \
      --predictions "$PRED_DIR/math/qwen_math_train.json" \
      --dataset math
else
    echo "⚠ Warning: Qwen2.5-7B MATH predictions not found, skipping..."
fi

if [ -f "$PRED_DIR/medqa/qwen_medqa_train.json" ]; then
    echo "
[5/6] Judging Qwen2.5-7B MedQA..."
    python run_judge_datasets_simple.py \
      --predictions "$PRED_DIR/medqa/qwen_medqa_train.json" \
      --dataset medqa
else
    echo "⚠ Warning: Qwen2.5-7B MedQA predictions not found, skipping..."
fi

if [ -f "$PRED_DIR/boolq/qwen_boolq_train.json" ]; then
    echo "
[6/6] Judging Qwen2.5-7B BoolQ..."
    python run_judge_datasets_simple.py \
      --predictions "$PRED_DIR/boolq/qwen_boolq_train.json" \
      --dataset boolq
else
    echo "⚠ Warning: Qwen2.5-7B BoolQ predictions not found, skipping..."
fi

echo "
=================================="
echo "All judging completed!"
echo "=================================="
echo ""
echo "Tagged output files created in prediction directories"
echo "(Look for *_tagged.json files)"

