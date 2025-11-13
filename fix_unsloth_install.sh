#!/bin/bash
#
# Fix Unsloth Installation Issues (SIMPLE VERSION)
# Run this if you get unsloth_zoo errors

echo "=========================================="
echo "Fixing Unsloth Installation (Simple Fix)"
echo "=========================================="

# Just install the missing package - that's all!
echo ""
echo "Installing unsloth-zoo (the only missing package)..."
pip install unsloth-zoo

# Verify installation
echo ""
echo "Verifying installation..."
python -c "from unsloth import FastLanguageModel; print('✓ Unsloth imported successfully')"

echo ""
echo "=========================================="
echo "✓ Installation fixed!"
echo "=========================================="
echo ""
echo "Now you can run training:"
echo "  bash train_fast_4gpu.sh qwen"
echo "=========================================="

