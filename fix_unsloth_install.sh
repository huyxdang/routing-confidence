#!/bin/bash
#
# Fix Unsloth Installation Issues
# Run this if you get unsloth_zoo errors

echo "=========================================="
echo "Fixing Unsloth Installation"
echo "=========================================="

# Install unsloth_zoo first
echo ""
echo "Step 1: Installing unsloth-zoo..."
pip install unsloth-zoo

# Reinstall unsloth to ensure compatibility
echo ""
echo "Step 2: Reinstalling unsloth..."
pip install --upgrade --force-reinstall --no-cache-dir "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"

# Verify installation
echo ""
echo "Step 3: Verifying installation..."
python -c "from unsloth import FastLanguageModel; print('✓ Unsloth imported successfully')"

echo ""
echo "=========================================="
echo "✓ Installation fixed!"
echo "=========================================="
echo ""
echo "Now you can run training:"
echo "  bash train_fast_4gpu.sh qwen"
echo "=========================================="

