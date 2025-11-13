"""
Quick test script to verify stable setup works (no Unsloth)
"""

print("Testing stable setup (no Unsloth)...")
print("=" * 60)

# Test 1: Check imports
print("\n1. Testing imports...")
try:
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    print("   ✓ Transformers imported")
except ImportError as e:
    print(f"   ✗ Transformers import failed: {e}")
    exit(1)

try:
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    print("   ✓ PEFT imported")
except ImportError as e:
    print(f"   ✗ PEFT import failed: {e}")
    exit(1)

try:
    import torch
    print("   ✓ PyTorch imported")
    print(f"   ✓ CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"   ✓ GPU count: {torch.cuda.device_count()}")
except ImportError as e:
    print(f"   ✗ PyTorch import failed: {e}")
    exit(1)

try:
    import bitsandbytes
    print("   ✓ BitsAndBytes imported")
except ImportError as e:
    print(f"   ✗ BitsAndBytes import failed: {e}")
    print("   → Install with: pip install bitsandbytes")
    exit(1)

# Test 2: Check training script syntax
print("\n2. Testing training script syntax...")
try:
    import train_confidence_lora
    print("   ✓ Training script imports successfully")
    print(f"   ✓ Confidence tokens defined: {train_confidence_lora.CONFIDENCE_TOKENS}")
except Exception as e:
    print(f"   ✗ Training script import failed: {e}")
    exit(1)

# Test 3: Check data files
print("\n3. Checking data files...")
import os

data_files = [
    "tagged/qwen/qwen_all_tagged.jsonl",
    "tagged/qwen/qwen_small_1k.jsonl",
    "tagged/mistral/mistral_all_tagged.jsonl",
]

for file_path in data_files:
    if os.path.exists(file_path):
        size_mb = os.path.getsize(file_path) / (1024 * 1024)
        print(f"   ✓ {file_path} ({size_mb:.1f} MB)")
    else:
        print(f"   ⚠ {file_path} not found")

print("\n" + "=" * 60)
print("✅ Setup verified! Ready to train.")
print("=" * 60)
print("\nQuick test command:")
print("  python train_confidence_lora.py \\")
print("      --model_name 'Qwen/Qwen2.5-7B-Instruct' \\")
print("      --train_data 'tagged/qwen/qwen_small_1k.jsonl' \\")
print("      --output_dir './checkpoints/test' \\")
print("      --num_epochs 1 \\")
print("      --batch_size 4 \\")
print("      --max_steps 10")

