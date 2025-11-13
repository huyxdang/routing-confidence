# 🚀 Quick Start - Stable Version (No Unsloth)

## ✅ What's Fixed

- ❌ **Removed Unsloth** (caused segfaults)
- ✅ **Using standard PEFT** (stable, no crashes)
- ✅ **All features working** (gradient masking, validation, calibration)

---

## 🎯 3-Step Quick Start

### Step 1: Test Setup (2 minutes)
```bash
python test_stable_setup.py
```

Expected output:
```
✅ Setup verified! Ready to train.
```

### Step 2: Quick Test (10 minutes, 1K examples)
```bash
python train_confidence_lora.py \
    --model_name "Qwen/Qwen2.5-7B-Instruct" \
    --train_data "tagged/qwen/qwen_small_1k.jsonl" \
    --output_dir "./checkpoints/test_1k" \
    --num_epochs 1 \
    --batch_size 4 \
    --max_steps 50
```

**What this verifies:**
- ✅ No segfaults
- ✅ Model loads
- ✅ Training runs
- ✅ Loss decreases
- ✅ Validation works

### Step 3: Full Training (1-1.5 hours, 4 GPUs)
```bash
bash train_fast_4gpu.sh qwen
```

---

## 📊 Training Time Expectations

| Dataset Size | GPUs | Time | Use Case |
|--------------|------|------|----------|
| **1,000** | 1 | **10 min** | **Debug & verify** |
| 5,000 | 1 | 25 min | Check learning |
| 5,000 | 4 | 8 min | Check learning (fast) |
| **18,663** | 4 | **1-1.5 hrs** | **Full training** |

---

## 🔧 If You Get Errors

### "CUDA out of memory"
```bash
--batch_size 2  # Reduce from 4 or 8
```

### "Cannot load model"
```bash
# Check GPU memory first
nvidia-smi

# Make sure you have enough space
# H200 has 141GB - plenty for 7B model
```

### "Module not found"
```bash
# Install missing dependencies
pip install transformers peft bitsandbytes accelerate
```

---

## 📈 Expected Results

### After 10 minutes (1K examples, testing):
```
Loss: 0.5 → 0.2
Validation Accuracy: ~55-60%
Calibration Error: ~0.20
✅ Verifies everything works!
```

### After 1-1.5 hours (full training):
```
Loss: 0.2 → 0.08
Validation Accuracy: 70-80%
Calibration Error: 0.10-0.15
✅ Production-ready model!
```

---

## 🎯 Recommended Workflow

```bash
# Morning: Quick test
python train_confidence_lora.py \
    --train_data "tagged/qwen/qwen_small_1k.jsonl" \
    --num_epochs 1 \
    --batch_size 4 \
    --max_steps 50 \
    --output_dir "./checkpoints/morning_test"

# Verify loss is decreasing ✓

# Afternoon: Full training
bash train_fast_4gpu.sh qwen

# Check back in 1-1.5 hours ✓
```

---

## 💡 Key Commands

### Single GPU (slower but simpler)
```bash
python train_confidence_lora.py \
    --model_name "Qwen/Qwen2.5-7B-Instruct" \
    --train_data "tagged/qwen/qwen_all_tagged.jsonl" \
    --output_dir "./checkpoints/qwen_single" \
    --num_epochs 2 \
    --batch_size 8
```

### Multi-GPU (faster, recommended)
```bash
accelerate launch \
    --num_processes 4 \
    train_confidence_lora.py \
    --model_name "Qwen/Qwen2.5-7B-Instruct" \
    --train_data "tagged/qwen/qwen_all_tagged.jsonl" \
    --output_dir "./checkpoints/qwen_multi" \
    --num_epochs 2 \
    --batch_size 16
```

---

## ✅ Verification Checklist

Before full training:

- [ ] `python test_stable_setup.py` passes
- [ ] `nvidia-smi` shows all GPUs
- [ ] Quick test (1K, 10 min) completes without errors
- [ ] Loss is decreasing
- [ ] Validation runs

Then proceed with full training!

---

## 📚 Documentation

- **`STABLE_VERSION_INFO.md`**: Technical details of changes
- **`README_TRAINING.md`**: Full training guide
- **`FAST_TRAINING_GUIDE.md`**: Multi-GPU optimization guide
- **`train_confidence_lora.py`**: Main training script (no Unsloth)

---

**🚀 Ready to train! Start with the 10-minute test, then run full training.**

