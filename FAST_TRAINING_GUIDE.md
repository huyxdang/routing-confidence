# ⚡ Fast Training Guide - 1 Hour on 4x H200 GPUs

Get your training done in ~1 hour instead of 6-9 hours!

## 🎯 Target Timeline

| Setup | GPUs | Batch Size | Epochs | Time |
|-------|------|------------|--------|------|
| Original | 1x H200 | 8 | 3 | 6-9 hours |
| **Fast** | **4x H200** | **16×4=64** | **2** | **~1 hour** |

**Speedup factors:**
- 4x GPUs → 4x faster
- 2 epochs instead of 3 → 1.5x faster  
- Larger effective batch size → Better GPU utilization
- **Total: ~6-7x speedup → ~1 hour**

---

## 🚀 Quick Start

### Option 1: Use the Fast Training Script (Easiest)

```bash
# Make executable
chmod +x train_fast_4gpu.sh

# Train Qwen
bash train_fast_4gpu.sh qwen

# Train Mistral
bash train_fast_4gpu.sh mistral
```

### Option 2: Manual Command

```bash
accelerate launch \
    --config_file train_config_4gpu.yaml \
    --num_processes 4 \
    train_confidence_lora.py \
    --model_name "Qwen/Qwen2.5-7B-Instruct" \
    --train_data "tagged/qwen/qwen_all_tagged.jsonl" \
    --output_dir "./checkpoints/qwen_fast" \
    --num_epochs 2 \
    --batch_size 16 \
    --gradient_accumulation_steps 2 \
    --learning_rate 3e-4 \
    --eval_steps 500 \
    --bf16
```

---

## ⚙️ Optimization Settings

### 1. Multi-GPU Configuration

**File: `train_config_4gpu.yaml`** (already created)
```yaml
num_processes: 4      # 4 GPUs
distributed_type: MULTI_GPU
mixed_precision: bf16
```

### 2. Batch Size Optimization

**Per GPU:** 16  
**Total GPUs:** 4  
**Gradient Accumulation:** 2  
**Effective Batch Size:** 16 × 4 × 2 = **128**

This is 16x larger than the original (8 × 1 × 1 = 8), allowing:
- Better gradient estimates
- Fewer update steps needed
- Better GPU utilization

### 3. Reduced Epochs

**Original:** 3 epochs  
**Fast:** 2 epochs  

With 8x larger effective batch size, each epoch is more efficient, so 2 epochs is sufficient.

### 4. Adjusted Learning Rate

**Original:** 2e-4  
**Fast:** 3e-4  

With larger batch size, we can use higher learning rate (following the linear scaling rule).

### 5. Reduced Validation Frequency

**Original:** Every 500 steps  
**Fast:** Every 500 steps (but with 4x speedup, this is effectively 2000 original steps)

### 6. Optimized Validation Size

Modify `ValidationCallback` to use fewer samples:

```python
# In train_confidence_lora.py, line ~515
results = self._validate_dataset(
    val_dataset,
    dataset_name,
    max_samples=200  # Reduced from 500
)
```

---

## 📊 Expected Performance

### Timeline Breakdown (1 Hour Total)

| Phase | Time | Details |
|-------|------|---------|
| **Model Loading** | 3-5 min | Load + tokenizer + LoRA setup |
| **Epoch 1** | 25-28 min | Training + validation |
| **Epoch 2** | 25-28 min | Training + validation |
| **Final Saving** | 2-3 min | Save checkpoints + push to HF |
| **Total** | **~60 min** | |

### Memory Usage (per GPU)

**With batch_size=16 on each GPU:**
- Model (8-bit): ~8 GB
- Activations + gradients: ~20-25 GB
- **Total: ~30-35 GB per GPU**
- **H200 has 141 GB → plenty of headroom**

### Training Metrics

After 2 epochs:
- **Validation Accuracy:** 65-75% (slightly lower than 3 epochs)
- **Calibration Error:** 0.12-0.18 (slightly higher than 3 epochs)
- Still good enough for most applications!

---

## 🔧 Setup Instructions

### 1. Install Accelerate

```bash
pip install accelerate
```

### 2. Configure Accelerate

```bash
# Option A: Use provided config
cp train_config_4gpu.yaml ~/.cache/huggingface/accelerate/default_config.yaml

# Option B: Generate interactively
accelerate config
# Select:
# - Multi-GPU
# - 4 GPUs
# - bf16
# - No DeepSpeed
```

### 3. Verify GPU Setup

```bash
# Check all GPUs visible
nvidia-smi

# Should show 4 H200 GPUs
```

### 4. Test Multi-GPU

```bash
# Quick test (10 steps)
accelerate launch \
    --num_processes 4 \
    train_confidence_lora.py \
    --model_name "Qwen/Qwen2.5-7B-Instruct" \
    --train_data "tagged/qwen/qwen_all_tagged.jsonl" \
    --output_dir "./checkpoints/test_multi_gpu" \
    --num_epochs 1 \
    --batch_size 4 \
    --max_steps 10

# Should see:
# "Distributed training with 4 processes"
```

---

## 💡 Advanced Optimizations

### Further Speed Boost (Target: 45 minutes)

If you need even faster training:

#### 1. Increase Batch Size More

```bash
--batch_size 24  # Up from 16
--gradient_accumulation_steps 2
# Effective: 24 × 4 × 2 = 192
```

**Memory check:**
```bash
# Monitor during training
watch -n 1 nvidia-smi
```

#### 2. Reduce Validation Overhead

```python
# In ValidationCallback.__init__()
max_samples=100  # Down from 200
```

#### 3. Flash Attention 2

```python
# In initialize_model_with_tokens()
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=model_name,
    max_seq_length=max_seq_length,
    load_in_8bit=True,
    use_flash_attention_2=True,  # Add this
)
```

#### 4. Only 1 Epoch (if acceptable)

```bash
--num_epochs 1  # 30 minutes total
```

---

## 📈 Scaling Guidelines

| GPUs | Batch Size/GPU | Grad Accum | Effective BS | Est. Time | LR |
|------|----------------|------------|--------------|-----------|-----|
| 1 | 8 | 1 | 8 | 6-9 hours | 2e-4 |
| 2 | 12 | 2 | 48 | 2-3 hours | 2.5e-4 |
| 4 | 16 | 2 | 128 | **1 hour** | **3e-4** |
| 8 | 16 | 2 | 256 | 30-40 min | 4e-4 |

**Learning Rate Scaling Rule:**
```
new_lr = base_lr × sqrt(new_batch_size / base_batch_size)
new_lr = 2e-4 × sqrt(128 / 8) = 2e-4 × 4 = 8e-4
```

We use 3e-4 (slightly conservative) for stability.

---

## 🔍 Monitoring Multi-GPU Training

### 1. GPU Utilization

```bash
# Real-time monitoring
watch -n 0.5 nvidia-smi

# Look for:
# - All 4 GPUs at ~95%+ utilization
# - Memory usage ~30-35 GB per GPU
# - Temperature stable
```

### 2. Training Logs

```bash
# Follow logs
tail -f checkpoints/qwen_fast/logs/events.out.tfevents.*

# Or use TensorBoard
tensorboard --logdir checkpoints/qwen_fast/logs --port 6006
```

### 3. Process Distribution

```bash
# Check processes on GPUs
ps aux | grep train_confidence_lora

# Should see 4 Python processes
```

---

## ⚠️ Troubleshooting

### "Out of Memory" on Multi-GPU

```bash
# Reduce per-GPU batch size
--batch_size 12  # Down from 16

# Or reduce sequence length
--max_seq_length 1024  # Down from 2048
```

### "NCCL Error" or GPU Communication Issues

```bash
# Set environment variables
export NCCL_DEBUG=INFO
export NCCL_P2P_DISABLE=1  # Disable peer-to-peer if issues

# Then run training
```

### Slow Startup

Normal! Multi-GPU initialization takes 3-5 minutes:
- Loading model on each GPU
- Setting up distributed communication
- NCCL initialization

### Uneven GPU Utilization

If GPU 0 is at 100% but others at 50%:

```bash
# Use more data workers
--dataloader_num_workers 4

# Or increase gradient accumulation
--gradient_accumulation_steps 4
```

### Validation Taking Too Long

Reduce validation samples:

```python
# In train_confidence_lora.py, ValidationCallback._validate_dataset()
num_samples = min(len(val_dataset), 100)  # Down from 500
```

---

## 📊 Comparison: 1 GPU vs 4 GPU

### Training Configuration

| Setting | 1 GPU (Slow) | 4 GPUs (Fast) |
|---------|--------------|---------------|
| Batch size per GPU | 8 | 16 |
| Gradient accumulation | 1 | 2 |
| **Effective batch size** | **8** | **128** |
| Epochs | 3 | 2 |
| Learning rate | 2e-4 | 3e-4 |
| Validation samples | 500 | 200 |
| **Total time** | **6-9 hours** | **~1 hour** |

### Expected Metrics

| Metric | 1 GPU (3 epochs) | 4 GPUs (2 epochs) |
|--------|------------------|-------------------|
| Val Accuracy | 70-80% | 65-75% |
| Calib Error | 0.10-0.15 | 0.12-0.18 |
| Training Loss | 0.05-0.10 | 0.08-0.12 |

**Note:** 4-GPU setup is slightly less optimized (fewer epochs) but still achieves good performance.

---

## 🎯 Recommended Workflow

### For Quick Iteration (Development)

```bash
# 1 epoch, 30 minutes
bash train_fast_4gpu.sh qwen --num_epochs 1
```

### For Production Model (Best Quality)

```bash
# 2 epochs, 1 hour (balanced)
bash train_fast_4gpu.sh qwen
```

### For Maximum Quality (If Time Allows)

```bash
# 3 epochs, 1.5 hours
bash train_fast_4gpu.sh qwen --num_epochs 3
```

---

## 📝 Summary

### What Changes for 1-Hour Training?

✅ **Use 4 GPUs** with `accelerate launch`  
✅ **Increase batch size**: 16 per GPU (128 effective)  
✅ **Reduce epochs**: 2 instead of 3  
✅ **Increase learning rate**: 3e-4 instead of 2e-4  
✅ **Reduce validation**: 200 samples instead of 500  

### Commands

```bash
# Setup (one time)
pip install accelerate
cp train_config_4gpu.yaml ~/.cache/huggingface/accelerate/default_config.yaml

# Train (1 hour)
bash train_fast_4gpu.sh qwen
```

### Expected Result

- **Time:** ~1 hour (vs 6-9 hours)
- **Quality:** 65-75% accuracy (vs 70-80%)
- **Still production-ready!**

---

## 🚀 Ready to Go!

```bash
# Make script executable
chmod +x train_fast_4gpu.sh

# Start training
bash train_fast_4gpu.sh qwen

# Monitor
watch -n 1 nvidia-smi
```

**Your model will be ready in ~1 hour!** ⚡

