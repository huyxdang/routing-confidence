# LoRA Fine-tuning for Domain-Specific Confidence Tokens

Train language models to append confidence tokens (`<C_READ>`, `<U_READ>`, `<C_MED>`, `<U_MED>`) after their predictions, with domain-specific confidence calibration.

## Overview

This training script fine-tunes 7B parameter models using:
- **LoRA (Low-Rank Adaptation)** for parameter-efficient training
- **Unsloth** framework for 2x faster training
- **8-bit quantization** for H200 GPU (141GB)
- **Gradient masking** to compute loss ONLY on confidence tokens
- **On-the-fly validation** with calibration error metrics

### What We're Training

The model learns to predict:
```
Input:  "Question: What is 2+2?"
Output: "4 <C_READ>"
         ↑  ↑
      answer  confidence token (what we optimize)
```

**Key Innovation**: Loss is computed ONLY on the confidence token, not the answer.

---

## Installation

### 1. Environment Setup

```bash
# Create conda environment
conda create -n confidence-train python=3.10 -y
conda activate confidence-train

# Install CUDA toolkit (for H200)
conda install nvidia/label/cuda-12.1.0::cuda-toolkit -y
```

### 2. Install Dependencies

```bash
pip install -r requirements_training.txt
```

### 3. Verify Installation

```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}')"
python -c "from unsloth import FastLanguageModel; print('Unsloth loaded successfully')"
```

---

## Quick Start

### Basic Training

```bash
python train_confidence_lora.py \
    --model_name "Qwen/Qwen2.5-7B-Instruct" \
    --train_data "tagged/qwen/qwen_all_tagged.jsonl" \
    --output_dir "./checkpoints/qwen_confidence" \
    --num_epochs 3 \
    --batch_size 8
```

### Training with HuggingFace Hub Upload

```bash
python train_confidence_lora.py \
    --model_name "mistralai/Mistral-7B-Instruct-v0.2" \
    --train_data "tagged/mistral/mistral_all_tagged.jsonl" \
    --output_dir "./checkpoints/mistral_confidence" \
    --hf_repo "huyxdang/mistral-confidence-lora" \
    --num_epochs 3 \
    --batch_size 8 \
    --learning_rate 2e-4
```

---

## Data Format

### Training Data (JSONL)

Expected format: `tagged/{model_name}/{model_name}_all_tagged.jsonl`

```json
{"question": "Is the sky blue?", "tagged_response": "Yes <C_READ>", "dataset": "boolq"}
{"question": "What is the treatment?", "tagged_response": "A: Medication <C_MED>", "dataset": "medqa"}
```

**Fields:**
- `question`: The input question
- `tagged_response`: Model's answer + confidence token
- `dataset`: Dataset name ("boolq" or "medqa")

### Validation Data

Automatically loaded from HuggingFace:
- `huyxdang/medqa-split` (validation split)
- `huyxdang/boolq-split` (validation split)

These contain raw questions/answers WITHOUT confidence tokens.

---

## Configuration

### Model Selection

```bash
--model_name "Qwen/Qwen2.5-7B-Instruct"
# or
--model_name "mistralai/Mistral-7B-Instruct-v0.2"
```

### LoRA Configuration

```bash
--lora_r 16              # LoRA rank (higher = more parameters)
--lora_alpha 32          # LoRA scaling factor
--lora_dropout 0.05      # Dropout for LoRA layers
```

**Default target modules:**
- `q_proj`, `k_proj`, `v_proj`, `o_proj` (attention)
- `gate_proj`, `up_proj`, `down_proj` (MLP)

### Training Hyperparameters

```bash
--num_epochs 3                    # Number of training epochs
--batch_size 8                    # Per-device batch size
--gradient_accumulation_steps 1   # Gradient accumulation
--learning_rate 2e-4              # Learning rate
--warmup_ratio 0.03               # Warmup proportion
--weight_decay 0.01               # Weight decay
--max_grad_norm 1.0               # Gradient clipping
```

### Calibration Settings

```bash
--calib_beta 100          # Bin size for calibration error
--calib_p_norm "2"        # Norm type: "1", "2", or "infty"
```

### System Settings

```bash
--max_seq_length 2048     # Maximum sequence length
--bf16                    # Use BF16 (default for H200)
--seed 42                 # Random seed
```

---

## Validation Process

During validation, the script:

1. **Generate prediction**: Model outputs answer + confidence token
   ```
   Input:  "Question: What is 2+2?"
   Output: "4 <C_READ>"
   ```

2. **Parse components**: Extract answer and confidence token
   ```python
   answer = "4"
   predicted_token = "<C_READ>"
   ```

3. **Check correctness**: Compare answer to ground truth
   ```python
   is_correct = (answer == ground_truth)  # True
   ```

4. **Determine expected token**: Based on correctness
   ```python
   expected_token = "<C_READ>" if is_correct else "<U_READ>"
   ```

5. **Validation accuracy**: Does predicted token match expected?
   ```python
   validation_correct = (predicted_token == expected_token)
   ```

6. **Calibration**: Extract probability distribution
   ```python
   p_confident = p(<C_READ>) / (p(<C_READ>) + p(<U_READ>))
   ```

### Two Metrics Tracked

1. **Validation Accuracy**: % of times model outputs correct confidence token
2. **Calibration Error**: How well probabilities match actual correctness

---

## Output Structure

```
checkpoints/
└── qwen_confidence_20241113_143022/
    ├── checkpoint-500/        # Periodic checkpoints
    ├── checkpoint-1000/
    ├── best_accuracy/         # Best validation accuracy model
    ├── best_calibration/      # Best calibration error model
    ├── final/                 # Final trained model
    ├── reliability_epoch_1.json  # Per-epoch metrics
    ├── reliability_epoch_2.json
    └── logs/                  # TensorBoard logs
```

### Checkpoint Contents

Each checkpoint contains:
- `adapter_config.json` - LoRA configuration
- `adapter_model.bin` - LoRA weights
- `tokenizer_config.json` - Tokenizer with special tokens
- `special_tokens_map.json` - Token mappings

---

## Monitoring Training

### TensorBoard

```bash
tensorboard --logdir checkpoints/qwen_confidence_20241113_143022/logs
```

Metrics tracked:
- `train/loss` - Training loss
- `train/learning_rate` - Learning rate schedule
- `train/grad_norm` - Gradient norm

### Console Output

Example validation output:
```
======================================================================
VALIDATION - Epoch 1.0
======================================================================

Validating on medqa...
100%|██████████| 500/500 [05:23<00:00,  1.55it/s]

MEDQA Results:
  Validation Accuracy: 68.40%
  Calibration Error: 0.1523
  Samples evaluated: 500

Validating on boolq...
100%|██████████| 500/500 [03:12<00:00,  2.60it/s]

BOOLQ Results:
  Validation Accuracy: 72.80%
  Calibration Error: 0.1247
  Samples evaluated: 500

======================================================================
OVERALL VALIDATION METRICS:
  Average Validation Accuracy: 70.60%
  Average Calibration Error: 0.1385
======================================================================

✓ New best validation accuracy! Saving to ./checkpoints/.../best_accuracy
✓ New best calibration error! Saving to ./checkpoints/.../best_calibration
```

---

## Testing

### Run All Tests

```bash
# Install test dependencies
pip install pytest pytest-cov

# Run all tests
pytest tests/test_training.py -v

# Run with coverage
pytest tests/test_training.py --cov=train_confidence_lora --cov-report=html
```

### Run Specific Test Classes

```bash
# Test calibration error
pytest tests/test_training.py::TestCalibration -v

# Test gradient masking (CRITICAL)
pytest tests/test_training.py::TestGradientMasking -v

# Test validation logic
pytest tests/test_training.py::TestValidationLogic -v
```

### Test Categories

- **TestSpecialTokens**: Token initialization and vocabulary
- **TestGradientMasking**: Gradient masking logic (CRITICAL)
- **TestCalibration**: Calibration error calculation
- **TestDataPipeline**: Data loading and collation
- **TestAnswerExtraction**: Answer parsing functions
- **TestValidationLogic**: Confidence token matching
- **TestProbabilityExtraction**: Softmax over domain tokens
- **TestIntegration**: End-to-end workflows
- **TestSanityChecks**: Basic sanity checks

---

## GPU Memory Usage

### Expected Memory (8-bit quantization)

**Model size:**
- Qwen 7B: ~8GB
- Mistral 7B: ~8GB

**Total training memory:**
- Batch size 8: ~25-30GB
- Batch size 16: ~40-45GB
- Batch size 32: ~70-80GB

### Memory Optimization Tips

1. **Reduce batch size:**
   ```bash
   --batch_size 4 --gradient_accumulation_steps 2
   ```

2. **Reduce sequence length:**
   ```bash
   --max_seq_length 1024
   ```

3. **Enable gradient checkpointing** (already enabled by default)

---

## Troubleshooting

### Out of Memory (OOM)

```bash
# Reduce batch size
--batch_size 4

# Reduce sequence length
--max_seq_length 1024

# Use gradient accumulation
--gradient_accumulation_steps 4
```

### Slow Training

```bash
# Verify Unsloth is working
python -c "from unsloth import FastLanguageModel; print('Unsloth OK')"

# Check GPU utilization
nvidia-smi -l 1
```

### NaN Loss

```bash
# Reduce learning rate
--learning_rate 1e-4

# Increase warmup
--warmup_ratio 0.1

# Check data for issues
python -c "
from train_confidence_lora import ConfidenceDataset
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-7B-Instruct')
ds = ConfidenceDataset('tagged/qwen/qwen_all_tagged.jsonl', tokenizer)
print(f'Loaded {len(ds)} examples')
print(ds[0])
"
```

### Validation Not Running

- Ensure HuggingFace datasets are accessible
- Check internet connection for dataset downloads
- Use `--eval_steps` to control validation frequency

---

## Advanced Usage

### Custom Validation Datasets

```python
# Modify in train_confidence_lora.py
val_datasets = {
    "custom_dataset": load_dataset("your/dataset", split="validation")
}
```

### Different Confidence Tokens

```python
# Modify CONFIDENCE_TOKENS in train_confidence_lora.py
CONFIDENCE_TOKENS = ["<C_DOMAIN1>", "<U_DOMAIN1>", "<C_DOMAIN2>", "<U_DOMAIN2>"]
```

### Custom LoRA Target Modules

```python
# In setup_lora()
target_modules=["q_proj", "v_proj"]  # Only attention query and value
```

---

## Expected Results

### Training Timeline (3 epochs, ~19K samples)

- **Epoch 1**: ~2-3 hours
- **Epoch 2**: ~2-3 hours
- **Epoch 3**: ~2-3 hours
- **Total**: ~6-9 hours

### Target Metrics

**After 3 epochs:**
- Validation Accuracy: 70-80%
- Calibration Error: 0.10-0.15

**Good signs:**
- ✓ Training loss decreasing
- ✓ Validation accuracy increasing
- ✓ Calibration error decreasing
- ✓ No NaN losses

**Warning signs:**
- ⚠ Loss not decreasing after 500 steps
- ⚠ Validation accuracy < 60%
- ⚠ Calibration error > 0.30

---

## Citation

If you use this code, please cite:

```bibtex
@software{confidence_tokens_2024,
  title={Domain-Specific Confidence Token Training},
  author={Your Name},
  year={2024},
  url={https://github.com/yourusername/routing-confidence}
}
```

---

## License

MIT License

---

## Contact

For questions or issues:
- Open an issue on GitHub
- Contact: your.email@example.com

