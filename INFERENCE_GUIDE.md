# Dataset Inference and Confidence Token Tagging Guide

Complete guide for running inference on MATH, MedQA, and BoolQ datasets, then tagging responses with domain-specific confidence tokens based on correctness.

## Overview

This pipeline generates training data for routing models by:
1. Running inference on datasets without confidence prompting
2. Evaluating answers with LLM-as-judge
3. Appending domain-specific confidence tokens based on correctness

## Datasets and Domains

| Dataset | Source | Domain | Correct Token | Incorrect Token | Train Size |
|---------|--------|--------|---------------|-----------------|------------|
| **MATH** | `huyxdang/math-split` | `MATH` | `<CN_MATH>` | `<UN_MATH>` | 6,750 |
| **MedQA** | `huyxdang/medqa-split` | `MED` | `<CN_MED>` | `<UN_MED>` | varies |
| **BoolQ** | `huyxdang/boolq-split` | `READ` | `<CN_READ>` | `<UN_READ>` | ~8,484 |

## Models

- **Llama3-8B-Instruct**: `meta-llama/Meta-Llama-3-8B-Instruct`
- **Qwen2.5-7B-Instruct**: `Qwen/Qwen2.5-7B-Instruct`

## Quick Start

### Prerequisites

```bash
# Install dependencies
pip install -r requirements.txt

# Set OpenAI API key (for judging step)
export OPENAI_API_KEY="your-api-key-here"
```

### Option 1: Run Everything Automatically

```bash
# Step 1: Run all inference (6 model x dataset combinations)
cd inference
./run_all_inference.sh

# Step 2: Judge all predictions and add confidence tokens
cd ../eval
./run_all_judge.sh
```

### Option 2: Run Manually

**Step 1: Run Inference**

```bash
cd inference

# Llama3-8B on MATH
python run_dataset_inference.py \
  --model_name meta-llama/Meta-Llama-3-8B-Instruct \
  --dataset math \
  --split train

# Qwen2.5-7B on MedQA
python run_dataset_inference.py \
  --model_name Qwen/Qwen2.5-7B-Instruct \
  --dataset medqa \
  --split train

# ... (see inference/README_DATASETS.md for all combinations)
```

**Step 2: Judge and Tag**

```bash
cd eval

# Judge Llama3-8B MATH predictions
python run_judge_datasets.py \
  --predictions ../inference/predictions/metallama38binstruct_math_train.json \
  --dataset math

# Judge Qwen2.5-7B MedQA predictions
python run_judge_datasets.py \
  --predictions ../inference/predictions/qwen257binstruct_medqa_train.json \
  --dataset medqa

# ... (see eval script for all combinations)
```

## Output Files

### After Inference

```
inference/predictions/
├── metallama38binstruct_math_train.json
├── metallama38binstruct_medqa_train.json
├── metallama38binstruct_boolq_train.json
├── qwen257binstruct_math_train.json
├── qwen257binstruct_medqa_train.json
└── qwen257binstruct_boolq_train.json
```

### After Judging

```
eval/judged/
├── metallama38binstruct_math_train_tagged.json
├── metallama38binstruct_medqa_train_tagged.json
├── metallama38binstruct_boolq_train_tagged.json
├── qwen257binstruct_math_train_tagged.json
├── qwen257binstruct_medqa_train_tagged.json
└── qwen257binstruct_boolq_train_tagged.json
```

## Output Format

### Prediction Files (after inference)

```json
{
  "0": {
    "response": "The answer is 42.",
    "question": "What is 6 times 7?",
    "correct_answer": "42"
  }
}
```

### Tagged Files (after judging)

```json
{
  "0": {
    "response": "The answer is 42.",
    "question": "What is 6 times 7?",
    "correct_answer": "42",
    "judge_response": {
      "correct_answer": "42",
      "model_answer": "42",
      "reasoning": "The extracted answer matches the correct answer exactly.",
      "correct": "yes"
    },
    "correct": true,
    "confidence_token": "<CN_MATH>",
    "tagged_response": "The answer is 42. <CN_MATH>"
  }
}
```

## Configuration Details

### Dataset-Specific Settings

| Dataset | Max Tokens | Prompt Type | Special Handling |
|---------|-----------|-------------|------------------|
| MATH | 512 | Math problem | Full solution expected |
| MedQA | 256 | Multiple choice | Options included |
| BoolQ | 128 | Yes/No | Passage included |

### Judge Model

Default: `gpt-4o-2024-08-06`

The judge model evaluates whether the model's response matches the ground truth answer and determines the appropriate confidence token.

## Features

✅ **Resumable**: Both inference and judging support resuming from partial progress  
✅ **Incremental Saving**: Progress saved after each batch  
✅ **Parallel Judging**: Async OpenAI API calls with configurable workers  
✅ **Dataset-Specific Prompts**: Optimized for each dataset type  
✅ **Token Limits**: Appropriate max_tokens per dataset  

## Command Reference

### Inference Script

```bash
python run_dataset_inference.py \
  --model_name <model-path> \
  --dataset <math|medqa|boolq> \
  --split train \
  --output predictions/output.json \
  --tensor_parallel_size 1 \
  --batch_size 50
```

### Judge Script

```bash
python run_judge_datasets.py \
  --predictions <prediction-file> \
  --dataset <math|medqa|boolq> \
  --split train \
  --output judged/output_tagged.json \
  --judge gpt-4o-2024-08-06 \
  --num_workers 50
```

## Troubleshooting

### CUDA Out of Memory

Reduce batch size or use tensor parallelism:
```bash
--tensor_parallel_size 2  # Use 2 GPUs
--batch_size 25           # Smaller batches
```

### OpenAI Rate Limits

Reduce parallel workers:
```bash
--num_workers 20  # Instead of default 50
```

### Missing Predictions

The judge script will warn if prediction files are not found. Make sure to run inference first.

## Next Steps

After generating tagged predictions:
1. Combine datasets for multi-domain training
2. Use tagged responses for fine-tuning routing models
3. Evaluate model performance on confidence calibration

## Documentation

- `inference/README_DATASETS.md`: Detailed inference documentation
- `inference/run_all_inference.sh`: Automated inference script
- `eval/run_all_judge.sh`: Automated judging script

## Notes

- Confidence tokens are appended to the **end** of each response
- Domain identifiers (MATH, MED, READ) help track dataset source
- All processing uses deterministic settings (temperature=0)
- Intermediate results are saved for crash recovery

