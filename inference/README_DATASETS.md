# Dataset Inference and Evaluation

This directory contains scripts for running inference on MATH, MedQA, and BoolQ datasets, then evaluating with LLM-as-judge to add domain-specific confidence tokens.

## Overview

### Datasets

| Dataset | HF Path | Domain | Confidence Tokens |
|---------|---------|--------|-------------------|
| MATH | `huyxdang/math-split` | `MATH` | `<CN_MATH>` / `<UN_MATH>` |
| MedQA | `huyxdang/medqa-split` | `MED` | `<CN_MED>` / `<UN_MED>` |
| BoolQ | `huyxdang/boolq-split` | `READ` | `<CN_READ>` / `<UN_READ>` |

### Models

- `meta-llama/Meta-Llama-3-8B-Instruct`
- `Qwen/Qwen2.5-7B-Instruct`

## Step 1: Run Inference

Use `run_dataset_inference.py` to generate predictions without confidence prompting.

### Usage

```bash
python run_dataset_inference.py \
  --model_name meta-llama/Meta-Llama-3-8B-Instruct \
  --dataset math \
  --split train \
  --output predictions/llama3-8b_math_train.json \
  --tensor_parallel_size 1 \
  --batch_size 50
```

### Arguments

- `--model_name`: HuggingFace model name (required)
- `--dataset`: Dataset to use: `math`, `medqa`, or `boolq` (required)
- `--split`: Dataset split (default: `train`)
- `--output`: Output JSON file (default: auto-generated)
- `--tensor_parallel_size`: Number of GPUs for tensor parallelism (default: 1)
- `--batch_size`: Batch size for progress saving (default: 50)

### Dataset-Specific Settings

The script automatically configures:

- **MATH**: 512 max tokens, math problem prompts
- **MedQA**: 256 max tokens, multiple-choice prompts with options
- **BoolQ**: 128 max tokens, reading comprehension prompts with passage

### Examples

**Run all datasets for Llama3-8B:**

```bash
# MATH
python run_dataset_inference.py \
  --model_name meta-llama/Meta-Llama-3-8B-Instruct \
  --dataset math \
  --split train

# MedQA
python run_dataset_inference.py \
  --model_name meta-llama/Meta-Llama-3-8B-Instruct \
  --dataset medqa \
  --split train

# BoolQ
python run_dataset_inference.py \
  --model_name meta-llama/Meta-Llama-3-8B-Instruct \
  --dataset boolq \
  --split train
```

**Run all datasets for Qwen2.5-7B:**

```bash
# MATH
python run_dataset_inference.py \
  --model_name Qwen/Qwen2.5-7B-Instruct \
  --dataset math \
  --split train

# MedQA
python run_dataset_inference.py \
  --model_name Qwen/Qwen2.5-7B-Instruct \
  --dataset medqa \
  --split train

# BoolQ
python run_dataset_inference.py \
  --model_name Qwen/Qwen2.5-7B-Instruct \
  --dataset boolq \
  --split train
```

## Step 2: Judge and Tag Predictions

Use `../eval/run_judge_datasets.py` to evaluate predictions and append confidence tokens.

### Prerequisites

```bash
# Set OpenAI API key for judge model
export OPENAI_API_KEY="your-api-key"
```

### Usage

```bash
python ../eval/run_judge_datasets.py \
  --predictions predictions/llama3-8b_math_train.json \
  --dataset math \
  --split train \
  --output judged/llama3-8b_math_train_tagged.json \
  --judge gpt-4o-2024-08-06 \
  --num_workers 50
```

### Arguments

- `--predictions`: Path to predictions JSON file (required)
- `--dataset`: Dataset name: `math`, `medqa`, or `boolq` (required)
- `--split`: Dataset split (default: `train`)
- `--output`: Output file for tagged predictions (default: `{predictions}_tagged.json`)
- `--judge`: Judge model to use (default: `gpt-4o-2024-08-06`)
- `--num_workers`: Number of parallel workers (default: 50)

### Output Format

The judge script produces JSON files with the following structure:

```json
{
  "0": {
    "response": "The answer is 42.",
    "question": "What is 6 times 7?",
    "correct_answer": "42",
    "judge_response": {
      "correct_answer": "42",
      "model_answer": "42",
      "reasoning": "The extracted answer matches the correct answer.",
      "correct": "yes"
    },
    "correct": true,
    "confidence_token": "<CN_MATH>",
    "tagged_response": "The answer is 42. <CN_MATH>"
  }
}
```

### Examples

**Judge all Llama3-8B predictions:**

```bash
python ../eval/run_judge_datasets.py \
  --predictions predictions/llama3-8b_math_train.json \
  --dataset math

python ../eval/run_judge_datasets.py \
  --predictions predictions/llama3-8b_medqa_train.json \
  --dataset medqa

python ../eval/run_judge_datasets.py \
  --predictions predictions/llama3-8b_boolq_train.json \
  --dataset boolq
```

**Judge all Qwen2.5-7B predictions:**

```bash
python ../eval/run_judge_datasets.py \
  --predictions predictions/qwen25-7b_math_train.json \
  --dataset math

python ../eval/run_judge_datasets.py \
  --predictions predictions/qwen25-7b_medqa_train.json \
  --dataset medqa

python ../eval/run_judge_datasets.py \
  --predictions predictions/qwen25-7b_boolq_train.json \
  --dataset boolq
```

## Output Structure

After running both steps, you'll have:

```
predictions/
  llama3-8b_math_train.json           # Raw predictions
  llama3-8b_medqa_train.json
  llama3-8b_boolq_train.json
  qwen25-7b_math_train.json
  qwen25-7b_medqa_train.json
  qwen25-7b_boolq_train.json

judged/
  llama3-8b_math_train_tagged.json    # Tagged with confidence tokens
  llama3-8b_medqa_train_tagged.json
  llama3-8b_boolq_train_tagged.json
  qwen25-7b_math_train_tagged.json
  qwen25-7b_medqa_train_tagged.json
  qwen25-7b_boolq_train_tagged.json
```

## Resumability

Both scripts support resuming from partial progress:

- **Inference**: Automatically skips already processed examples
- **Judge**: Skips already judged predictions

You can safely interrupt and restart the scripts.

## Notes

- Inference uses vLLM for fast batch processing
- Judge uses async OpenAI API for parallel evaluation
- All scripts save progress incrementally
- Confidence tokens are appended to the original response
- Domain-specific tokens help identify the source dataset

