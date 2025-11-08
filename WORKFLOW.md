# Complete Workflow: Dataset Inference → Judging → Tagged Training Data

## Overview

This workflow generates training data with domain-specific confidence tokens for routing models.

```
┌─────────────┐      ┌──────────────┐      ┌─────────────────┐
│  Datasets   │  →   │  Inference   │  →   │   LLM Judge     │
│ (train split)│      │  (no conf)   │      │ + Token Tagging │
└─────────────┘      └──────────────┘      └─────────────────┘
     MATH                 vLLM                  GPT-4o Judge
     MedQA           (Llama3/Qwen)           + Confidence Tokens
     BoolQ             Batch Process          <CN_X> / <UN_X>
```

## Step-by-Step Workflow

### 1️⃣ Prepare Environment

```bash
# Navigate to project
cd /Users/huydang/Desktop/routing-confidence

# Install dependencies (if not already done)
pip install -r requirements.txt

# Set OpenAI API key for judging
export OPENAI_API_KEY="your-openai-api-key"
```

### 2️⃣ Test Configuration (Optional but Recommended)

```bash
cd inference
python test_inference.py
```

**Expected Output:**
- ✅ Successfully loads all 3 datasets
- ✅ Shows sample prompts for each dataset
- ✅ Displays dataset configurations

### 3️⃣ Run Inference

**Option A: Automated (All 6 combinations)**

```bash
cd inference
./run_all_inference.sh
```

This runs:
- Llama3-8B-Instruct on MATH, MedQA, BoolQ
- Qwen2.5-7B-Instruct on MATH, MedQA, BoolQ

**Option B: Manual (Single dataset)**

```bash
cd inference

# Example: Llama3-8B on MATH
python run_dataset_inference.py \
  --model_name meta-llama/Meta-Llama-3-8B-Instruct \
  --dataset math \
  --split train \
  --batch_size 50
```

**Output:** `predictions/metallama38binstruct_math_train.json`

### 4️⃣ Judge and Tag Predictions

**Option A: Automated (All 6 files)**

```bash
cd eval
./run_all_judge.sh
```

**Option B: Manual (Single file)**

```bash
cd eval

# Example: Judge Llama3-8B MATH predictions
python run_judge_datasets.py \
  --predictions ../inference/predictions/metallama38binstruct_math_train.json \
  --dataset math \
  --num_workers 50
```

**Output:** `judged/metallama38binstruct_math_train_tagged.json`

### 5️⃣ Verify Results

```bash
cd eval/judged

# Check output files
ls -lh

# View sample from a tagged file
head -n 50 metallama38binstruct_math_train_tagged.json
```

## Expected Timeline

| Task | Time Estimate | Notes |
|------|---------------|-------|
| Test configuration | 2-5 min | Downloads datasets |
| Inference per dataset | 10-60 min | Depends on dataset size & GPU |
| All 6 inference runs | 1-6 hours | Can run in parallel on multiple GPUs |
| Judging per dataset | 5-30 min | Depends on API rate limits |
| All 6 judging runs | 30-180 min | Async parallel processing |

## Output Files Summary

After completing the workflow:

```
inference/predictions/
├── metallama38binstruct_math_train.json      (~6,750 examples)
├── metallama38binstruct_medqa_train.json     (varies)
├── metallama38binstruct_boolq_train.json     (~8,484 examples)
├── qwen257binstruct_math_train.json          (~6,750 examples)
├── qwen257binstruct_medqa_train.json         (varies)
└── qwen257binstruct_boolq_train.json         (~8,484 examples)

eval/judged/
├── metallama38binstruct_math_train_tagged.json
├── metallama38binstruct_medqa_train_tagged.json
├── metallama38binstruct_boolq_train_tagged.json
├── qwen257binstruct_math_train_tagged.json
├── qwen257binstruct_medqa_train_tagged.json
└── qwen257binstruct_boolq_train_tagged.json
```

## Sample Output Structure

### Before Judging (Inference Output)
```json
{
  "0": {
    "response": "To solve this, we need to...\n\nThe answer is 42.",
    "question": "What is 6 times 7?",
    "correct_answer": "42"
  }
}
```

### After Judging (Tagged Output)
```json
{
  "0": {
    "response": "To solve this, we need to...\n\nThe answer is 42.",
    "question": "What is 6 times 7?",
    "correct_answer": "42",
    "judge_response": {
      "correct_answer": "42",
      "model_answer": "42",
      "reasoning": "The model correctly solved the multiplication problem.",
      "correct": "yes"
    },
    "correct": true,
    "confidence_token": "<CN_MATH>",
    "tagged_response": "To solve this, we need to...\n\nThe answer is 42. <CN_MATH>"
  },
  "1": {
    "response": "I think the answer is 48.",
    "question": "What is 6 times 7?",
    "correct_answer": "42",
    "judge_response": {
      "correct_answer": "42",
      "model_answer": "48",
      "reasoning": "The model provided an incorrect answer.",
      "correct": "no"
    },
    "correct": false,
    "confidence_token": "<UN_MATH>",
    "tagged_response": "I think the answer is 48. <UN_MATH>"
  }
}
```

## Confidence Token Reference

| Dataset | Domain | Correct | Incorrect |
|---------|--------|---------|-----------|
| MATH | MATH | `<CN_MATH>` | `<UN_MATH>` |
| MedQA | MED | `<CN_MED>` | `<UN_MED>` |
| BoolQ | READ | `<CN_READ>` | `<UN_READ>` |

## Using the Tagged Data

The `tagged_response` field contains the model output with the appropriate confidence token appended. Use this for:

### 1. Training Data for Router Models
```python
# Combine all tagged responses
training_data = []
for dataset in ['math', 'medqa', 'boolq']:
    for model in ['llama3', 'qwen25']:
        with open(f'judged/{model}_{dataset}_train_tagged.json') as f:
            data = json.load(f)
            for item in data.values():
                training_data.append({
                    'input': item['question'],
                    'output': item['tagged_response'],
                    'domain': extract_domain(item['confidence_token']),
                    'correct': item['correct']
                })
```

### 2. Confidence Calibration Analysis
```python
# Analyze accuracy by domain
from collections import defaultdict

stats = defaultdict(lambda: {'correct': 0, 'total': 0})
for item in all_tagged_data:
    domain = extract_domain(item['confidence_token'])
    stats[domain]['total'] += 1
    if item['correct']:
        stats[domain]['correct'] += 1

for domain, counts in stats.items():
    accuracy = counts['correct'] / counts['total'] * 100
    print(f"{domain}: {accuracy:.2f}% accuracy")
```

### 3. Multi-Domain Fine-Tuning
Use the tagged responses to fine-tune models that learn to:
- Predict domain-specific confidence tokens
- Route queries to appropriate specialized models
- Calibrate confidence based on domain

## Troubleshooting

### Issue: CUDA Out of Memory
**Solution:**
```bash
# Reduce batch size or use tensor parallelism
python run_dataset_inference.py \
  --tensor_parallel_size 2 \
  --batch_size 25 \
  ...
```

### Issue: OpenAI Rate Limit
**Solution:**
```bash
# Reduce parallel workers
python run_judge_datasets.py \
  --num_workers 20 \
  ...
```

### Issue: Process Interrupted
**Solution:**
Both scripts are resumable! Just re-run the same command and it will continue from where it left off.

### Issue: Missing API Key
**Solution:**
```bash
export OPENAI_API_KEY="your-key-here"
# Or add to ~/.bashrc or ~/.zshrc
```

## Documentation Files

- 📘 `INFERENCE_GUIDE.md` - Complete usage guide
- 📗 `IMPLEMENTATION_SUMMARY.md` - Technical implementation details
- 📕 `inference/README_DATASETS.md` - Detailed inference documentation
- 📙 `WORKFLOW.md` - This file (step-by-step workflow)

## Quick Commands Cheat Sheet

```bash
# Test
cd inference && python test_inference.py

# Run all inference
cd inference && ./run_all_inference.sh

# Run all judging
cd eval && ./run_all_judge.sh

# Check progress
ls -lh inference/predictions/
ls -lh eval/judged/

# View sample output
head -100 eval/judged/*.json | less
```

## Success Criteria

✅ All 6 prediction files generated in `inference/predictions/`  
✅ All 6 tagged files generated in `eval/judged/`  
✅ Each tagged file has `tagged_response` and `confidence_token` fields  
✅ Accuracy metrics displayed for each dataset  
✅ Confidence tokens correctly assigned based on correctness  

## Next Steps After Completion

1. **Combine datasets** for multi-domain training
2. **Analyze accuracy** by domain and model
3. **Fine-tune router models** using tagged responses
4. **Evaluate calibration** of confidence predictions
5. **Create validation sets** using val/test splits

Enjoy your tagged training data! 🚀

