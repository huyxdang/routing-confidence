# Project Structure Analysis

## Overview
This project is a **routing confidence** system that evaluates and trains models to output domain-specific confidence tokens (e.g., `<C_MED>`, `<U_MED>`, `<C_READ>`, `<U_READ>`, `<C_MATH>`, `<U_MATH>`) based on prediction correctness.

## Folder Structure & Purpose

### 📁 `correct_incorrect/`
**Purpose**: Stores predictions separated by correctness, with dataset-specific tagging scripts.

**Structure**:
- `boolq/` - BoolQ (reading comprehension) dataset predictions
  - `qwen/` - Qwen model predictions
  - `tag_boolq.py` - Script to tag BoolQ predictions with `<C_READ>` (correct) or `<U_READ>` (incorrect)
- `medqa/` - MedQA (medical QA) dataset predictions
  - `qwen/` - Qwen model predictions
  - `tag_medqa.py` - Script to tag MedQA predictions with `<C_MED>` (correct) or `<U_MED>` (incorrect)

**Files**:
- `*_correct.json` - Correct predictions
- `*_incorrect.json` - Incorrect predictions
- `*_tagged.json` - Combined tagged predictions

**Issues**:
- Duplicate tagging logic in `tag_boolq.py` and `tag_medqa.py` (nearly identical)
- Scripts are dataset-specific but could be generalized

---

### 📁 `data/`
**Purpose**: Dataset preparation and storage.

**Structure**:
- `split/` - Scripts to split datasets into train/val/test
  - `boolq_split.py` - Splits BoolQ: uses validation as test, splits train 90/10
  - `MATH_split.py` - Splits MATH: keeps test, splits train 90/10
  - `MedQA_split.py` - Splits MedQA: uses original train/dev/test from Kaggle
  - `upload_HF.py` - Orchestrates splitting and uploading to HuggingFace
- `tagged/` - Processed tagged datasets
  - `mistral/` - Mistral model tagged predictions
  - `qwen/` - Qwen model tagged predictions

**Issues**:
- Inconsistent naming: `MATH_split.py` vs `MedQA_split.py` (case inconsistency)
- `upload_HF.py` imports from local files but could use proper package structure

---

### 📁 `data_pipeline/`
**Purpose**: Complete pipeline for processing predictions (evaluate → tag → clean → upload).

**Files**:
- `evaluation.py` - Separates predictions into correct/incorrect (uses `eval/eval_simple.py`)
- `tagging.py` - Adds confidence tokens (`<C_MED>`, `<U_MED>`, etc.)
- `cleaning.py` - Removes unnecessary fields, keeps only essential ones
- `medqa_pipeline.py` - Main orchestrator that runs all steps
- `README.md` - Documentation for the pipeline

**Issues**:
- Pipeline is named `medqa_pipeline.py` but supports all datasets (boolq, math, medqa)
- Good modular design but could benefit from better naming

---

### 📁 `eval/`
**Purpose**: Evaluation and judging of model predictions.

**Files**:
- `eval_simple.py` - Pattern-based evaluation (no LLM) for MATH, MedQA, BoolQ
  - Extracts answers using regex/pattern matching
  - Compares with ground truth
  - Fast, deterministic evaluation
- `run_judge_datasets.py` - LLM-as-judge evaluation using OpenAI API
  - Uses GPT-4o to judge correctness
  - Adds domain-specific confidence tokens (`<CN_MATH>`, `<UN_MED>`, etc.)
  - Note: Different token format than `pipeline/` (`<CN_*>` vs `<C_*>`)
- `test_eval_simple.py` - Comprehensive unit tests for `eval_simple.py`

**Issues**:
- **TOKEN INCONSISTENCY**: `run_judge_datasets.py` uses `<CN_*>/<UN_*>` while `pipeline/` uses `<C_*>/<U_*>`
- Two evaluation methods (pattern-based vs LLM) serve different purposes but naming doesn't make this clear

---

### 📁 `inference/`
**Purpose**: Run model inference on datasets.

**Files**:
- `run_dataset_inference.py` - Runs vLLM inference on MATH, MedQA, BoolQ
  - Batch processing with vLLM
  - Supports resuming from partial results
  - Creates prompts based on dataset type

**Issues**:
- Good structure, minimal issues

---

### 📁 Root Level Files

**Files**:
- `train_sft.py` - Supervised fine-tuning script
  - Trains model to output confidence tokens
  - Uses LoRA for efficient training
  - Includes calibration error calculation
  - Quick evaluation callback during training
- `README.md` - Main project documentation (describes calibration evaluation workflow)
- `requirements.txt` - Dependencies

**Issues**:
- `train_sft.py` references `merged_data/` which doesn't exist (deleted in git status)
- Main README describes a different workflow than what the codebase actually does

---

## Critical Issues & Improvements

### 🔴 High Priority

1. **Token Format Inconsistency**
   - `eval/run_judge_datasets.py`: Uses `<CN_MATH>`, `<UN_MED>`, `<CN_READ>`
   - `pipeline/tagging.py`: Uses `<C_MED>`, `<U_MED>`, `<C_READ>`, `<U_READ>`, `<C_MATH>`, `<U_MATH>`
   - **Fix**: Standardize on one format across the codebase

2. **Duplicate Code**
   - `data_pipeline/` and `pipeline/` contain nearly identical modules
   - **Fix**: Remove `data_pipeline/` or clearly document which to use

3. **Missing Data Path**
   - `train_sft.py` references `merged_data/full_train_data_sft.jsonl` which doesn't exist
   - **Fix**: Update path or create data preparation script

### 🟡 Medium Priority

4. **Naming Inconsistencies**
   - `MATH_split.py` vs `MedQA_split.py` (case)
   - `medqa_pipeline.py` supports all datasets (misleading name)
   - **Fix**: Use consistent naming conventions

5. **Folder Organization**
   - `correct_incorrect/` contains both data and scripts
   - `data/tagged/` vs `correct_incorrect/` - unclear distinction
   - **Fix**: Separate scripts from data, clarify data organization

6. **Documentation Gaps**
   - Main README describes different workflow than codebase
   - Missing documentation for token format differences
   - **Fix**: Update README to match actual codebase structure

### 🟢 Low Priority

7. **Code Duplication in Tagging Scripts**
   - `correct_incorrect/boolq/tag_boolq.py` and `correct_incorrect/medqa/tag_medqa.py` are nearly identical
   - **Fix**: Create shared utility function

8. **Import Path Issues**
   - Some scripts use `sys.path.insert()` instead of proper package structure
   - **Fix**: Convert to proper Python package with `__init__.py` files

---

## Suggested Folder Reorganization

```
routing-confidence/
├── data/
│   ├── raw/              # Raw datasets
│   ├── processed/        # Split datasets
│   └── predictions/      # Model predictions (organized by model/dataset)
├── scripts/
│   ├── data_preparation/ # Dataset splitting scripts
│   ├── inference/        # Inference scripts
│   ├── evaluation/       # Evaluation scripts
│   └── training/         # Training scripts
├── pipeline/             # Processing pipeline modules
├── eval/                 # Evaluation utilities
├── tests/                # Test files
└── config/               # Configuration files
```

---

## Recommended Improvements

### 1. Standardize Token Format
Create a configuration file defining token formats:
```python
# config/tokens.py
CONFIDENCE_TOKENS = {
    'medqa': {'correct': '<C_MED>', 'incorrect': '<U_MED>'},
    'boolq': {'correct': '<C_READ>', 'incorrect': '<U_READ>'},
    'math': {'correct': '<C_MATH>', 'incorrect': '<U_MATH>'},
}
```

### 2. Consolidate Duplicate Code
- Remove `data_pipeline/` folder
- Use only `pipeline/` modules
- Update all imports

### 3. Create Shared Utilities
- Extract common tagging logic into `pipeline/utils.py`
- Create shared dataset configuration in `config/datasets.py`

### 4. Improve Documentation
- Update main README to reflect actual workflow
- Add docstrings to all modules
- Document token format decisions
- Add examples for each pipeline step

### 5. Fix Import Structure
- Convert to proper Python package
- Remove `sys.path.insert()` hacks
- Use relative imports where appropriate

### 6. Add Configuration Management
- Create `config/` directory for:
  - Dataset configurations
  - Token definitions
  - Model paths
  - Training hyperparameters

---

## Workflow Summary

1. **Data Preparation**: `data/split/*.py` → Split datasets into train/val/test
2. **Inference**: `inference/run_dataset_inference.py` → Generate predictions
3. **Evaluation**: 
   - Fast: `eval/eval_simple.py` (pattern-based)
   - Accurate: `eval/run_judge_datasets.py` (LLM-as-judge)
4. **Processing**: `pipeline/medqa_pipeline.py` → Evaluate → Tag → Clean
5. **Training**: `train_sft.py` → Fine-tune model to output confidence tokens

