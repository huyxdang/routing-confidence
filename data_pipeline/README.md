# Prediction Processing Pipeline

Complete pipeline for processing predictions: **evaluate → tag → clean → upload**

## Quick Start

### MedQA Example

```bash
python pipeline/medqa_pipeline.py \
    --input inference/predictions/medqa/qwen_medqa_train.json \
    --output-dir correct_incorrect/medqa/qwen \
    --upload \
    --hf_repo huyxdang/qwen-medqa-tagged
```

## Pipeline Steps

1. **Evaluation**: Separates predictions into correct/incorrect
2. **Tagging**: Adds confidence tokens (`<C_MED>`, `<U_MED>`, etc.)
3. **Cleaning**: Removes unnecessary fields, keeps only essential ones
4. **Upload** (optional): Uploads to HuggingFace Hub

## Usage

### Full Pipeline

```bash
python pipeline/medqa_pipeline.py \
    --input <predictions_file> \
    --output-dir <output_directory> \
    --dataset medqa \
    --upload \
    --hf_repo <username>/<repo-name>
```

### Skip Steps

```bash
# Skip evaluation (use existing correct/incorrect files)
python pipeline/medqa_pipeline.py \
    --input <predictions_file> \
    --skip-evaluation \
    --output-dir <output_directory>

# Skip tagging (use existing tagged file)
python pipeline/medqa_pipeline.py \
    --input <predictions_file> \
    --skip-evaluation \
    --skip-tagging \
    --output-dir <output_directory>

# Skip cleaning (keep all fields)
python pipeline/medqa_pipeline.py \
    --input <predictions_file> \
    --output-dir <output_directory> \
    --skip-cleaning
```

## Module Structure

- **`evaluation.py`**: Evaluation and separation functions
- **`tagging.py`**: Confidence token tagging functions
- **`cleaning.py`**: Field cleaning functions
- **`medqa_pipeline.py`**: Main pipeline orchestrator

## Output Files

The pipeline creates:
- `{base_name}_correct.json` - Correct predictions
- `{base_name}_incorrect.json` - Incorrect predictions
- `{base_name}_tagged.json` - Combined tagged predictions
- `{base_name}_tagged_cleaned.json` - Final cleaned output

## Supported Datasets

- `medqa` - Medical QA (uses `<C_MED>`, `<U_MED>`)
- `boolq` - Reading comprehension (uses `<C_READ>`, `<U_READ>`)
- `math` - Mathematics (uses `<C_MATH>`, `<U_MATH>`)

