# Quick Reference: Running Inference One-by-One

## Command Template

```bash
python run_dataset_inference.py \
  --model_name <MODEL> \
  --dataset <DATASET> \
  --split train
```

## Datasets

- `math` - MATH dataset (6,750 examples, 512 max tokens)
- `medqa` - MedQA dataset (varies, 256 max tokens)  
- `boolq` - BoolQ dataset (~8,484 examples, 128 max tokens)

## Models

- `mistralai/Mistral-7B-Instruct-v0.3`
- `Qwen/Qwen2.5-7B-Instruct`

## Examples

### Run Single Dataset

```bash
# Just MATH with Mistral
python run_dataset_inference.py \
  --model_name mistralai/Mistral-7B-Instruct-v0.3 \
  --dataset math
```

### Run with Custom Settings

```bash
# BoolQ with Qwen, custom batch size
python run_dataset_inference.py \
  --model_name Qwen/Qwen2.5-7B-Instruct \
  --dataset boolq \
  --batch_size 100 \
  --tensor_parallel_size 1
```

### Run with Custom Output Path

```bash
# Save to specific location
python run_dataset_inference.py \
  --model_name meta-llama/Meta-Llama-3-8B-Instruct \
  --dataset medqa \
  --output my_predictions/llama_medqa_experiment1.json
```

## Sequential Execution (One After Another)

If you want to run multiple datasets sequentially in one command:

```bash
# Run all MATH, then all MedQA, then all BoolQ
python run_dataset_inference.py --model_name mistralai/Mistral-7B-Instruct-v0.3 --dataset math && \
python run_dataset_inference.py --model_name mistralai/Mistral-7B-Instruct-v0.3 --dataset medqa && \
python run_dataset_inference.py --model_name mistralai/Mistral-7B-Instruct-v0.3 --dataset boolq
```

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--model_name` | *required* | HuggingFace model path |
| `--dataset` | *required* | Dataset: `math`, `medqa`, or `boolq` |
| `--split` | `train` | Dataset split to use |
| `--output` | auto | Output JSON file path |
| `--tensor_parallel_size` | 1 | Number of GPUs for tensor parallelism |
| `--batch_size` | 50 | Batch size for progress saving |

## Output Files

Default output paths (auto-generated):

- `predictions/mistral7binstructv03_math_train.json`
- `predictions/mistral7binstructv03_medqa_train.json`
- `predictions/mistral7binstructv03_boolq_train.json`
- `predictions/qwen257binstruct_math_train.json`
- `predictions/qwen257binstruct_medqa_train.json`
- `predictions/qwen257binstruct_boolq_train.json`

## Resumability

All commands are **resumable**! If interrupted, just run the same command again and it will continue from where it stopped.

## Monitoring Progress

The script shows:
- Loading progress
- Current batch being processed
- Number of predictions saved
- Average tokens per response

Example output:
```
Processing batch 51-100/6750...
Saved 100/6750 predictions
```

## Tips

💡 **Start small**: Try one dataset first to verify everything works  
💡 **Check output**: Look at the generated JSON to verify quality  
💡 **Monitor resources**: Watch GPU memory usage during first run  
💡 **Use batch_size**: Adjust based on available memory  
💡 **Save frequently**: Default batch_size=50 saves every 50 examples  

## After Inference

Once you have prediction files, judge them:

```bash
cd ../eval
python run_judge_datasets.py \
  --predictions ../inference/predictions/mistral7binstructv03_math_train.json \
  --dataset math
```

## Help

```bash
python run_dataset_inference.py --help
```

