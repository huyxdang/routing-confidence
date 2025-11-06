# Routing Confidence: Calibration Evaluation

Evaluate calibration quality of Qwen 2.5 models (1.5B, 3B, 7B, 32B) on SimpleQA-verified dataset.

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Set OpenAI API key (required for judge evaluation)
export OPENAI_API_KEY='your-key-here'

# 1. Prepare dataset (300 questions from SimpleQA-verified)
python data/prepare_dataset.py

# 2. Run inference (fast with vLLM)
python inference/run_qwen_inference.py \
    --model_name Qwen/Qwen2.5-7B-Instruct \
    --questions data/simpleqa_300.json \
    --output results/qwen2.5-7b_predictions.json

# 3. Run judge evaluation
python eval/run_judge_results.py \
    --dataset data/simpleqa_300.json \
    --predictions results/qwen2.5-7b_predictions.json \
    --num_workers 50
```

## Features

- ✅ **Fast inference** with vLLM batch processing
- ✅ **Confidence in [0,1]** - Models prompted to output confidence as decimals
- ✅ **RMS calibration** - Uses HLE's calibration metrics
- ✅ **Overconfidence analysis** - Tracks calibration trends across model sizes

## Models

| Model | Parameters | Command |
|-------|-----------|---------|
| Qwen2.5-1.5B | 1.5B | `--model_name Qwen/Qwen2.5-1.5B-Instruct` |
| Qwen2.5-3B | 3B | `--model_name Qwen/Qwen2.5-3B-Instruct` |
| Qwen2.5-7B | 7B | `--model_name Qwen/Qwen2.5-7B-Instruct` |
| Qwen2.5-32B | 32B | `--model_name Qwen/Qwen2.5-32B-Instruct --tensor_parallel_size 2` |

## Confidence Handling

Models are prompted to output confidence in **[0, 1] range**:

```
Answer: Paris
Confidence: 0.95
```

The judge extracts confidence and uses it for RMS calibration error calculations.

## Project Structure

```
routing-confidence-1/
├── data/
│   └── prepare_dataset.py       # Download & prepare SimpleQA-300
├── inference/
│   └── run_qwen_inference.py    # Fast vLLM inference with [0,1] confidence
├── eval/
│   └── run_judge_results.py     # Judge evaluation + RMS calibration
└── requirements.txt             # Dependencies (vllm, openai, etc.)
```

## Output Format

Predictions are saved as:

```json
{
  "simpleqa_0": {
    "response": "Answer: Paris\nConfidence: 0.95"
  }
}
```

Judge adds correctness and metrics:

```json
{
  "accuracy": 45.67,
  "rms_calibration_error": 0.1234,
  "mean_confidence": 0.7850,
  "overconfidence": 0.3283
}
```
