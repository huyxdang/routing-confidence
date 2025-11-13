"""
Evaluation module for separating predictions into correct/incorrect.
"""
import json
import os
from tqdm import tqdm
from eval.eval_simple import (
    extract_medqa_answer,
    extract_boolq_answer,
    extract_math_answer,
    DATASET_CONFIGS
)


def load_json_dataset(file_path: str) -> dict:
    """Load JSON dataset and return as dict with string keys."""
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    if isinstance(data, dict):
        # Already in dict format
        return data
    elif isinstance(data, list):
        # Convert list to dict
        return {str(i): item for i, item in enumerate(data)}
    else:
        raise ValueError(f"Unexpected data format: {type(data)}")


def judge_prediction(prediction: dict, dataset_name: str) -> dict:
    """
    Judge a single prediction and add correctness information.
    
    Returns prediction with added 'judge_response' and 'correct' fields.
    """
    if dataset_name not in DATASET_CONFIGS:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    config = DATASET_CONFIGS[dataset_name]
    domain = config['domain']
    
    response = prediction.get('response', '')
    ground_truth = prediction.get('correct_answer', '')
    
    # Extract answer based on dataset
    if dataset_name == 'medqa':
        result = extract_medqa_answer(response, ground_truth)
    elif dataset_name == 'boolq':
        result = extract_boolq_answer(response, ground_truth)
    elif dataset_name == 'math':
        result = extract_math_answer(response, ground_truth)
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")
    
    # Add judge_response and correct fields
    judged_pred = prediction.copy()
    judged_pred['judge_response'] = {
        'extracted_answer': result['extracted_answer'],
        'ground_truth': result['ground_truth'],
        'reasoning': result.get('reasoning', ''),
        'correct': 'yes' if result['is_correct'] else 'no'
    }
    judged_pred['correct'] = result['is_correct']
    
    return judged_pred


def evaluate_and_separate(
    predictions_file: str,
    dataset_name: str,
    output_dir: str = None
) -> tuple:
    """
    Evaluate predictions and separate into correct/incorrect files.
    
    Args:
        predictions_file: Path to input predictions JSON file
        dataset_name: Name of dataset ('medqa', 'boolq', 'math')
        output_dir: Directory to save output files (default: same as input)
    
    Returns:
        (correct_file, incorrect_file, stats_dict)
    """
    print(f"\n{'='*70}")
    print(f"EVALUATION: {dataset_name.upper()}")
    print(f"{'='*70}")
    
    # Load predictions
    print(f"\n1. Loading predictions from: {predictions_file}")
    predictions = load_json_dataset(predictions_file)
    print(f"   ✓ Loaded {len(predictions):,} predictions")
    
    # Judge all predictions
    print(f"\n2. Evaluating predictions...")
    correct_predictions = {}
    incorrect_predictions = {}
    
    for idx, prediction in tqdm(predictions.items(), desc="Evaluating"):
        judged_pred = judge_prediction(prediction, dataset_name)
        
        if judged_pred['correct']:
            correct_predictions[idx] = judged_pred
        else:
            incorrect_predictions[idx] = judged_pred
    
    # Calculate statistics
    total = len(predictions)
    correct_count = len(correct_predictions)
    incorrect_count = len(incorrect_predictions)
    accuracy = (correct_count / total * 100) if total > 0 else 0
    
    stats = {
        'total': total,
        'correct': correct_count,
        'incorrect': incorrect_count,
        'accuracy': accuracy
    }
    
    print(f"\n   ✓ Correct: {correct_count:,} ({accuracy:.2f}%)")
    print(f"   ✓ Incorrect: {incorrect_count:,} ({100-accuracy:.2f}%)")
    
    # Determine output paths
    if output_dir is None:
        base_name = os.path.splitext(predictions_file)[0]
        output_dir = os.path.dirname(base_name) or '.'
        base_name = os.path.basename(base_name)
    else:
        os.makedirs(output_dir, exist_ok=True)
        base_name = os.path.splitext(os.path.basename(predictions_file))[0]
    
    correct_file = os.path.join(output_dir, f"{base_name}_correct.json")
    incorrect_file = os.path.join(output_dir, f"{base_name}_incorrect.json")
    
    # Save files
    print(f"\n3. Saving separated predictions...")
    with open(correct_file, 'w', encoding='utf-8') as f:
        json.dump(correct_predictions, f, indent=2, ensure_ascii=False)
    print(f"   ✓ {correct_file} ({len(correct_predictions):,} predictions)")
    
    with open(incorrect_file, 'w', encoding='utf-8') as f:
        json.dump(incorrect_predictions, f, indent=2, ensure_ascii=False)
    print(f"   ✓ {incorrect_file} ({len(incorrect_predictions):,} predictions)")
    
    print(f"\n{'='*70}")
    print(f"EVALUATION COMPLETE")
    print(f"{'='*70}")
    
    return correct_file, incorrect_file, stats

