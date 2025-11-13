"""
Evaluate the model and separate by correctness using simple pattern matching.
"""
import os
import json
import argparse
import re
from tqdm import tqdm
from datasets import load_dataset


# Dataset configurations
DATASET_CONFIGS = {
    'math': {
        'domain': 'MATH',
        'answer_field': 'solution'
    },
    'medqa': {
        'domain': 'MED',
        'answer_field': 'answer'
    },
    'boolq': {
        'domain': 'READ',
        'answer_field': 'answer'
    }
}


def extract_boxed_answer(text):
    """Extract answer from LaTeX \\boxed{...} notation."""
    # Find all \boxed{...} patterns
    patterns = [
        r'\\boxed\{([^}]+)\}',  # \boxed{answer}
        r'\\boxed\s*\{([^}]+)\}',  # \boxed {answer}
    ]
    
    for pattern in patterns:
        matches = re.findall(pattern, text)
        if matches:
            # Return the last boxed answer (usually the final answer)
            return matches[-1].strip()
    
    return None


def normalize_answer(answer):
    """Normalize answer for comparison."""
    if answer is None:
        return ""
    
    answer = str(answer).strip().lower()
    
    # Remove common punctuation
    answer = answer.replace(',', '').replace('.', '').replace('$', '')
    
    # Remove extra whitespace
    answer = ' '.join(answer.split())
    
    return answer


def extract_math_answer(response, ground_truth):
    """Extract and compare MATH answer."""
    # Extract from response
    response_answer = extract_boxed_answer(response)
    
    # Extract from ground truth (it's usually in boxed format too)
    ground_truth_answer = extract_boxed_answer(ground_truth)
    
    # If ground truth doesn't have boxed, use the whole thing
    if ground_truth_answer is None:
        ground_truth_answer = ground_truth
    
    # Normalize both
    response_norm = normalize_answer(response_answer)
    ground_truth_norm = normalize_answer(ground_truth_answer)
    
    # Check if they match
    is_correct = response_norm == ground_truth_norm
    
    return {
        "extracted_answer": response_answer if response_answer else "None",
        "ground_truth": ground_truth_answer if ground_truth_answer else ground_truth[:100],
        "is_correct": is_correct,
        "reasoning": f"Extracted: '{response_norm}' vs Ground truth: '{ground_truth_norm}'"
    }


def extract_boolq_answer(response, ground_truth):
    """Extract and compare BoolQ answer (Yes/No vs True/False)."""
    response_lower = response.lower().strip()
    
    # Look for yes/no at the beginning (first 50 chars)
    beginning = response_lower[:50]
    
    # Extract yes/no
    if 'yes' in beginning:
        extracted = 'yes'
    elif 'no' in beginning:
        extracted = 'no'
    else:
        # Try to find it anywhere in first sentence
        first_sentence = response_lower.split('.')[0] if '.' in response_lower else response_lower[:100]
        if 'yes' in first_sentence:
            extracted = 'yes'
        elif 'no' in first_sentence:
            extracted = 'no'
        else:
            extracted = 'unknown'
    
    # Ground truth is True/False
    ground_truth_bool = ground_truth
    expected = 'yes' if ground_truth_bool else 'no'
    
    is_correct = extracted == expected
    
    return {
        "extracted_answer": extracted,
        "ground_truth": expected,
        "is_correct": is_correct,
        "reasoning": f"Extracted: '{extracted}' vs Expected: '{expected}'"
    }


def extract_medqa_answer(response, ground_truth):
    """Extract and compare MedQA answer (A/B/C/D/E)."""
    response_stripped = response.strip()
    
    # Look for pattern "LETTER: TEXT" at the beginning
    letter_colon_pattern = r'^([ABCDE]):\s*'
    match = re.match(letter_colon_pattern, response_stripped, re.IGNORECASE)
    
    if match:
        # Found "C: ..." format
        extracted_letter = match.group(1).upper()
    else:
        # Try to find just the letter at the beginning
        response_upper = response_stripped.upper()
        beginning = response_upper[:20]
        
        option_pattern = r'\b([ABCDE])\b'
        matches = re.findall(option_pattern, beginning)
        
        if matches:
            extracted_letter = matches[0]
        else:
            # Try to find it in first sentence
            first_sentence = response_upper.split('.')[0] if '.' in response_upper else response_upper[:100]
            matches = re.findall(option_pattern, first_sentence)
            if matches:
                extracted_letter = matches[0]
            else:
                extracted_letter = 'unknown'
    
    # Ground truth should be a single letter (A-E)
    expected_letter = ground_truth.strip().upper()
    
    # Simple letter comparison
    is_correct = (extracted_letter == expected_letter)
    
    return {
        "extracted_answer": extracted_letter,
        "ground_truth": expected_letter,
        "is_correct": is_correct,
        "reasoning": f"Extracted: '{extracted_letter}' vs Ground truth: '{expected_letter}'"
    }


def judge_prediction(prediction, dataset_name, domain):
    """Judge a single prediction using pattern matching."""
    response = prediction["response"]
    correct_answer = prediction["correct_answer"]
    
    # Extract and compare based on dataset type
    if dataset_name == 'math':
        result = extract_math_answer(response, correct_answer)
    elif dataset_name == 'boolq':
        result = extract_boolq_answer(response, correct_answer)
    elif dataset_name == 'medqa':
        result = extract_medqa_answer(response, correct_answer)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    is_correct = result["is_correct"]
    
    # Update prediction with evaluation info (NO LABELING)
    prediction["judge_response"] = {
        "extracted_answer": result["extracted_answer"],
        "ground_truth": result["ground_truth"],
        "reasoning": result["reasoning"],
        "correct": "yes" if is_correct else "no"
    }
    prediction["correct"] = is_correct
    
    return prediction


def calculate_accuracy(predictions):
    """Calculate accuracy from judged predictions."""
    total = 0
    correct = 0
    
    for pred in predictions.values():
        if "correct" in pred:
            total += 1
            if pred["correct"]:
                correct += 1
    
    if total > 0:
        accuracy = correct / total * 100
        return accuracy, correct, total
    return 0, 0, 0


def main(args):
    """Main function to judge predictions and separate by correctness."""
    
    # Validate dataset
    if args.dataset not in DATASET_CONFIGS:
        raise ValueError(f"Unknown dataset: {args.dataset}. Must be one of: {list(DATASET_CONFIGS.keys())}")
    
    config = DATASET_CONFIGS[args.dataset]
    domain = config['domain']
    
    print(f"\n{'='*60}")
    print(f"Evaluating predictions for {args.dataset} dataset")
    print(f"Method: Pattern-based extraction (no LLM)")
    print(f"Domain: {domain}")
    print(f"Output: Separate correct and incorrect predictions")
    print(f"{'='*60}\n")
    
    # Load predictions
    print(f"Loading predictions from: {args.predictions}")
    with open(args.predictions, 'r') as f:
        predictions = json.load(f)
    
    print(f"Loaded {len(predictions)} predictions")
    
    # For MedQA, load the dataset to get answer_idx (letter answers)
    answer_idx_map = {}
    if args.dataset == 'medqa':
        print(f"Loading MedQA dataset to get answer_idx (letter answers)...")
        try:
            dataset = load_dataset('huyxdang/medqa-split', split='train')
            for i, example in enumerate(dataset):
                if 'answer_idx' in example:
                    answer_idx_map[str(i)] = example['answer_idx']
            print(f"✓ Loaded answer_idx for {len(answer_idx_map)} examples")
            
            # Replace correct_answer with answer_idx for all predictions
            for idx in predictions:
                if idx in answer_idx_map:
                    predictions[idx]['correct_answer'] = answer_idx_map[idx]
        except Exception as e:
            print(f"⚠ Warning: Could not load answer_idx from dataset: {e}")
            print(f"  Will use existing correct_answer field")
    
    # Set output files
    if args.output_prefix is None:
        base_name = os.path.splitext(args.predictions)[0]
        args.output_prefix = base_name
    
    correct_output = f"{args.output_prefix}_correct.json"
    incorrect_output = f"{args.output_prefix}_incorrect.json"
    
    # Create output directory if needed
    output_dir = os.path.dirname(args.output_prefix)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"Starting evaluation (pattern matching)")
    print(f"{'='*60}\n")
    
    # Judge all predictions with progress bar
    correct_predictions = {}
    incorrect_predictions = {}
    
    for idx, prediction in tqdm(predictions.items(), desc="Evaluating"):
        judged_pred = judge_prediction(prediction, args.dataset, domain)
        
        if judged_pred["correct"]:
            correct_predictions[idx] = judged_pred
        else:
            incorrect_predictions[idx] = judged_pred
    
    # Calculate statistics
    total = len(predictions)
    correct_count = len(correct_predictions)
    incorrect_count = len(incorrect_predictions)
    accuracy = (correct_count / total * 100) if total > 0 else 0
    
    print(f"\n{'='*60}")
    print(f"EVALUATION COMPLETE")
    print(f"{'='*60}")
    print(f"Dataset: {args.dataset} ({domain})")
    print(f"Total predictions: {total}")
    print(f"Correct: {correct_count} ({accuracy:.2f}%)")
    print(f"Incorrect: {incorrect_count} ({100-accuracy:.2f}%)")
    print(f"{'='*60}\n")
    
    # Save separated predictions
    print(f"Saving results...")
    print(f"  Correct predictions → {correct_output}")
    with open(correct_output, 'w') as f:
        json.dump(correct_predictions, f, indent=2)
    
    print(f"  Incorrect predictions → {incorrect_output}")
    with open(incorrect_output, 'w') as f:
        json.dump(incorrect_predictions, f, indent=2)
    
    print(f"\n✓ Successfully saved {correct_count} correct and {incorrect_count} incorrect predictions!")
    
    # Show sample predictions
    print(f"\n{'='*60}")
    print("SAMPLE CORRECT PREDICTIONS:")
    print(f"{'='*60}")
    for i, (idx, pred) in enumerate(list(correct_predictions.items())[:2]):
        print(f"\n[Correct Example {i+1}]")
        print(f"  Response: {pred['response'][:150]}...")
        print(f"  Extracted: {pred['judge_response']['extracted_answer']}")
        print(f"  Ground Truth: {pred['judge_response']['ground_truth']}")
    
    print(f"\n{'='*60}")
    print("SAMPLE INCORRECT PREDICTIONS:")
    print(f"{'='*60}")
    for i, (idx, pred) in enumerate(list(incorrect_predictions.items())[:2]):
        print(f"\n[Incorrect Example {i+1}]")
        print(f"  Response: {pred['response'][:150]}...")
        print(f"  Extracted: {pred['judge_response']['extracted_answer']}")
        print(f"  Ground Truth: {pred['judge_response']['ground_truth']}")
    
    print(f"\n{'='*60}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate predictions and separate by correctness (no LLM needed)"
    )
    parser.add_argument(
        "--predictions",
        type=str,
        required=True,
        help="Path to predictions JSON file"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=['math', 'medqa', 'boolq'],
        help="Dataset name"
    )
    parser.add_argument(
        "--output-prefix",
        type=str,
        default=None,
        help="Output file prefix (creates {prefix}_correct.json and {prefix}_incorrect.json)"
    )
    
    args = parser.parse_args()
    main(args)

