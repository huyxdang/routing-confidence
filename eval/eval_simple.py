"""
Evaluate model predictions using simple pattern matching (no LLM-as-judge needed).
Much faster and free compared to OpenAI API approach.
"""
import os
import json
import argparse
import re
from tqdm import tqdm



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
    """Extract and compare MedQA answer (A/B/C/D)."""
    response_upper = response.upper().strip()
    
    # Look for A, B, C, or D at the beginning (first 20 chars)
    beginning = response_upper[:20]
    
    # Pattern to find option letter
    option_pattern = r'\b([ABCD])\b'
    matches = re.findall(option_pattern, beginning)
    
    if matches:
        extracted = matches[0]
    else:
        # Try to find it in first sentence
        first_sentence = response_upper.split('.')[0] if '.' in response_upper else response_upper[:100]
        matches = re.findall(option_pattern, first_sentence)
        if matches:
            extracted = matches[0]
        else:
            extracted = 'unknown'
    
    # Ground truth is the answer string
    expected = ground_truth.strip().upper()
    
    # Sometimes ground truth might be full text, extract letter if needed
    if len(expected) > 1:
        # Try to extract letter from ground truth
        gt_matches = re.findall(option_pattern, expected[:20])
        if gt_matches:
            expected = gt_matches[0]
    
    is_correct = extracted == expected
    
    return {
        "extracted_answer": extracted,
        "ground_truth": expected,
        "is_correct": is_correct,
        "reasoning": f"Extracted: '{extracted}' vs Expected: '{expected}'"
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
    
    # Assign confidence token
    if is_correct:
        confidence_token = f"<CN_{domain}>"
    else:
        confidence_token = f"<UN_{domain}>"
    
    # Create tagged response
    tagged_response = f"{response} {confidence_token}"
    
    # Update prediction
    prediction["judge_response"] = {
        "extracted_answer": result["extracted_answer"],
        "ground_truth": result["ground_truth"],
        "reasoning": result["reasoning"],
        "correct": "yes" if is_correct else "no"
    }
    prediction["correct"] = is_correct
    prediction["tagged_response"] = tagged_response
    prediction["confidence_token"] = confidence_token
    
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
    """Main function to judge predictions."""
    
    # Validate dataset
    if args.dataset not in DATASET_CONFIGS:
        raise ValueError(f"Unknown dataset: {args.dataset}. Must be one of: {list(DATASET_CONFIGS.keys())}")
    
    config = DATASET_CONFIGS[args.dataset]
    domain = config['domain']
    
    print(f"\n{'='*60}")
    print(f"Judging predictions for {args.dataset} dataset")
    print(f"Method: Pattern-based extraction (no LLM)")
    print(f"Domain: {domain}")
    print(f"Confidence tokens: <CN_{domain}> (correct) / <UN_{domain}> (incorrect)")
    print(f"{'='*60}\n")
    
    # Load predictions
    print(f"Loading predictions from: {args.predictions}")
    with open(args.predictions, 'r') as f:
        predictions = json.load(f)
    
    print(f"Loaded {len(predictions)} predictions")
    
    # Set output file
    if args.output is None:
        base_name = os.path.splitext(args.predictions)[0]
        args.output = f"{base_name}_tagged.json"
    
    # Create output directory if needed
    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # Load existing judged predictions if resuming
    if os.path.exists(args.output):
        with open(args.output, 'r') as f:
            judged_predictions = json.load(f)
        print(f"Resuming from existing file with {len(judged_predictions)} judged predictions")
        # Merge with loaded predictions
        for idx, judged_pred in judged_predictions.items():
            if idx in predictions:
                predictions[idx] = judged_pred
    
    # Count how many need judging
    to_judge = sum(1 for p in predictions.values() if "judge_response" not in p)
    print(f"\nPredictions to judge: {to_judge}/{len(predictions)}")
    
    if to_judge == 0:
        print("✓ All predictions already judged!")
    else:
        print(f"\n{'='*60}")
        print(f"Starting judging process (pattern matching)")
        print(f"{'='*60}\n")
        
        # Judge all predictions with progress bar
        judged_count = 0
        for idx, prediction in tqdm(predictions.items(), desc="Judging"):
            if "judge_response" not in prediction:
                predictions[idx] = judge_prediction(prediction, args.dataset, domain)
                judged_count += 1
                
                # Save periodically (every 1000)
                if judged_count % 1000 == 0:
                    with open(args.output, 'w') as f:
                        json.dump(predictions, f, indent=2)
        
        print(f"\n{'='*60}")
        print(f"Judging complete!")
        print(f"{'='*60}")
        print(f"  Judgments processed: {judged_count}")
        print(f"{'='*60}\n")
        
        # Save final results
        print(f"Saving results to: {args.output}")
        with open(args.output, 'w') as f:
            json.dump(predictions, f, indent=2)
        
        print(f"✓ Saved successfully!")
    
    # Calculate and display accuracy
    accuracy, correct, total = calculate_accuracy(predictions)
    
    print(f"\n{'='*60}")
    print(f"RESULTS")
    print(f"{'='*60}")
    print(f"Dataset: {args.dataset} ({domain})")
    print(f"Total predictions: {total}")
    print(f"Correct: {correct}")
    print(f"Incorrect: {total - correct}")
    print(f"Accuracy: {accuracy:.2f}%")
    print(f"{'='*60}\n")
    
    # Show sample tagged responses
    print("Sample tagged responses:")
    sample_count = 0
    for idx, pred in predictions.items():
        if "tagged_response" in pred and sample_count < 3:
            print(f"\n[Example {sample_count + 1}]")
            print(f"Correct: {pred['correct']}")
            print(f"Token: {pred['confidence_token']}")
            print(f"Extracted: {pred['judge_response']['extracted_answer']}")
            print(f"Tagged: {pred['tagged_response'][:200]}...")
            sample_count += 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Judge model predictions using pattern matching (no LLM needed)"
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
        "--output",
        type=str,
        default=None,
        help="Output file for tagged predictions (default: {predictions}_tagged.json)"
    )
    
    args = parser.parse_args()
    main(args)

