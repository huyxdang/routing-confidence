"""
Tagging module for adding confidence tokens to predictions.
"""
import json
import os


def tag_predictions(predictions: dict, confidence_token: str) -> dict:
    """
    Add tagged_response field with confidence token to predictions.
    
    Args:
        predictions: Dict of predictions (with string keys)
        confidence_token: Token to append (e.g., '<C_MED>', '<U_MED>')
    
    Returns:
        Dict of predictions with 'tagged_response' field added
    """
    tagged = {}
    for idx, pred in predictions.items():
        tagged_pred = pred.copy()
        tagged_pred["tagged_response"] = f"{pred['response']} {confidence_token}"
        tagged[idx] = tagged_pred
    return tagged


def tag_medqa_predictions(
    correct_file: str,
    incorrect_file: str,
    output_dir: str = None
) -> str:
    """
    Tag MedQA predictions with confidence tokens.
    
    Args:
        correct_file: Path to correct predictions JSON file
        incorrect_file: Path to incorrect predictions JSON file
        output_dir: Directory to save output (default: same as input files)
    
    Returns:
        Path to combined tagged file
    """
    print(f"\n{'='*70}")
    print(f"TAGGING: MedQA Predictions")
    print(f"{'='*70}")
    
    # Determine output directory
    if output_dir is None:
        output_dir = os.path.dirname(correct_file) or '.'
    
    # Determine base name from correct file
    base_name = os.path.splitext(os.path.basename(correct_file))[0]
    # Remove '_correct' suffix if present
    if base_name.endswith('_correct'):
        base_name = base_name[:-8]
    
    # Load predictions
    print(f"\n1. Loading predictions...")
    with open(correct_file, 'r', encoding='utf-8') as f:
        correct_predictions = json.load(f)
    print(f"   ✓ Loaded {len(correct_predictions):,} correct predictions")
    
    with open(incorrect_file, 'r', encoding='utf-8') as f:
        incorrect_predictions = json.load(f)
    print(f"   ✓ Loaded {len(incorrect_predictions):,} incorrect predictions")
    
    # Tag predictions
    print(f"\n2. Tagging predictions...")
    correct_tagged = tag_predictions(correct_predictions, "<C_MED>")
    incorrect_tagged = tag_predictions(incorrect_predictions, "<U_MED>")
    print(f"   ✓ Tagged correct with <C_MED>")
    print(f"   ✓ Tagged incorrect with <U_MED>")
    
    # Combine
    print(f"\n3. Combining tagged predictions...")
    combined = {}
    combined.update(correct_tagged)
    combined.update(incorrect_tagged)
    print(f"   ✓ Combined {len(combined):,} predictions")
    
    # Save combined file
    combined_file = os.path.join(output_dir, f"{base_name}_tagged.json")
    with open(combined_file, 'w', encoding='utf-8') as f:
        json.dump(combined, f, indent=2, ensure_ascii=False)
    print(f"   ✓ Saved to: {combined_file}")
    
    print(f"\n{'='*70}")
    print(f"TAGGING COMPLETE")
    print(f"{'='*70}")
    
    return combined_file


def tag_boolq_predictions(
    correct_file: str,
    incorrect_file: str,
    output_dir: str = None
) -> str:
    """
    Tag BoolQ predictions with confidence tokens.
    
    Args:
        correct_file: Path to correct predictions JSON file
        incorrect_file: Path to incorrect predictions JSON file
        output_dir: Directory to save output (default: same as input files)
    
    Returns:
        Path to combined tagged file
    """
    print(f"\n{'='*70}")
    print(f"TAGGING: BoolQ Predictions")
    print(f"{'='*70}")
    
    # Determine output directory
    if output_dir is None:
        output_dir = os.path.dirname(correct_file) or '.'
    
    # Determine base name from correct file
    base_name = os.path.splitext(os.path.basename(correct_file))[0]
    if base_name.endswith('_correct'):
        base_name = base_name[:-8]
    
    # Load predictions
    print(f"\n1. Loading predictions...")
    with open(correct_file, 'r', encoding='utf-8') as f:
        correct_predictions = json.load(f)
    print(f"   ✓ Loaded {len(correct_predictions):,} correct predictions")
    
    with open(incorrect_file, 'r', encoding='utf-8') as f:
        incorrect_predictions = json.load(f)
    print(f"   ✓ Loaded {len(incorrect_predictions):,} incorrect predictions")
    
    # Tag predictions
    print(f"\n2. Tagging predictions...")
    correct_tagged = tag_predictions(correct_predictions, "<C_READ>")
    incorrect_tagged = tag_predictions(incorrect_predictions, "<U_READ>")
    print(f"   ✓ Tagged correct with <C_READ>")
    print(f"   ✓ Tagged incorrect with <U_READ>")
    
    # Combine
    print(f"\n3. Combining tagged predictions...")
    combined = {}
    combined.update(correct_tagged)
    combined.update(incorrect_tagged)
    print(f"   ✓ Combined {len(combined):,} predictions")
    
    # Save combined file
    combined_file = os.path.join(output_dir, f"{base_name}_tagged.json")
    with open(combined_file, 'w', encoding='utf-8') as f:
        json.dump(combined, f, indent=2, ensure_ascii=False)
    print(f"   ✓ Saved to: {combined_file}")
    
    print(f"\n{'='*70}")
    print(f"TAGGING COMPLETE")
    print(f"{'='*70}")
    
    return combined_file


def tag_math_predictions(
    correct_file: str,
    incorrect_file: str,
    output_dir: str = None
) -> str:
    """
    Tag MATH predictions with confidence tokens.
    
    Args:
        correct_file: Path to correct predictions JSON file
        incorrect_file: Path to incorrect predictions JSON file
        output_dir: Directory to save output (default: same as input files)
    
    Returns:
        Path to combined tagged file
    """
    print(f"\n{'='*70}")
    print(f"TAGGING: MATH Predictions")
    print(f"{'='*70}")
    
    # Determine output directory
    if output_dir is None:
        output_dir = os.path.dirname(correct_file) or '.'
    
    # Determine base name from correct file
    base_name = os.path.splitext(os.path.basename(correct_file))[0]
    if base_name.endswith('_correct'):
        base_name = base_name[:-8]
    
    # Load predictions
    print(f"\n1. Loading predictions...")
    with open(correct_file, 'r', encoding='utf-8') as f:
        correct_predictions = json.load(f)
    print(f"   ✓ Loaded {len(correct_predictions):,} correct predictions")
    
    with open(incorrect_file, 'r', encoding='utf-8') as f:
        incorrect_predictions = json.load(f)
    print(f"   ✓ Loaded {len(incorrect_predictions):,} incorrect predictions")
    
    # Tag predictions
    print(f"\n2. Tagging predictions...")
    correct_tagged = tag_predictions(correct_predictions, "<C_MATH>")
    incorrect_tagged = tag_predictions(incorrect_predictions, "<U_MATH>")
    print(f"   ✓ Tagged correct with <C_MATH>")
    print(f"   ✓ Tagged incorrect with <U_MATH>")
    
    # Combine
    print(f"\n3. Combining tagged predictions...")
    combined = {}
    combined.update(correct_tagged)
    combined.update(incorrect_tagged)
    print(f"   ✓ Combined {len(combined):,} predictions")
    
    # Save combined file
    combined_file = os.path.join(output_dir, f"{base_name}_tagged.json")
    with open(combined_file, 'w', encoding='utf-8') as f:
        json.dump(combined, f, indent=2, ensure_ascii=False)
    print(f"   ✓ Saved to: {combined_file}")
    
    print(f"\n{'='*70}")
    print(f"TAGGING COMPLETE")
    print(f"{'='*70}")
    
    return combined_file

