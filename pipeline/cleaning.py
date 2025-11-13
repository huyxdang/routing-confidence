"""
Cleaning module for removing unnecessary fields from predictions.
"""
import json
import os


def clean_medqa_record(record: dict) -> dict:
    """
    Clean a MedQA record to keep only specified fields.
    
    Keeps:
    - response
    - question
    - options
    - prompt
    - ground_truth (extracted from judge_response)
    - correct
    - tagged_response
    
    Removes everything else.
    """
    cleaned = {}
    
    # Keep these fields directly
    for field in ['response', 'question', 'options', 'prompt', 'correct', 'tagged_response']:
        if field in record:
            cleaned[field] = record[field]
    
    # Extract ground_truth from judge_response
    if 'judge_response' in record and isinstance(record['judge_response'], dict):
        if 'ground_truth' in record['judge_response']:
            cleaned['ground_truth'] = record['judge_response']['ground_truth']
    elif 'ground_truth' in record:
        # If ground_truth is already at top level, keep it
        cleaned['ground_truth'] = record['ground_truth']
    
    return cleaned


def clean_predictions(
    tagged_file: str,
    output_file: str = None,
    dataset_name: str = 'medqa'
) -> str:
    """
    Clean tagged predictions file to keep only specified fields.
    
    Args:
        tagged_file: Path to tagged predictions JSON file
        output_file: Path to output cleaned file (default: adds '_cleaned' suffix)
        dataset_name: Name of dataset ('medqa', 'boolq', 'math')
    
    Returns:
        Path to cleaned file
    """
    print(f"\n{'='*70}")
    print(f"CLEANING: {dataset_name.upper()} Tagged Predictions")
    print(f"{'='*70}")
    
    # Determine output file
    if output_file is None:
        base_name = os.path.splitext(tagged_file)[0]
        output_file = f"{base_name}_cleaned.json"
    
    # Load data
    print(f"\n1. Loading tagged predictions from: {tagged_file}")
    with open(tagged_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Convert to list if needed
    if isinstance(data, dict):
        if all(k.isdigit() for k in list(data.keys())[:100]):
            sorted_keys = sorted(data.keys(), key=int)
            records = [data[k] for k in sorted_keys]
        else:
            records = list(data.values())
    else:
        records = data
    
    print(f"   ✓ Loaded {len(records):,} records")
    
    # Show original fields
    if records:
        print(f"   Original fields: {list(records[0].keys())}")
    
    # Clean records
    print(f"\n2. Cleaning records...")
    cleaned_records = []
    
    if dataset_name == 'medqa':
        for record in records:
            cleaned = clean_medqa_record(record)
            cleaned_records.append(cleaned)
    else:
        # For other datasets, use same logic (can be extended later)
        for record in records:
            cleaned = clean_medqa_record(record)  # Reuse for now
            cleaned_records.append(cleaned)
    
    print(f"   ✓ Cleaned {len(cleaned_records):,} records")
    
    # Show cleaned fields
    if cleaned_records:
        print(f"   Cleaned fields: {list(cleaned_records[0].keys())}")
    
    # Convert back to dict format with string keys
    output_data = {str(i): record for i, record in enumerate(cleaned_records)}
    
    # Save cleaned data
    print(f"\n3. Saving cleaned data to: {output_file}")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    print(f"   ✓ Saved {len(cleaned_records):,} cleaned records")
    
    print(f"\n{'='*70}")
    print(f"CLEANING COMPLETE")
    print(f"{'='*70}")
    
    return output_file

