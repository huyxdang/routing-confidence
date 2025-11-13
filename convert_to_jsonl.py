#!/usr/bin/env python3
"""
Convert JSON dict format to JSONL format for training.

Converts: {"0": {...}, "1": {...}} → JSONL (one object per line)
"""

import json
import argparse
import os


def clean_record(record: dict) -> dict:
    """
    Clean a record:
    - Normalize correct_answer to always be a string
    - Remove judge_response
    - Extract extracted_answer to top level if it exists in judge_response
    """
    cleaned = record.copy()
    
    # Normalize correct_answer to string
    if 'correct_answer' in cleaned:
        ca = cleaned['correct_answer']
        if isinstance(ca, bool):
            # Convert boolean to lowercase string
            cleaned['correct_answer'] = str(ca).lower()
        elif ca is None:
            cleaned['correct_answer'] = ""
        else:
            # Already a string or other type, convert to string
            cleaned['correct_answer'] = str(ca)
    
    # Extract extracted_answer from judge_response if it exists
    if 'judge_response' in cleaned and isinstance(cleaned['judge_response'], dict):
        judge_resp = cleaned['judge_response']
        if 'extracted_answer' in judge_resp:
            cleaned['extracted_answer'] = judge_resp['extracted_answer']
        # Remove judge_response
        del cleaned['judge_response']
    
    return cleaned


def convert_json_to_jsonl(input_file: str, output_file: str = None):
    """
    Convert JSON dict format to JSONL format and clean records.
    
    Args:
        input_file: Path to input JSON file (dict format)
        output_file: Path to output JSONL file (default: adds .jsonl extension)
    """
    print("="*70)
    print("Converting JSON to JSONL Format and Cleaning")
    print("="*70)
    print(f"Input: {input_file}")
    print(f"Output: {output_file or 'auto'}")
    print("="*70)
    
    # Load JSON file
    print("\n1. Loading JSON file...")
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Convert dict to list
    if isinstance(data, dict):
        if all(k.isdigit() for k in list(data.keys())[:100]):
            # Sort by numeric key
            sorted_keys = sorted(data.keys(), key=int)
            records = [data[k] for k in sorted_keys]
        else:
            records = list(data.values())
    elif isinstance(data, list):
        records = data
    else:
        raise ValueError(f"Unexpected data format: {type(data)}")
    
    print(f"   ✓ Loaded {len(records):,} records")
    
    # Clean records
    print("\n2. Cleaning records...")
    print("   - Normalizing correct_answer to string (True/False → 'true'/'false')")
    print("   - Removing judge_response")
    print("   - Extracting extracted_answer to top level")
    
    cleaned_records = []
    bool_count = 0
    string_count = 0
    
    for record in records:
        cleaned = clean_record(record)
        cleaned_records.append(cleaned)
        
        # Count original types for stats
        if 'correct_answer' in record:
            if isinstance(record['correct_answer'], bool):
                bool_count += 1
            else:
                string_count += 1
    
    print(f"   ✓ Cleaned {len(cleaned_records):,} records")
    print(f"     Converted {bool_count:,} boolean correct_answer values to strings")
    print(f"     Kept {string_count:,} string correct_answer values")
    
    # Determine output file
    if output_file is None:
        base_name = os.path.splitext(input_file)[0]
        output_file = f"{base_name}.jsonl"
    
    # Write JSONL format
    print(f"\n3. Writing JSONL format to: {output_file}")
    os.makedirs(os.path.dirname(output_file) if os.path.dirname(output_file) else '.', exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for record in cleaned_records:
            f.write(json.dumps(record, ensure_ascii=False) + '\n')
    
    print(f"   ✓ Wrote {len(cleaned_records):,} records to JSONL file")
    
    # Show sample
    print("\n3. Sample record (first line):")
    print("-"*70)
    if records:
        sample = json.dumps(records[0], ensure_ascii=False)
        if len(sample) > 200:
            print(sample[:200] + "...")
        else:
            print(sample)
    print("-"*70)
    
    print("\n" + "="*70)
    print("CONVERSION COMPLETE!")
    print("="*70)
    print(f"✓ Converted {len(records):,} records to JSONL format")
    print(f"✓ Output saved to: {output_file}")
    print("="*70)


def main():
    parser = argparse.ArgumentParser(
        description="Convert JSON dict format to JSONL format"
    )
    parser.add_argument(
        '--input',
        type=str,
        required=True,
        help='Input JSON file (dict format)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output JSONL file (default: adds .jsonl extension)'
    )
    
    args = parser.parse_args()
    
    convert_json_to_jsonl(
        args.input,
        args.output
    )


if __name__ == "__main__":
    main()

