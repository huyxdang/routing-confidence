#!/usr/bin/env python3
"""
Convert JSON dict format to JSONL and clean judge_response.

1. Converts {"0": {...}, "1": {...}} to JSONL format
2. Removes judge_response field
3. Extracts extracted_answer to top level
4. Saves as .jsonl file
"""

import json
import argparse
import os


def load_json_dataset(file_path: str) -> dict:
    """Load JSON dataset and return as dict with string keys."""
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    if isinstance(data, dict):
        return data
    elif isinstance(data, list):
        return {str(i): item for i, item in enumerate(data)}
    else:
        raise ValueError(f"Unexpected data format: {type(data)}")


def clean_record(record: dict) -> dict:
    """
    Clean a record:
    - Remove judge_response
    - Extract extracted_answer to top level if it exists in judge_response
    """
    cleaned = record.copy()
    
    # Extract extracted_answer from judge_response if it exists
    if 'judge_response' in cleaned and isinstance(cleaned['judge_response'], dict):
        judge_resp = cleaned['judge_response']
        if 'extracted_answer' in judge_resp:
            cleaned['extracted_answer'] = judge_resp['extracted_answer']
        # Remove judge_response
        del cleaned['judge_response']
    
    return cleaned


def convert_to_jsonl(
    input_file: str,
    output_file: str = None
):
    """
    Convert JSON dict to JSONL format and clean records.
    
    Args:
        input_file: Path to input JSON file (dict format)
        output_file: Path to output JSONL file (default: replaces .json with .jsonl)
    """
    print("="*70)
    print("Convert to JSONL and Clean Records")
    print("="*70)
    print(f"Input: {input_file}")
    print(f"Output: {output_file or 'auto'}")
    print("="*70)
    
    # Determine output file
    if output_file is None:
        base_name = os.path.splitext(input_file)[0]
        output_file = f"{base_name}.jsonl"
    
    # Load data
    print("\n1. Loading JSON file...")
    data = load_json_dataset(input_file)
    print(f"   ✓ Loaded {len(data):,} records")
    
    # Show sample before cleaning
    if data:
        sample_before = list(data.values())[0]
        print(f"\n   Sample record (before cleaning):")
        print(f"     Fields: {list(sample_before.keys())}")
        if 'judge_response' in sample_before:
            print(f"     judge_response: {sample_before['judge_response']}")
    
    # Clean and convert to list
    print("\n2. Cleaning records (removing judge_response, extracting extracted_answer)...")
    cleaned_records = []
    
    for idx, record in data.items():
        cleaned = clean_record(record)
        cleaned_records.append(cleaned)
    
    print(f"   ✓ Cleaned {len(cleaned_records):,} records")
    
    # Show sample after cleaning
    if cleaned_records:
        sample_after = cleaned_records[0]
        print(f"\n   Sample record (after cleaning):")
        print(f"     Fields: {list(sample_after.keys())}")
        if 'extracted_answer' in sample_after:
            print(f"     extracted_answer: {sample_after['extracted_answer']}")
        if 'judge_response' in sample_after:
            print(f"     ⚠ Warning: judge_response still present!")
        else:
            print(f"     ✓ judge_response removed")
    
    # Save as JSONL
    print(f"\n3. Saving as JSONL to: {output_file}")
    os.makedirs(os.path.dirname(output_file) if os.path.dirname(output_file) else '.', exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for record in cleaned_records:
            f.write(json.dumps(record, ensure_ascii=False) + '\n')
    
    print(f"   ✓ Saved {len(cleaned_records):,} records as JSONL")
    
    print("\n" + "="*70)
    print("COMPLETE!")
    print("="*70)
    print(f"✓ Converted to JSONL format")
    print(f"✓ Removed judge_response field")
    print(f"✓ Extracted extracted_answer to top level")
    print(f"✓ Output saved to: {output_file}")
    print("\nYou can now use this file with:")
    print(f"  load_dataset('json', data_files='{output_file}', split='train')")
    print("="*70)


def main():
    parser = argparse.ArgumentParser(
        description="Convert JSON dict to JSONL and clean judge_response"
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
        help='Output JSONL file (default: replaces .json with .jsonl)'
    )
    
    args = parser.parse_args()
    
    convert_to_jsonl(
        args.input,
        args.output
    )


if __name__ == "__main__":
    main()

