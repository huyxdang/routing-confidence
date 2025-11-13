#!/usr/bin/env python3
"""
Strip dataset to only fields needed for SFT training.

Only keeps:
- question
- tagged_response

Removes everything else.
"""

import json
import argparse
import os


def strip_to_sft_fields(input_file: str, output_file: str = None):
    """
    Strip dataset to only SFT-required fields.
    
    Args:
        input_file: Path to input JSONL file
        output_file: Path to output JSONL file (default: adds _sft suffix)
    """
    print("="*70)
    print("Stripping to SFT-Required Fields Only")
    print("="*70)
    print(f"Input: {input_file}")
    print(f"Output: {output_file or 'auto'}")
    print("\nKeeping only:")
    print("  - question")
    print("  - tagged_response")
    print("="*70)
    
    # Determine output file
    if output_file is None:
        base_name = os.path.splitext(input_file)[0]
        output_file = f"{base_name}_sft.jsonl"
    
    # Load and process
    print("\n1. Processing records...")
    stripped_count = 0
    skipped_count = 0
    
    with open(input_file, 'r', encoding='utf-8') as f_in, \
         open(output_file, 'w', encoding='utf-8') as f_out:
        
        for line_num, line in enumerate(f_in, 1):
            try:
                record = json.loads(line.strip())
                
                # Keep only required fields
                stripped = {
                    'question': record.get('question', ''),
                    'tagged_response': record.get('tagged_response', '')
                }
                
                # Skip if missing required fields
                if not stripped['question'] or not stripped['tagged_response']:
                    skipped_count += 1
                    continue
                
                # Write stripped record
                f_out.write(json.dumps(stripped, ensure_ascii=False) + '\n')
                stripped_count += 1
                
            except json.JSONDecodeError as e:
                print(f"   ⚠ Warning: Skipped line {line_num} (JSON decode error): {e}")
                skipped_count += 1
                continue
    
    print(f"   ✓ Processed {stripped_count + skipped_count:,} records")
    print(f"   ✓ Kept {stripped_count:,} records")
    if skipped_count > 0:
        print(f"   ⚠ Skipped {skipped_count:,} records (missing required fields)")
    
    # Show sample
    print("\n2. Sample stripped record:")
    print("-"*70)
    with open(output_file, 'r', encoding='utf-8') as f:
        first_line = f.readline()
        if first_line:
            sample = json.loads(first_line)
            print(f"Fields: {list(sample.keys())}")
            print(f"Question preview: {sample['question'][:100]}...")
            print(f"Tagged response preview: {sample['tagged_response'][:100]}...")
    print("-"*70)
    
    print("\n" + "="*70)
    print("COMPLETE!")
    print("="*70)
    print(f"✓ Stripped {stripped_count:,} records to SFT-required fields only")
    print(f"✓ Output saved to: {output_file}")
    print("\nFile size reduced - ready for training!")
    print("="*70)


def main():
    parser = argparse.ArgumentParser(
        description="Strip dataset to only fields needed for SFT (question, tagged_response)"
    )
    parser.add_argument(
        '--input',
        type=str,
        required=True,
        help='Input JSONL file'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output JSONL file (default: adds _sft suffix)'
    )
    
    args = parser.parse_args()
    
    strip_to_sft_fields(
        args.input,
        args.output
    )


if __name__ == "__main__":
    main()

