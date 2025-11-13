#!/usr/bin/env python3
"""
Complete pipeline for processing MedQA predictions:
1. Evaluation (separate correct/incorrect)
2. Tagging (add confidence tokens)
3. Cleaning (remove unnecessary fields)
4. Optional: Upload to HuggingFace

Usage:
    python pipeline/medqa_pipeline.py \
        --input inference/predictions/medqa/qwen_medqa_train.json \
        --output correct_incorrect/medqa/qwen/qwen_medqa_train_tagged_cleaned.json \
        --upload \
        --hf_repo huyxdang/qwen-medqa-tagged
"""

import argparse
import os
import sys

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline.evaluation import evaluate_and_separate
from pipeline.tagging import tag_medqa_predictions
from pipeline.cleaning import clean_predictions
from upload_to_hf import upload_dataset


def main():
    parser = argparse.ArgumentParser(
        description="Complete pipeline for MedQA predictions: evaluate → tag → clean → upload"
    )
    parser.add_argument(
        '--input',
        type=str,
        required=True,
        help='Input predictions JSON file'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output cleaned tagged file (default: auto-generated)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='Directory for intermediate and output files (default: same as input)'
    )
    parser.add_argument(
        '--dataset',
        type=str,
        default='medqa',
        choices=['medqa', 'boolq', 'math'],
        help='Dataset name (default: medqa)'
    )
    parser.add_argument(
        '--upload',
        action='store_true',
        help='Upload to HuggingFace after processing'
    )
    parser.add_argument(
        '--hf-repo',
        type=str,
        default=None,
        help='HuggingFace repo name (required if --upload)'
    )
    parser.add_argument(
        '--skip-evaluation',
        action='store_true',
        help='Skip evaluation step (use existing correct/incorrect files)'
    )
    parser.add_argument(
        '--skip-tagging',
        action='store_true',
        help='Skip tagging step (use existing tagged file)'
    )
    parser.add_argument(
        '--skip-cleaning',
        action='store_true',
        help='Skip cleaning step'
    )
    
    args = parser.parse_args()
    
    print("="*70)
    print("MEDQA PREDICTION PIPELINE")
    print("="*70)
    print(f"Input: {args.input}")
    print(f"Dataset: {args.dataset}")
    print(f"Output directory: {args.output_dir or 'auto'}")
    print("="*70)
    
    # Determine output directory
    if args.output_dir is None:
        args.output_dir = os.path.dirname(args.input) or '.'
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Step 1: Evaluation
    if not args.skip_evaluation:
        print("\n" + "="*70)
        print("STEP 1: EVALUATION")
        print("="*70)
        correct_file, incorrect_file, stats = evaluate_and_separate(
            args.input,
            args.dataset,
            args.output_dir
        )
    else:
        print("\n" + "="*70)
        print("STEP 1: EVALUATION (SKIPPED)")
        print("="*70)
        # Try to find existing files
        base_name = os.path.splitext(os.path.basename(args.input))[0]
        correct_file = os.path.join(args.output_dir, f"{base_name}_correct.json")
        incorrect_file = os.path.join(args.output_dir, f"{base_name}_incorrect.json")
        
        if not os.path.exists(correct_file) or not os.path.exists(incorrect_file):
            print(f"✗ Error: Could not find existing files:")
            print(f"  Expected: {correct_file}")
            print(f"  Expected: {incorrect_file}")
            sys.exit(1)
        
        print(f"✓ Using existing files:")
        print(f"  {correct_file}")
        print(f"  {incorrect_file}")
    
    # Step 2: Tagging
    if not args.skip_tagging:
        print("\n" + "="*70)
        print("STEP 2: TAGGING")
        print("="*70)
        
        if args.dataset == 'medqa':
            tagged_file = tag_medqa_predictions(
                correct_file,
                incorrect_file,
                args.output_dir
            )
        elif args.dataset == 'boolq':
            from pipeline.tagging import tag_boolq_predictions
            tagged_file = tag_boolq_predictions(
                correct_file,
                incorrect_file,
                args.output_dir
            )
        elif args.dataset == 'math':
            from pipeline.tagging import tag_math_predictions
            tagged_file = tag_math_predictions(
                correct_file,
                incorrect_file,
                args.output_dir
            )
    else:
        print("\n" + "="*70)
        print("STEP 2: TAGGING (SKIPPED)")
        print("="*70)
        # Try to find existing tagged file
        base_name = os.path.splitext(os.path.basename(args.input))[0]
        tagged_file = os.path.join(args.output_dir, f"{base_name}_tagged.json")
        
        if not os.path.exists(tagged_file):
            print(f"✗ Error: Could not find existing tagged file:")
            print(f"  Expected: {tagged_file}")
            sys.exit(1)
        
        print(f"✓ Using existing tagged file: {tagged_file}")
    
    # Step 3: Cleaning
    if not args.skip_cleaning:
        print("\n" + "="*70)
        print("STEP 3: CLEANING")
        print("="*70)
        
        cleaned_file = clean_predictions(
            tagged_file,
            args.output,
            args.dataset
        )
    else:
        print("\n" + "="*70)
        print("STEP 3: CLEANING (SKIPPED)")
        print("="*70)
        cleaned_file = tagged_file
        print(f"✓ Using tagged file as final output: {cleaned_file}")
    
    # Step 4: Upload (optional)
    if args.upload:
        print("\n" + "="*70)
        print("STEP 4: UPLOAD TO HUGGINGFACE")
        print("="*70)
        
        if not args.hf_repo:
            print("✗ Error: --hf-repo required when using --upload")
            sys.exit(1)
        
        # Temporarily update the dataset path in upload_to_hf
        import upload_to_hf
        original_path = upload_to_hf.DATASETS['medqa_tagged']['path']
        upload_to_hf.DATASETS['medqa_tagged']['path'] = cleaned_file
        
        try:
            upload_dataset('medqa_tagged', args.hf_repo)
        finally:
            # Restore original path
            upload_to_hf.DATASETS['medqa_tagged']['path'] = original_path
    else:
        print("\n" + "="*70)
        print("STEP 4: UPLOAD (SKIPPED)")
        print("="*70)
        print("Use --upload to upload to HuggingFace")
    
    # Final summary
    print("\n" + "="*70)
    print("PIPELINE COMPLETE!")
    print("="*70)
    print(f"✓ Final output: {cleaned_file}")
    print("\nFiles created:")
    if not args.skip_evaluation:
        print(f"  - {correct_file}")
        print(f"  - {incorrect_file}")
    if not args.skip_tagging:
        print(f"  - {tagged_file}")
    if not args.skip_cleaning:
        print(f"  - {cleaned_file}")
    print("="*70)


if __name__ == "__main__":
    main()

