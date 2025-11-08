"""
Pipeline to create train/val/test splits for datasets and upload to Hugging Face.

This script:
1. Loads datasets from Hugging Face
2. Creates appropriate train/validation/test splits
3. Verifies split integrity
4. Uploads processed datasets to Hugging Face

Datasets handled:
- BoolQ: Use validation as test, split train into train (90%) + validation (10%)
- MATH: Keep test, split train into train (90%) + validation (10%)
- MedQA: Filter English only, handle existing or create new splits
"""

from datasets import DatasetDict
import argparse
from typing import Optional

# Import split functions from separate files
from boolq_split import split_boolq
from MATH_split import split_math
from MedQA_split import split_medqa


# Hugging Face username
HF_USERNAME = "huyxdang"


def verify_splits(dataset_dict: DatasetDict, dataset_name: str, original_train_size: Optional[int] = None) -> None:
    """
    Verify that dataset splits are correct.
    
    Args:
        dataset_dict: DatasetDict containing train/validation/test splits
        dataset_name: Name of the dataset for logging
        original_train_size: Original size of train split (for verification)
    """
    print(f"\n{'='*60}")
    print(f"Verification for {dataset_name}")
    print(f"{'='*60}")
    
    # Check all three splits exist
    required_splits = ['train', 'validation', 'test']
    for split in required_splits:
        if split not in dataset_dict:
            raise ValueError(f"Missing required split: {split}")
        print(f"✓ {split} split exists: {len(dataset_dict[split])} examples")
    
    # Verify fields are consistent across splits
    train_features = set(dataset_dict['train'].features.keys())
    val_features = set(dataset_dict['validation'].features.keys())
    test_features = set(dataset_dict['test'].features.keys())
    
    if train_features != val_features or train_features != test_features:
        raise ValueError("Features mismatch between splits!")
    print(f"✓ All splits have consistent features: {list(train_features)}")
    
    # Verify split sizes
    total_examples = len(dataset_dict['train']) + len(dataset_dict['validation']) + len(dataset_dict['test'])
    print(f"\nTotal examples: {total_examples}")
    print(f"  Train: {len(dataset_dict['train'])} ({len(dataset_dict['train'])/total_examples*100:.1f}%)")
    print(f"  Validation: {len(dataset_dict['validation'])} ({len(dataset_dict['validation'])/total_examples*100:.1f}%)")
    print(f"  Test: {len(dataset_dict['test'])} ({len(dataset_dict['test'])/total_examples*100:.1f}%)")
    
    # Verify split ratios if original train size is provided
    if original_train_size:
        new_train_size = len(dataset_dict['train'])
        new_val_size = len(dataset_dict['validation'])
        combined = new_train_size + new_val_size
        
        expected_train_size = int(original_train_size * 0.9)
        expected_val_size = original_train_size - expected_train_size
        
        print(f"\nSplit verification (from original train of {original_train_size}):")
        print(f"  Expected train: ~{expected_train_size} (90%)")
        print(f"  Actual train: {new_train_size}")
        print(f"  Expected validation: ~{expected_val_size} (10%)")
        print(f"  Actual validation: {new_val_size}")
        
        # Allow small variance due to rounding
        if abs(new_train_size - expected_train_size) > 2 or abs(new_val_size - expected_val_size) > 2:
            print("⚠ Warning: Split sizes differ from expected (may be due to rounding)")
    
    print(f"✓ Verification complete for {dataset_name}")
    print(f"{'='*60}\n")


# Split functions are imported from separate files:
# - boolq_split.py contains split_boolq()
# - MATH_split.py contains split_math()
# - MedQA_split.py contains split_medqa()


def upload_dataset(dataset_dict: DatasetDict, repo_name: str, private: bool = False) -> None:
    """
    Upload dataset to Hugging Face Hub.
    
    Args:
        dataset_dict: DatasetDict to upload
        repo_name: Name of the repository (without username)
        private: Whether to make the dataset private
    """
    repo_id = f"{HF_USERNAME}/{repo_name}"
    
    print(f"\n{'='*60}")
    print(f"Uploading to Hugging Face: {repo_id}")
    print(f"{'='*60}")
    
    try:
        # Push to hub
        dataset_dict.push_to_hub(
            repo_id=repo_id,
            private=private
        )
        print(f"✓ Successfully uploaded to https://huggingface.co/datasets/{repo_id}")
    except Exception as e:
        print(f"✗ Error uploading dataset: {e}")
        print(f"  Make sure you're logged in with: huggingface-cli login")
        raise


def main(args):
    """Main pipeline to process and upload all datasets."""
    
    print("\n" + "#"*60)
    print("# Dataset Split and Upload Pipeline")
    print("#"*60)
    print(f"Target HF username: {HF_USERNAME}")
    print("#"*60)
    
    datasets_to_process = []
    
    if not args.skip_boolq:
        datasets_to_process.append(('boolq', split_boolq, 'boolq-split'))
    
    if not args.skip_math:
        datasets_to_process.append(('math', split_math, 'math-split'))
    
    if not args.skip_medqa:
        datasets_to_process.append(('medqa', split_medqa, 'medqa-split'))
    
    # Process each dataset
    results = {}
    for name, split_func, repo_name in datasets_to_process:
        try:
            print(f"\n\n{'#'*60}")
            print(f"# Processing {name.upper()}")
            print(f"{'#'*60}")
            
            # Create splits
            dataset_dict = split_func()
            
            # Verify splits
            verify_splits(dataset_dict, name.upper())
            
            results[name] = dataset_dict
            
            # Upload to HuggingFace if not in dry-run mode
            if not args.dry_run:
                upload_dataset(dataset_dict, repo_name, private=args.private)
            else:
                print(f"\n[DRY RUN] Skipping upload for {repo_name}")
            
            print(f"\n✓ Successfully processed {name.upper()}")
            
        except Exception as e:
            print(f"\n✗ Error processing {name.upper()}: {e}")
            if args.stop_on_error:
                raise
            else:
                print("Continuing with next dataset...")
    
    # Print summary
    print("\n\n" + "#"*60)
    print("# SUMMARY")
    print("#"*60)
    print(f"\nProcessed {len(results)}/{len(datasets_to_process)} datasets successfully:")
    for name in results:
        print(f"  ✓ {name.upper()}")
    
    if args.dry_run:
        print("\n[DRY RUN MODE] No datasets were uploaded.")
    else:
        print(f"\nDatasets uploaded to: https://huggingface.co/{HF_USERNAME}")
    
    print("\n" + "#"*60 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Create train/val/test splits and upload datasets to Hugging Face"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Process datasets but don't upload to Hugging Face"
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Make uploaded datasets private"
    )
    parser.add_argument(
        "--skip-boolq",
        action="store_true",
        help="Skip processing BoolQ dataset"
    )
    parser.add_argument(
        "--skip-math",
        action="store_true",
        help="Skip processing MATH dataset"
    )
    parser.add_argument(
        "--skip-medqa",
        action="store_true",
        help="Skip processing MedQA dataset"
    )
    parser.add_argument(
        "--stop-on-error",
        action="store_true",
        help="Stop processing if any dataset fails (default: continue)"
    )
    
    args = parser.parse_args()
    main(args)