"""Split MedQA dataset into train, val, test"""

import kagglehub
import json
from datasets import Dataset, DatasetDict
from pathlib import Path


def load_jsonl(file_path):
    """Load data from a JSONL file."""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data


def split_medqa() -> DatasetDict:
    """
    Split MedQA dataset (US English version from Kaggle).
    
    Strategy:
    - Download from Kaggle using kagglehub
    - Load US version (English)
    - Use original splits: train.jsonl, dev.jsonl (as validation), test.jsonl
    - No additional splitting performed
    
    Returns:
        DatasetDict with train/validation/test splits
    """
    print("\n" + "="*60)
    print("Processing MedQA dataset (US English version)")
    print("="*60)
    
    # Download dataset from Kaggle
    print("Downloading MedQA dataset from Kaggle...")
    path = kagglehub.dataset_download("moaaztameer/medqa-usmle")
    print(f"Dataset downloaded to: {path}")
    
    # Construct paths to US version files
    base_path = Path(path) / "MedQA-USMLE" / "questions" / "US"
    
    train_file = base_path / "train.jsonl"
    dev_file = base_path / "dev.jsonl"
    test_file = base_path / "test.jsonl"
    
    # Check if files exist
    for file_path in [train_file, dev_file, test_file]:
        if not file_path.exists():
            raise FileNotFoundError(f"Expected file not found: {file_path}")
    
    print(f"\nLoading data from:")
    print(f"  Train: {train_file}")
    print(f"  Dev: {dev_file}")
    print(f"  Test: {test_file}")
    
    # Load the data
    train_data = load_jsonl(train_file)
    dev_data = load_jsonl(dev_file)
    test_data = load_jsonl(test_file)
    
    print(f"\nOriginal splits:")
    print(f"  Train: {len(train_data)} examples")
    print(f"  Dev: {len(dev_data)} examples")
    print(f"  Test: {len(test_data)} examples")
    
    # Convert to HuggingFace datasets using original splits
    train_dataset = Dataset.from_list(train_data)
    validation_dataset = Dataset.from_list(dev_data)
    test_dataset = Dataset.from_list(test_data)
    
    # Create new dataset dict with original splits
    new_dataset = DatasetDict({
        'train': train_dataset,
        'validation': validation_dataset,
        'test': test_dataset
    })
    
    print(f"\nFinal splits (using original data):")
    print(f"  Train: {len(new_dataset['train'])} examples")
    print(f"  Validation: {len(new_dataset['validation'])} examples (from dev.jsonl)")
    print(f"  Test: {len(new_dataset['test'])} examples")
    
    return new_dataset


if __name__ == "__main__":
    # Test the function
    dataset = split_medqa()
    print("\n" + "="*60)
    print("Split completed successfully!")
    print(f"Total examples: {sum(len(dataset[split]) for split in dataset)}")
    print("="*60)