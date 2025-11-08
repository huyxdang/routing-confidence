"""Split BoolQ dataset into train, val, test"""

from datasets import load_dataset, DatasetDict

# Seed for reproducibility
RANDOM_SEED = 42


def split_boolq() -> DatasetDict:
    """
    Split BoolQ dataset.
    
    Strategy:
    - Use existing validation split as test split
    - Split original train into train (90%) and validation (10%)
    
    Returns:
        DatasetDict with train/validation/test splits
    """
    print("\n" + "="*60)
    print("Processing BoolQ dataset")
    print("="*60)
    
    # Load original dataset
    print("Loading google/boolq...")
    dataset = load_dataset("google/boolq")
    
    print(f"Original splits:")
    print(f"  Train: {len(dataset['train'])} examples")
    print(f"  Validation: {len(dataset['validation'])} examples")
    
    # Use validation as test
    test_split = dataset['validation']
    
    # Split train into new train (90%) and validation (10%)
    train_val_split = dataset['train'].train_test_split(test_size=0.1, seed=RANDOM_SEED)
    train_split = train_val_split['train']
    validation_split = train_val_split['test']
    
    # Create new dataset dict
    new_dataset = DatasetDict({
        'train': train_split,
        'validation': validation_split,
        'test': test_split
    })
    
    print(f"\nNew splits:")
    print(f"  Train: {len(new_dataset['train'])} examples ({len(new_dataset['train'])/len(dataset['train'])*100:.1f}% of original train)")
    print(f"  Validation: {len(new_dataset['validation'])} examples ({len(new_dataset['validation'])/len(dataset['train'])*100:.1f}% of original train)")
    print(f"  Test: {len(new_dataset['test'])} examples (original validation)")
    
    return new_dataset


if __name__ == "__main__":
    # Test the function
    dataset = split_boolq()
    print("\n" + "="*60)
    print("Split completed successfully!")
    print(f"Total examples: {sum(len(dataset[split]) for split in dataset)}")
    print("="*60)
