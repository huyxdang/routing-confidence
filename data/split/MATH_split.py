"""Split MATH dataset into train, val, test"""

from datasets import load_dataset, DatasetDict

# Seed for reproducibility
RANDOM_SEED = 42


def split_math() -> DatasetDict:
    """
    Split MATH dataset.
    
    Strategy:
    - Use existing MATH dataset from huyxdang/MATH-dataset (7.5k train, 5k test)
    - Split train (7.5k) into train (90%) and validation (10%)
    - Keep existing test split as test
    
    Returns:
        DatasetDict with train/validation/test splits
    """
    print("\n" + "="*60)
    print("Processing MATH dataset")
    print("="*60)
    
    # Load original dataset from huyxdang/MATH-dataset
    print("Loading huyxdang/MATH-dataset...")
    dataset = load_dataset("huyxdang/MATH-dataset")
    
    print(f"Original splits:")
    print(f"  Train: {len(dataset['train'])} examples")
    print(f"  Test: {len(dataset['test'])} examples")
    
    # Keep test split
    test_split = dataset['test']
    
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
    print(f"  Test: {len(new_dataset['test'])} examples (original test)")
    
    return new_dataset


if __name__ == "__main__":
    # Test the function
    dataset = split_math()
    print("\n" + "="*60)
    print("Split completed successfully!")
    print(f"Total examples: {sum(len(dataset[split]) for split in dataset)}")
    print("="*60)