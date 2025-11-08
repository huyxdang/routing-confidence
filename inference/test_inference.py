"""
Quick test script to verify inference pipeline works correctly.
Tests dataset loading and prompt generation without running full inference.
"""
from run_dataset_inference import (
    DATASET_CONFIGS,
    load_dataset_split,
    create_prompt
)


def test_dataset(dataset_name):
    """Test loading and prompt generation for a dataset."""
    print(f"\n{'='*60}")
    print(f"Testing {dataset_name.upper()} dataset")
    print(f"{'='*60}")
    
    # Get config
    config = DATASET_CONFIGS[dataset_name]
    print(f"\nConfiguration:")
    print(f"  HF Path: {config['hf_path']}")
    print(f"  Domain: {config['domain']}")
    print(f"  Max Tokens: {config['max_tokens']}")
    print(f"  Question Field: {config['question_field']}")
    print(f"  Answer Field: {config['answer_field']}")
    
    # Load dataset
    print(f"\nLoading dataset...")
    examples = load_dataset_split(dataset_name, split='train')
    
    # Show first example
    first_example = examples[0]
    print(f"\nFirst example fields: {list(first_example.keys())}")
    
    # Generate prompt
    print(f"\nGenerated prompt for first example:")
    print(f"{'-'*60}")
    prompt = create_prompt(first_example, dataset_name)
    print(prompt)
    print(f"{'-'*60}")
    
    print(f"\nGround truth answer:")
    print(f"  {first_example[config['answer_field']][:200]}...")
    
    print(f"\n✓ {dataset_name.upper()} test passed!")


def main():
    """Test all datasets."""
    print("\n" + "="*60)
    print("INFERENCE PIPELINE TEST")
    print("="*60)
    print("\nThis script tests dataset loading and prompt generation")
    print("without running actual model inference.")
    
    datasets = ['math', 'medqa', 'boolq']
    
    for dataset_name in datasets:
        try:
            test_dataset(dataset_name)
        except Exception as e:
            print(f"\n✗ Error testing {dataset_name}: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\n{'='*60}")
    print("TEST SUMMARY")
    print(f"{'='*60}")
    print("If all tests passed, you can now run full inference with:")
    print("  python run_dataset_inference.py --model_name <model> --dataset <dataset>")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()

