"""
Combine all tagged datasets into one JSONL file per model.
Each line contains: question, tagged_response, dataset
"""
import json
import os


def combine_tagged_datasets(model_name, datasets=['boolq', 'medqa']):
    """Combine all tagged datasets for a model into one JSONL file."""
    
    print(f"\n{'='*70}")
    print(f"Combining tagged datasets for: {model_name}")
    print(f"{'='*70}\n")
    
    model_dir = f"tagged/{model_name}"
    output_file = os.path.join(model_dir, f"{model_name}_all_tagged.jsonl")
    
    # Ensure output directory exists
    os.makedirs(model_dir, exist_ok=True)
    
    total_records = 0
    dataset_counts = {}
    
    # Open output file for writing
    with open(output_file, 'w') as out_f:
        for dataset in datasets:
            input_file = os.path.join(model_dir, f"{model_name}_{dataset}_train_tagged.json")
            
            print(f"Loading: {input_file}")
            
            try:
                with open(input_file, 'r') as in_f:
                    data = json.load(in_f)
                
                count = 0
                for idx, item in data.items():
                    # Extract only the 3 fields we need
                    record = {
                        "question": item["question"],
                        "tagged_response": item["tagged_response"],
                        "dataset": dataset
                    }
                    
                    # Write as single line JSON
                    out_f.write(json.dumps(record) + '\n')
                    count += 1
                    total_records += 1
                
                dataset_counts[dataset] = count
                print(f"  ✓ Added {count} records from {dataset}")
                
            except FileNotFoundError:
                print(f"  ⚠ File not found: {input_file}")
            except Exception as e:
                print(f"  ⚠ Error processing {dataset}: {e}")
    
    print(f"\n{'='*70}")
    print(f"SUMMARY for {model_name}")
    print(f"{'='*70}")
    for dataset, count in dataset_counts.items():
        print(f"  {dataset}: {count} records")
    print(f"  Total: {total_records} records")
    print(f"\nSaved to: {output_file}")
    print(f"{'='*70}")
    
    # Show sample
    print(f"\nSample records (first 2):")
    with open(output_file, 'r') as f:
        for i, line in enumerate(f):
            if i >= 2:
                break
            record = json.loads(line)
            print(f"\n[Record {i+1}]")
            print(f"  Dataset: {record['dataset']}")
            print(f"  Question: {record['question'][:80]}...")
            print(f"  Tagged Response: {record['tagged_response'][:80]}...")


def main():
    """Main function to combine tagged datasets for all models."""
    
    models = ['mistral', 'qwen']
    datasets = ['boolq', 'medqa']  # No math for now
    
    print(f"\n{'='*70}")
    print("COMBINING TAGGED DATASETS")
    print(f"{'='*70}")
    print(f"Models: {models}")
    print(f"Datasets: {datasets}")
    print(f"Output format: JSONL (one JSON per line)")
    print(f"Fields: question, tagged_response, dataset")
    print(f"{'='*70}")
    
    for model in models:
        try:
            combine_tagged_datasets(model, datasets)
        except Exception as e:
            print(f"\n⚠ Error processing {model}: {e}")
    
    print(f"\n{'='*70}")
    print("✓ All models processed!")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()