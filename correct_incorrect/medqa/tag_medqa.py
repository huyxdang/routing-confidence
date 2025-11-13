"""
Tag MedQA predictions with confidence tokens.
- Correct answers get <C_MED>
- Incorrect answers get <U_MED>

Creates both separate tagged files and a combined file.
"""
import json
import os
import argparse


def tag_predictions(predictions, confidence_token):
    """Add tagged_response field with confidence token."""
    tagged = {}
    for idx, pred in predictions.items():
        # Create a copy with tagged response
        tagged_pred = pred.copy()
        tagged_pred["tagged_response"] = f"{pred['response']} {confidence_token}"
        tagged[idx] = tagged_pred
    
    return tagged


def process_medqa_model(model_name, base_dir="correct_incorrect/medqa"):
    """Process correct and incorrect files for a model."""
    
    model_dir = os.path.join(base_dir, model_name)
    
    # Define file paths
    correct_file = os.path.join(model_dir, f"{model_name}_medqa_train_correct.json")
    incorrect_file = os.path.join(model_dir, f"{model_name}_medqa_train_incorrect.json")
    
    correct_tagged_file = os.path.join(model_dir, f"{model_name}_medqa_train_correct_tagged.json")
    incorrect_tagged_file = os.path.join(model_dir, f"{model_name}_medqa_train_incorrect_tagged.json")
    combined_tagged_file = os.path.join(model_dir, f"{model_name}_medqa_train_tagged.json")
    
    print(f"\n{'='*70}")
    print(f"Processing MedQA predictions for: {model_name}")
    print(f"{'='*70}")
    
    # Load correct predictions
    print(f"\nLoading: {correct_file}")
    with open(correct_file, 'r') as f:
        correct_predictions = json.load(f)
    print(f"  Loaded {len(correct_predictions)} correct predictions")
    
    # Load incorrect predictions
    print(f"Loading: {incorrect_file}")
    with open(incorrect_file, 'r') as f:
        incorrect_predictions = json.load(f)
    print(f"  Loaded {len(incorrect_predictions)} incorrect predictions")
    
    # Tag correct predictions with <C_MED>
    print(f"\nTagging correct predictions with <C_MED>...")
    correct_tagged = tag_predictions(correct_predictions, "<C_MED>")
    
    # Tag incorrect predictions with <U_MED>
    print(f"Tagging incorrect predictions with <U_MED>...")
    incorrect_tagged = tag_predictions(incorrect_predictions, "<U_MED>")
    
    # Save separate tagged files
    print(f"\nSaving separate tagged files...")
    with open(correct_tagged_file, 'w') as f:
        json.dump(correct_tagged, f, indent=2)
    print(f"  ✓ {correct_tagged_file} ({len(correct_tagged)} predictions)")
    
    with open(incorrect_tagged_file, 'w') as f:
        json.dump(incorrect_tagged, f, indent=2)
    print(f"  ✓ {incorrect_tagged_file} ({len(incorrect_tagged)} predictions)")
    
    # Combine and save
    print(f"\nCombining all predictions...")
    combined = {}
    combined.update(correct_tagged)
    combined.update(incorrect_tagged)
    
    with open(combined_tagged_file, 'w') as f:
        json.dump(combined, f, indent=2)
    print(f"  ✓ {combined_tagged_file} ({len(combined)} predictions)")
    
    # Summary
    print(f"\n{'='*70}")
    print(f"SUMMARY for {model_name}")
    print(f"{'='*70}")
    print(f"Correct predictions: {len(correct_tagged)} (<C_MED>)")
    print(f"Incorrect predictions: {len(incorrect_tagged)} (<U_MED>)")
    print(f"Total combined: {len(combined)}")
    print(f"Accuracy: {len(correct_tagged)/len(combined)*100:.2f}%")
    print(f"{'='*70}")
    
    # Show sample
    print(f"\nSample tagged responses:")
    
    # Sample from correct
    if correct_tagged:
        sample_idx = list(correct_tagged.keys())[0]
        sample = correct_tagged[sample_idx]
        print(f"\n[Correct Example]")
        print(f"  Original: {sample['response'][:80]}...")
        print(f"  Tagged: {sample['tagged_response'][:80]}...")
    
    # Sample from incorrect
    if incorrect_tagged:
        sample_idx = list(incorrect_tagged.keys())[0]
        sample = incorrect_tagged[sample_idx]
        print(f"\n[Incorrect Example]")
        print(f"  Original: {sample['response'][:80]}...")
        print(f"  Tagged: {sample['tagged_response'][:80]}...")


def main():
    parser = argparse.ArgumentParser(
        description="Tag MedQA predictions with confidence tokens"
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=['mistral', 'qwen', 'all'],
        default='all',
        help="Model to process (default: all)"
    )
    parser.add_argument(
        "--base-dir",
        type=str,
        default="correct_incorrect/medqa",
        help="Base directory for MedQA predictions"
    )
    
    args = parser.parse_args()
    
    if args.model == 'all':
        models = ['mistral', 'qwen']
    else:
        models = [args.model]
    
    for model in models:
        try:
            process_medqa_model(model, args.base_dir)
        except FileNotFoundError as e:
            print(f"\n⚠ Error processing {model}: {e}")
            print(f"  Make sure files exist in: {os.path.join(args.base_dir, model)}/")
        except Exception as e:
            print(f"\n⚠ Unexpected error processing {model}: {e}")
    
    print(f"\n{'='*70}")
    print("✓ All models processed!")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()

