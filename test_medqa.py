"""
Script to explore the MedQA prediction file structure and show sample values for each field.
"""
import json

def explore_medqa_predictions():
    """Load and display MedQA prediction file structure with examples."""
    
    print("="*80)
    print("EXPLORING MEDQA PREDICTION FILE")
    print("="*80)
    
    # Load prediction file
    pred_file = 'inference/predictions/medqa/mistral_medqa_train.json'
    print(f"\nLoading: {pred_file}")
    
    with open(pred_file, 'r') as f:
        predictions = json.load(f)
    
    print(f"Total predictions: {len(predictions)}")
    
    # Get first few examples
    first_keys = list(predictions.keys())[:5]
    # Get more keys for statistics
    stats_keys = list(predictions.keys())[:min(100, len(predictions))]
    
    # Show column names from first example
    print("\n" + "="*80)
    print("FIELDS IN PREDICTION FILE")
    print("="*80)
    first_example = predictions[first_keys[0]]
    fields = list(first_example.keys())
    print(f"Fields: {fields}")
    
    # Show first 3 examples with all fields
    print("\n" + "="*80)
    print("SAMPLE EXAMPLES (First 3)")
    print("="*80)
    
    for i, key in enumerate(first_keys[:3]):
        print(f"\n{'='*80}")
        print(f"EXAMPLE {i} (Index: {key})")
        print(f"{'='*80}")
        
        example = predictions[key]
        
        for field in fields:
            value = example[field]
            print(f"\n[{field}]")
            
            # Pretty print based on type
            if isinstance(value, dict):
                print(json.dumps(value, indent=2, ensure_ascii=False))
            elif isinstance(value, str):
                # Truncate long strings
                if len(value) > 400:
                    print(f"{value[:400]}...")
                else:
                    print(value)
            else:
                print(value)
    
    # Show statistics
    print("\n" + "="*80)
    print("FIELD STATISTICS")
    print("="*80)
    
    # Check answer types
    answer_lengths = []
    question_lengths = []
    response_lengths = []
    
    for key in first_keys[:100]:
        example = predictions[key]
        if 'answer' in example:
            answer_lengths.append(len(example['correct_answer']))
        if 'question' in example:
            question_lengths.append(len(example['question']))
        if 'response' in example:
            response_lengths.append(len(example['response']))
    
    print(f"\n[question]")
    print(f"  Type: Text")
    if question_lengths:
        print(f"  Avg length (first 100): {sum(question_lengths)/len(question_lengths):.0f} chars")
        print(f"  Min length: {min(question_lengths)}")
        print(f"  Max length: {max(question_lengths)}")
    
    print(f"\n[correct_answer]")
    print(f"  Type: Text")
    if answer_lengths:
        print(f"  Avg length (first 100): {sum(answer_lengths)/len(answer_lengths):.0f} chars")
        print(f"  Sample answers (first 10):")
        for key in first_keys[:10]:
            ans = predictions[key]['correct_answer']
            if len(ans) > 60:
                print(f"    - {ans[:60]}...")
            else:
                print(f"    - {ans}")
    
    print(f"\n[response]")
    print(f"  Type: Text")
    if response_lengths:
        print(f"  Avg length (first 100): {sum(response_lengths)/len(response_lengths):.0f} chars")
        print(f"  Min length: {min(response_lengths)}")
        print(f"  Max length: {max(response_lengths)}")
    
    # Analyze response patterns
    print("\n" + "="*80)
    print("RESPONSE PATTERNS ANALYSIS")
    print("="*80)
    
    pattern_counts = {
        'starts_with_letter': 0,
        'starts_with_letter_colon': 0,
        'contains_explanation': 0,
        'other': 0
    }
    
    for key in first_keys[:100]:
        response = predictions[key]['response'].strip()
        
        if len(response) > 0 and response[0] in 'ABCDE':
            if len(response) > 1 and response[1] == ':':
                pattern_counts['starts_with_letter_colon'] += 1
            else:
                pattern_counts['starts_with_letter'] += 1
        else:
            pattern_counts['other'] += 1
        
        if 'Explanation' in response or 'explanation' in response:
            pattern_counts['contains_explanation'] += 1
    
    print("\nPattern breakdown (first 100 examples):")
    for pattern, count in pattern_counts.items():
        print(f"  {pattern}: {count}")

if __name__ == "__main__":
    explore_medqa_predictions()

