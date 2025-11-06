# data/prepare_dataset.py
"""
Download and prepare SimpleQA-verified dataset (subset of questions)
"""
import json
import random
import argparse
import os
from datasets import load_dataset

def prepare_simpleqa_subset(num_questions=300, seed=42, output_file=None):
    """
    Download SimpleQA-verified and extract a subset of questions
    
    Args:
        num_questions: Number of questions to sample
        seed: Random seed for reproducibility
        output_file: Output file path (if None, auto-generates based on num_questions)
    """
    print(f"Loading SimpleQA-verified dataset...")
    dataset = load_dataset("google/simpleqa-verified", split="eval")
    
    print(f"Total questions in dataset: {len(dataset)}")
    
    # Validate number of questions
    if num_questions > len(dataset):
        print(f"Warning: Requested {num_questions} questions but dataset only has {len(dataset)}. Using all {len(dataset)} questions.")
        num_questions = len(dataset)
    
    # Set seed for reproducibility
    random.seed(seed)
    
    # Sample questions
    indices = random.sample(range(len(dataset)), num_questions)
    indices.sort()  # Keep in order for consistency
    
    subset = dataset.select(indices)
    
    # Convert to our format
    questions = []
    for i, item in enumerate(subset):
        questions.append({
            "id": f"simpleqa_{i}",
            "original_index": int(item["original_index"]),
            "question": item["problem"],
            "answer": item["answer"],
            "topic": item["topic"],
            "answer_type": item["answer_type"],
            "multi_step": bool(item["multi_step"]),
            "requires_reasoning": bool(item["requires_reasoning"]),
            "urls": item["urls"]
        })
    
    # Determine output file
    if output_file is None:
        output_file = f"data/simpleqa_{num_questions}.json"
    
    # Create directory if needed
    output_dir = os.path.dirname(output_file)
    if output_dir:  # Only create directory if there's a path component
        os.makedirs(output_dir, exist_ok=True)
    
    # Save to file
    with open(output_file, "w") as f:
        json.dump(questions, f, indent=2)
    
    print(f"Saved {len(questions)} questions to {output_file}")
    
    # Print some statistics
    print("\n=== Dataset Statistics ===")
    topics = {}
    answer_types = {}
    for q in questions:
        topics[q["topic"]] = topics.get(q["topic"], 0) + 1
        answer_types[q["answer_type"]] = answer_types.get(q["answer_type"], 0) + 1
    
    print("\nTopics:")
    for topic, count in sorted(topics.items(), key=lambda x: x[1], reverse=True):
        print(f"  {topic}: {count}")
    
    print("\nAnswer Types:")
    for atype, count in sorted(answer_types.items(), key=lambda x: x[1], reverse=True):
        print(f"  {atype}: {count}")
    
    print(f"\nMulti-step questions: {sum(1 for q in questions if q['multi_step'])}")
    print(f"Requires reasoning: {sum(1 for q in questions if q['requires_reasoning'])}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download and prepare SimpleQA-verified dataset subset")
    parser.add_argument(
        "--num_questions",
        type=int,
        default=300,
        help="Number of questions to sample from the dataset (default: 300)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file path (default: data/simpleqa_{num_questions}.json)"
    )
    
    args = parser.parse_args()
    
    # Prepare dataset
    prepare_simpleqa_subset(
        num_questions=args.num_questions,
        seed=args.seed,
        output_file=args.output
    )

