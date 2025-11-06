# data/prepare_dataset.py
"""
Download and prepare SimpleQA-verified dataset (300 questions subset)
"""
import json
import random
from datasets import load_dataset

def prepare_simpleqa_subset(num_questions=300, seed=42):
    """
    Download SimpleQA-verified and extract a subset of questions
    """
    print(f"Loading SimpleQA-verified dataset...")
    dataset = load_dataset("google/simpleqa-verified", split="eval")
    
    print(f"Total questions in dataset: {len(dataset)}")
    
    # Set seed for reproducibility
    random.seed(seed)
    
    # Sample 300 questions
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
    
    # Save to file
    output_file = "data/simpleqa_300.json"
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
    prepare_simpleqa_subset()

