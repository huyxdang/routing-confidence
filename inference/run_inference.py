# inference/run_qwen_inference.py
"""
Run inference with Qwen models using vLLM for fast batch processing
Confidence scores prompted in [0,1] range
"""
import json
import argparse
import os
from vllm import LLM, SamplingParams
from datasets import load_dataset

# Confidence prompting template - IMPORTANT: Confidence must be in [0, 1] range
CONFIDENCE_PROMPT = """Answer the following question and provide your confidence level.

Question: {question}

Please provide:
1. Your answer to the question
2. Your confidence in this answer as a decimal number between 0.0 and 1.0 (where 0.0 = no confidence at all, 1.0 = completely certain)

IMPORTANT: The confidence score MUST be a decimal between 0 and 1 (e.g., 0.75, 0.9, 0.5).

Format your response EXACTLY as:
Answer: [your answer]
Confidence: [decimal number between 0.0 and 1.0]

Example:
Answer: Paris
Confidence: 0.95"""

def load_questions(questions_path):
    """
    Load questions from local JSON file or HuggingFace dataset.
    
    Args:
        questions_path: Path to local JSON file or HuggingFace dataset name
        
    Returns:
        List of question dictionaries with 'original_index' and 'question' fields
    """
    # Check if it's a local file
    if os.path.isfile(questions_path) or (questions_path.endswith('.json') and '/' in questions_path):
        # Load from local JSON file
        print(f"Loading questions from local file: {questions_path}")
        with open(questions_path, "r") as f:
            questions = json.load(f)
        
        if not isinstance(questions, list):
            raise ValueError(f"Local questions file must contain a JSON array, got {type(questions)}")
        
        # Validate required fields
        if questions and not all(key in questions[0] for key in ['original_index', 'question']):
            raise ValueError("Questions must contain 'original_index' and 'question' fields")
        
        return questions
    else:
        # Load from HuggingFace
        print(f"Loading questions from HuggingFace: {questions_path}")
        dataset = load_dataset(questions_path, split="eval").to_dict()
        # Convert to list of dictionaries
        questions = [dict(zip(dataset.keys(), values)) for values in zip(*dataset.values())]
        
        # Ensure we have the required fields
        if questions:
            # Handle field name differences (SimpleQA-verified uses 'problem' not 'question')
            if 'question' not in questions[0] and 'problem' in questions[0]:
                for q in questions:
                    q['question'] = q['problem']
            
            # Ensure original_index exists
            if 'original_index' not in questions[0]:
                # If using original SimpleQA-verified, it should have original_index
                # If not, we might need to use the index
                if 'original_index' not in dataset:
                    print("Warning: Dataset doesn't have 'original_index', using array index")
                    for i, q in enumerate(questions):
                        q['original_index'] = i
        
        return questions

def run_inference(model_name, questions_file, output_file, tensor_parallel_size=1, max_tokens=128, batch_size=50):
    """Run inference on all questions using vLLM with partial progress saving"""
    
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # Load questions (local JSON or HuggingFace)
    questions = load_questions(questions_file)
    
    print(f"Loaded {len(questions)} questions")
    print(f"Loading model: {model_name}")
    
    # Initialize vLLM
    llm = LLM(
        model=model_name,
        tensor_parallel_size=tensor_parallel_size,
        trust_remote_code=True
    )
    
    # Sampling params for deterministic generation
    params = SamplingParams(
        temperature=0.0,
        max_tokens=max_tokens,
        skip_special_tokens=True
    )
    
    # Process in batches and save progress
    predictions = {}
    total_questions = len(questions)
    all_outputs = []
    
    for batch_start in range(0, total_questions, batch_size):
        batch_end = min(batch_start + batch_size, total_questions)
        batch_questions = questions[batch_start:batch_end]
        
        print(f"Processing questions {batch_start + 1}-{batch_end}/{total_questions}...")
        
        # Create prompts for this batch
        batch_prompts = [
            CONFIDENCE_PROMPT.format(question=q["question"])
            for q in batch_questions
        ]
        
        # Generate for this batch
        batch_outputs = llm.generate(batch_prompts, params)
        all_outputs.extend(batch_outputs)
        
        # Add to predictions
        for i, output in enumerate(batch_outputs):
            question_idx = batch_start + i
            # Use original_index as the key for matching
            original_index = questions[question_idx]["original_index"]
            predictions[original_index] = {
                "response": output.outputs[0].text.strip()
            }
        
        # Save partial progress
        with open(output_file, "w") as f:
            json.dump(predictions, f, indent=2)
        
        print(f"  Saved {len(predictions)}/{total_questions} predictions")
    
    print(f"\nCompleted! Saved all {len(predictions)} predictions to {output_file}")
    
    # Calculate average tokens
    if all_outputs:
        avg_tokens = sum(len(o.outputs[0].token_ids) for o in all_outputs) / len(all_outputs)
        print(f"Average tokens per response: {avg_tokens:.1f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name",
        type=str,
        required=True,
        help="HuggingFace model name (e.g., Qwen/Qwen2.5-7B)"
    )
    parser.add_argument(
        "--questions",
        type=str,
        default="data/simpleqa_300.json",
        help="Path to local JSON file or HuggingFace dataset name (e.g., google/simpleqa-verified)"
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output file for predictions"
    )
    parser.add_argument(
        "--tensor_parallel_size",
        type=int,
        default=1,
        help="Number of GPUs for tensor parallelism (default: 1)"
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=128,
        help="Maximum tokens to generate per response (default: 128)"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=50,
        help="Number of questions to process before saving progress (default: 50)"
    )
    
    args = parser.parse_args()
    run_inference(
        args.model_name,
        args.questions,
        args.output,
        args.tensor_parallel_size,
        args.max_tokens,
        args.batch_size
    )