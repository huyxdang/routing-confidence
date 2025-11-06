# inference/run_qwen_inference.py
"""
Run inference with Qwen models using vLLM for fast batch processing
Confidence scores prompted in [0,1] range
"""
import json
import argparse
from vllm import LLM, SamplingParams

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

def run_inference(model_name, questions_file, output_file, tensor_parallel_size=1, max_tokens=128, batch_size=50):
    """Run inference on all questions using vLLM with partial progress saving"""
    # Load questions
    with open(questions_file, "r") as f:
        questions = json.load(f)
    
    print(f"Loaded {len(questions)} questions from {questions_file}")
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
            predictions[questions[question_idx]["id"]] = {
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
        help="HuggingFace model name (e.g., Qwen/Qwen2.5-7B-Instruct)"
    )
    parser.add_argument(
        "--questions",
        type=str,
        default="data/simpleqa_300.json",
        help="Path to questions file"
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