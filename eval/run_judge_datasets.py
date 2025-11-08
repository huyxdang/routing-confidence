"""
Evaluate model predictions on datasets using LLM-as-judge and append domain-specific confidence tokens.
"""
import os
import json
import copy
import argparse
import asyncio
from typing import Literal
from pydantic import BaseModel
from openai import AsyncOpenAI
from tqdm.asyncio import tqdm_asyncio
from datasets import load_dataset


client = AsyncOpenAI(timeout=300.0, max_retries=1)


# Dataset configurations
DATASET_CONFIGS = {
    'math': {
        'hf_path': 'huyxdang/math-split',
        'question_field': 'problem',
        'answer_field': 'solution',
        'domain': 'MATH'
    },
    'medqa': {
        'hf_path': 'huyxdang/medqa-split',
        'question_field': 'question',
        'answer_field': 'answer',
        'domain': 'MED'
    },
    'boolq': {
        'hf_path': 'huyxdang/boolq-split',
        'question_field': 'question',
        'answer_field': 'answer',
        'domain': 'READ'
    }
}


# Judge prompt template
JUDGE_PROMPT = """Judge whether the following [response] to [question] is correct or not based on the [correct_answer] below.

[question]: {question}

[response]: {response}

Your judgement must be in the format and criteria specified below:

extracted_final_answer: The final exact answer extracted from the [response]. Put the extracted answer as 'None' if there is no exact, final answer to extract from the response.

[correct_answer]: {correct_answer}

reasoning: Explain why the extracted_final_answer is correct or incorrect based on [correct_answer], focusing only on if there are meaningful differences between [correct_answer] and the extracted_final_answer. Do not comment on any background to the problem, do not attempt to solve the problem, do not argue for any answer different than [correct_answer], focus only on whether the answers match.

correct: Answer 'yes' if extracted_final_answer matches the [correct_answer] given above, or is within a small margin of error for numerical problems. Answer 'no' otherwise, i.e. if there is any inconsistency, ambiguity, non-equivalency, or if the extracted answer is incorrect."""


class ExtractedAnswer(BaseModel):
    extracted_final_answer: str
    reasoning: str
    correct: Literal["yes", "no"]
    strict: Literal[True]  # 100% reliability


async def extract_answer(question, correct_answer, response, judge_model):
    """Use LLM to judge if the response is correct."""
    prompt = JUDGE_PROMPT.format(
        question=question,
        correct_answer=correct_answer,
        response=response
    )
    
    try:
        response = await client.beta.chat.completions.parse(
            model=judge_model,
            max_completion_tokens=4096,
            messages=[
                {"role": "user", "content": prompt}
            ],
            response_format=ExtractedAnswer,
        )
        content = response.choices[0].message.parsed
        return {
            "correct_answer": correct_answer,
            "model_answer": content.extracted_final_answer,
            "reasoning": content.reasoning,
            "correct": content.correct
        }
    except Exception as e:
        print(f"Error in judge: {e}")
        return None


async def judge_prediction(index, prediction, dataset_example, domain, judge_model):
    """Judge a single prediction and add confidence token."""
    
    if "judge_response" in prediction:  # already judged
        return index, prediction
    
    question = prediction["question"]
    correct_answer = prediction["correct_answer"]
    response = prediction["response"]
    
    # Get judge result
    judge_result = await extract_answer(question, correct_answer, response, judge_model)
    
    if judge_result is None:
        return None, None
    
    # Determine if correct
    is_correct = judge_result["correct"] == "yes"
    
    # Append domain-specific confidence token
    if is_correct:
        confidence_token = f"<CN_{domain}>"
    else:
        confidence_token = f"<UN_{domain}>"
    
    # Create tagged response
    tagged_response = f"{response} {confidence_token}"
    
    # Update prediction
    prediction["judge_response"] = judge_result
    prediction["correct"] = is_correct
    prediction["tagged_response"] = tagged_response
    prediction["confidence_token"] = confidence_token
    
    return index, prediction


async def judge_all_predictions(predictions, dataset_examples, domain, judge_model, num_workers):
    """Judge all predictions in parallel."""
    
    async def bound_func(index, prediction, example):
        async with semaphore:
            result = await judge_prediction(index, prediction, example, domain, judge_model)
            return result
    
    semaphore = asyncio.Semaphore(num_workers)
    
    # Create tasks for all predictions
    tasks = []
    for index, prediction in predictions.items():
        example = dataset_examples[int(index)]
        tasks.append(bound_func(index, prediction, example))
    
    # Run all tasks with progress bar
    results = await tqdm_asyncio.gather(*tasks, desc="Judging predictions")
    
    return results


def load_dataset_split(dataset_name, split='train'):
    """Load dataset from HuggingFace."""
    config = DATASET_CONFIGS[dataset_name]
    print(f"Loading {dataset_name} dataset from {config['hf_path']} (split: {split})")
    
    dataset = load_dataset(config['hf_path'], split=split)
    
    # Convert to dict indexed by position
    examples = {}
    for i, example in enumerate(dataset):
        examples[i] = dict(example)
    
    print(f"Loaded {len(examples)} examples from {dataset_name}")
    return examples


def calculate_accuracy(predictions):
    """Calculate accuracy from judged predictions."""
    total = 0
    correct = 0
    
    for pred in predictions.values():
        if "correct" in pred:
            total += 1
            if pred["correct"]:
                correct += 1
    
    if total > 0:
        accuracy = correct / total * 100
        return accuracy, correct, total
    return 0, 0, 0


async def main(args):
    """Main function to judge predictions."""
    
    # Validate dataset
    if args.dataset not in DATASET_CONFIGS:
        raise ValueError(f"Unknown dataset: {args.dataset}. Must be one of: {list(DATASET_CONFIGS.keys())}")
    
    config = DATASET_CONFIGS[args.dataset]
    domain = config['domain']
    
    print(f"\n{'='*60}")
    print(f"Judging predictions for {args.dataset} dataset")
    print(f"Domain: {domain}")
    print(f"Confidence tokens: <CN_{domain}> (correct) / <UN_{domain}> (incorrect)")
    print(f"{'='*60}\n")
    
    # Load predictions
    print(f"Loading predictions from: {args.predictions}")
    with open(args.predictions, 'r') as f:
        predictions = json.load(f)
    
    print(f"Loaded {len(predictions)} predictions")
    
    # Set output file
    if args.output is None:
        base_name = os.path.splitext(args.predictions)[0]
        args.output = f"{base_name}_tagged.json"
    
    # Create output directory if needed
    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # Load existing judged predictions if resuming
    if os.path.exists(args.output):
        with open(args.output, 'r') as f:
            judged_predictions = json.load(f)
        print(f"Resuming from existing file with {len(judged_predictions)} judged predictions")
        # Merge with loaded predictions to ensure we have all fields
        for idx, judged_pred in judged_predictions.items():
            if idx in predictions:
                predictions[idx] = judged_pred
    
    # Load dataset
    dataset_examples = load_dataset_split(args.dataset, args.split)
    
    # Count how many need judging
    to_judge = sum(1 for p in predictions.values() if "judge_response" not in p)
    print(f"\nPredictions to judge: {to_judge}/{len(predictions)}")
    
    if to_judge == 0:
        print("All predictions already judged!")
    else:
        print(f"\nStarting judging with {args.num_workers} workers...")
        print(f"Judge model: {args.judge}\n")
        
        # Judge all predictions
        results = await judge_all_predictions(
            predictions,
            dataset_examples,
            domain,
            args.judge,
            args.num_workers
        )
        
        # Update predictions with results
        for index, prediction in results:
            if index is not None and prediction is not None:
                predictions[index] = prediction
        
        # Save results
        with open(args.output, 'w') as f:
            json.dump(predictions, f, indent=2)
        
        print(f"\nSaved judged predictions to: {args.output}")
    
    # Calculate and display accuracy
    accuracy, correct, total = calculate_accuracy(predictions)
    
    print(f"\n{'='*60}")
    print(f"RESULTS")
    print(f"{'='*60}")
    print(f"Dataset: {args.dataset} ({domain})")
    print(f"Total predictions: {total}")
    print(f"Correct: {correct}")
    print(f"Incorrect: {total - correct}")
    print(f"Accuracy: {accuracy:.2f}%")
    print(f"{'='*60}\n")
    
    # Show sample tagged responses
    print("Sample tagged responses:")
    sample_count = 0
    for idx, pred in predictions.items():
        if "tagged_response" in pred and sample_count < 3:
            print(f"\n[Example {sample_count + 1}]")
            print(f"Correct: {pred['correct']}")
            print(f"Token: {pred['confidence_token']}")
            print(f"Tagged: {pred['tagged_response'][:200]}...")
            sample_count += 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Judge model predictions and append domain-specific confidence tokens"
    )
    parser.add_argument(
        "--predictions",
        type=str,
        required=True,
        help="Path to predictions JSON file"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=['math', 'medqa', 'boolq'],
        help="Dataset name"
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        help="Dataset split (default: train)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file for tagged predictions (default: {predictions}_tagged.json)"
    )
    parser.add_argument(
        "--judge",
        type=str,
        default="gpt-4o-2024-08-06",
        help="Judge model to use (default: gpt-4o-2024-08-06)"
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=50,
        help="Number of parallel workers for judging (default: 50)"
    )
    
    args = parser.parse_args()
    asyncio.run(main(args))

