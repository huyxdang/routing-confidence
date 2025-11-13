"""
LoRA Fine-tuning Script for Domain-Specific Confidence Tokens

Train models to append confidence tokens (<C_READ>, <U_READ>, <C_MED>, <U_MED>) 
after predictions, with domain-specific confidence calibration.

Usage:
    python train_confidence_lora.py \
        --model_name "Qwen/Qwen2.5-7B-Instruct" \
        --train_data "tagged/qwen/qwen_all_tagged.jsonl" \
        --output_dir "./checkpoints/qwen_confidence" \
        --hf_repo "huyxdang/qwen-confidence-lora"
"""

import os
import json
import argparse
import re
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    TrainerCallback,
    TrainerState,
    TrainerControl,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from tqdm import tqdm


# Special tokens for confidence
CONFIDENCE_TOKENS = ["<C_READ>", "<U_READ>", "<C_MED>", "<U_MED>"]


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="LoRA fine-tuning for domain-specific confidence tokens"
    )
    
    # Model configuration
    parser.add_argument(
        "--model_name",
        type=str,
        required=True,
        choices=["Qwen/Qwen2.5-7B-Instruct", "mistralai/Mistral-7B-Instruct-v0.2"],
        help="Base model to fine-tune"
    )
    parser.add_argument(
        "--max_seq_length",
        type=int,
        default=2048,
        help="Maximum sequence length"
    )
    
    # Data paths
    parser.add_argument(
        "--train_data",
        type=str,
        required=True,
        help="Path to training JSONL file"
    )
    parser.add_argument(
        "--val_medqa_split",
        type=str,
        default="huyxdang/medqa-split",
        help="HuggingFace dataset for MedQA validation"
    )
    parser.add_argument(
        "--val_boolq_split",
        type=str,
        default="huyxdang/boolq-split",
        help="HuggingFace dataset for BoolQ validation"
    )
    
    # LoRA configuration
    parser.add_argument("--lora_r", type=int, default=16, help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=32, help="LoRA alpha")
    parser.add_argument("--lora_dropout", type=float, default=0.05, help="LoRA dropout")
    
    # Training configuration
    parser.add_argument("--num_epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--max_steps", type=int, default=None, help="Maximum number of training steps (overrides num_epochs if set)")
    parser.add_argument("--batch_size", type=int, default=8, help="Per-device batch size")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--warmup_ratio", type=float, default=0.03, help="Warmup ratio")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument("--max_grad_norm", type=float, default=1.0, help="Max gradient norm")
    
    # Calibration
    parser.add_argument("--calib_beta", type=int, default=100, help="Beta for calibration error bins")
    parser.add_argument("--calib_p_norm", type=str, default="2", choices=["1", "2", "infty"])
    
    # Output configuration
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory")
    parser.add_argument("--hf_repo", type=str, default=None, help="HuggingFace Hub repository")
    parser.add_argument("--logging_steps", type=int, default=10, help="Logging frequency")
    parser.add_argument("--save_steps", type=int, default=500, help="Checkpoint save frequency")
    parser.add_argument("--eval_steps", type=int, default=500, help="Evaluation frequency")
    
    # System
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--bf16", action="store_true", default=True, help="Use bf16")
    parser.add_argument("--fp16", action="store_true", default=False, help="Use fp16")
    
    return parser.parse_args()


def calib_err(confidence: np.ndarray, correct: np.ndarray, p: str = '2', beta: int = 100) -> float:
    """
    Calculate calibration error.
    
    Args:
        confidence: Array of confidence scores (0-1)
        correct: Array of binary correctness (0 or 1)
        p: Norm type ('1', '2', or 'infty')
        beta: Target bin size
    
    Returns:
        Calibration error score
    """
    if len(confidence) == 0:
        return 0.0
    
    idxs = np.argsort(confidence)
    confidence = confidence[idxs]
    correct = correct[idxs]
    
    bins = [[i * beta, (i + 1) * beta] for i in range(len(confidence) // beta)]
    if len(bins) == 0 or bins[-1][1] < len(confidence):
        bins.append([bins[-1][1] if bins else 0, len(confidence)])
    
    cerr = 0
    total_examples = len(confidence)
    
    for i in range(len(bins) - 1):
        bin_confidence = confidence[bins[i][0]:bins[i][1]]
        bin_correct = correct[bins[i][0]:bins[i][1]]
        num_examples_in_bin = len(bin_confidence)
        
        if num_examples_in_bin > 0:
            difference = np.abs(np.nanmean(bin_confidence) - np.nanmean(bin_correct))
            
            if p == '2':
                cerr += num_examples_in_bin / total_examples * np.square(difference)
            elif p == '1':
                cerr += num_examples_in_bin / total_examples * difference
            elif p == 'infty' or p == 'infinity' or p == 'max':
                cerr = np.maximum(cerr, difference)
            else:
                raise ValueError("p must be '1', '2', or 'infty'")
    
    if p == '2':
        cerr = np.sqrt(cerr)
    
    return cerr


def initialize_model_with_tokens(
    model_name: str,
    max_seq_length: int = 2048
) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    """
    Load model and add special tokens with proper initialization.
    
    CRITICAL: New token embeddings are initialized as average of existing embeddings.
    """
    print(f"\nLoading base model: {model_name}")
    print(f"Using 8-bit quantization for H200 (141GB available)")
    
    # Configure 8-bit quantization
    bnb_config = BitsAndBytesConfig(
        load_in_8bit=True,
        llm_int8_threshold=6.0,
        llm_int8_has_fp16_weight=False,
    )
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load model with 8-bit quantization
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.float16,
        trust_remote_code=True,
    )
    
    print(f"Original vocabulary size: {len(tokenizer)}")
    
    # Add special tokens
    num_added = tokenizer.add_special_tokens({
        "additional_special_tokens": CONFIDENCE_TOKENS
    })
    print(f"Added {num_added} special tokens: {CONFIDENCE_TOKENS}")
    
    # Resize model embeddings
    model.resize_token_embeddings(len(tokenizer))
    print(f"New vocabulary size: {len(tokenizer)}")
    
    # CRITICAL: Initialize new tokens as average of existing embeddings
    print("\nInitializing new token embeddings...")
    embeddings = model.get_input_embeddings()
    
    with torch.no_grad():
        # Calculate average of all original embeddings (excluding new tokens)
        original_embeddings = embeddings.weight[:-num_added]
        avg_embedding = original_embeddings.mean(dim=0)
        
        print(f"Average embedding stats:")
        print(f"  Mean: {avg_embedding.mean().item():.6f}")
        print(f"  Std: {avg_embedding.std().item():.6f}")
        
        # Initialize each new token
        for i, token in enumerate(CONFIDENCE_TOKENS):
            token_id = tokenizer.convert_tokens_to_ids(token)
            embeddings.weight[token_id] = avg_embedding.clone()
            
            print(f"✓ Initialized {token} (ID: {token_id}) as average embedding")
    
    # Verify initialization
    for token in CONFIDENCE_TOKENS:
        token_id = tokenizer.convert_tokens_to_ids(token)
        assert token_id != tokenizer.unk_token_id, f"Token {token} not in vocabulary!"
        
        # Check it's not NaN or zero
        emb = embeddings.weight[token_id]
        assert not torch.isnan(emb).any(), f"Token {token} has NaN values!"
        assert not torch.all(emb == 0), f"Token {token} is all zeros!"
    
    print("✓ All tokens initialized and verified successfully\n")
    
    return model, tokenizer


class ConfidenceDataset(Dataset):
    """Dataset for confidence token training."""
    
    def __init__(self, jsonl_path: str, tokenizer: AutoTokenizer, max_length: int = 2048):
        """
        Load training data from JSONL file.
        
        Expected format:
        {
          "question": "...",
          "tagged_response": "...answer... <C_READ>",
          "dataset": "boolq" or "medqa"
        }
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.examples = []
        
        print(f"Loading training data from: {jsonl_path}")
        
        with open(jsonl_path, 'r') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    record = json.loads(line)
                    
                    # Transform to training format
                    input_text = f"Question: {record['question']}\n\nAnswer:"
                    output_text = record['tagged_response']
                    dataset_name = record['dataset']
                    
                    self.examples.append({
                        "input": input_text,
                        "output": output_text,
                        "dataset": dataset_name
                    })
                    
                except json.JSONDecodeError as e:
                    print(f"Warning: Skipping line {line_num} due to JSON error: {e}")
                    continue
        
        print(f"Loaded {len(self.examples)} training examples")
        
        # Count by dataset
        dataset_counts = {}
        for ex in self.examples:
            dataset_counts[ex['dataset']] = dataset_counts.get(ex['dataset'], 0) + 1
        print(f"Dataset breakdown: {dataset_counts}")
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        example = self.examples[idx]
        
        # Tokenize full sequence (input + output with confidence token)
        full_text = example['input'] + example['output']
        tokenized = self.tokenizer(
            full_text,
            truncation=True,
            max_length=self.max_length,
            padding=False,
            return_tensors=None
        )
        
        return {
            "input_ids": tokenized['input_ids'],
            "attention_mask": tokenized['attention_mask'],
        }


class ConfidenceTokenCollator:
    """
    Data collator with gradient masking.
    
    CRITICAL: Loss is computed ONLY on the confidence token (last token in output).
    """
    
    def __init__(self, tokenizer: AutoTokenizer, pad_to_multiple_of: Optional[int] = None):
        self.tokenizer = tokenizer
        self.pad_to_multiple_of = pad_to_multiple_of
    
    def __call__(self, features: List[Dict]) -> Dict[str, torch.Tensor]:
        # Extract components
        input_ids = [f['input_ids'] for f in features]
        attention_masks = [f['attention_mask'] for f in features]
        
        # Pad sequences
        max_length = max(len(ids) for ids in input_ids)
        if self.pad_to_multiple_of:
            max_length = ((max_length + self.pad_to_multiple_of - 1) 
                          // self.pad_to_multiple_of * self.pad_to_multiple_of)
        
        padded_input_ids = []
        padded_attention_masks = []
        labels = []
        
        for i, ids in enumerate(input_ids):
            # Pad input_ids
            padding_length = max_length - len(ids)
            padded_ids = ids + [self.tokenizer.pad_token_id] * padding_length
            padded_input_ids.append(padded_ids)
            
            # Pad attention mask
            padded_mask = attention_masks[i] + [0] * padding_length
            padded_attention_masks.append(padded_mask)
            
            # Create labels with gradient masking
            # Format: [-100, -100, ..., -100, confidence_token_id]
            #         ^input masked    ^prediction masked  ^only this gets loss
            label = [-100] * max_length
            
            # The last non-pad token should be the confidence token
            last_token_pos = len(ids) - 1
            label[last_token_pos] = ids[last_token_pos]
            
            # Pad labels
            labels.append(label)
        
        return {
            "input_ids": torch.tensor(padded_input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(padded_attention_masks, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long)
        }


# Import answer extraction functions from eval_simple.py
def extract_boolq_answer(response: str, ground_truth: bool) -> Dict:
    """Extract and compare BoolQ answer (Yes/No vs True/False)."""
    response_lower = response.lower().strip()
    
    # Look for yes/no at the beginning (first 50 chars)
    beginning = response_lower[:50]
    
    # Extract yes/no
    if 'yes' in beginning:
        extracted = 'yes'
    elif 'no' in beginning:
        extracted = 'no'
    else:
        # Try to find it anywhere in first sentence
        first_sentence = response_lower.split('.')[0] if '.' in response_lower else response_lower[:100]
        if 'yes' in first_sentence:
            extracted = 'yes'
        elif 'no' in first_sentence:
            extracted = 'no'
        else:
            extracted = 'unknown'
    
    # Ground truth is True/False
    expected = 'yes' if ground_truth else 'no'
    is_correct = (extracted == expected)
    
    return {
        "extracted_answer": extracted,
        "ground_truth": expected,
        "is_correct": is_correct
    }


def extract_medqa_answer(response: str, ground_truth: str) -> Dict:
    """Extract and compare MedQA answer (A/B/C/D/E)."""
    response_stripped = response.strip()
    
    # Look for pattern "LETTER: TEXT" at the beginning
    letter_colon_pattern = r'^([ABCDE]):\s*'
    match = re.match(letter_colon_pattern, response_stripped, re.IGNORECASE)
    
    if match:
        extracted_letter = match.group(1).upper()
    else:
        # Try to find just the letter at the beginning
        response_upper = response_stripped.upper()
        beginning = response_upper[:20]
        
        option_pattern = r'\b([ABCDE])\b'
        matches = re.findall(option_pattern, beginning)
        
        if matches:
            extracted_letter = matches[0]
        else:
            # Try to find it in first sentence
            first_sentence = response_upper.split('.')[0] if '.' in response_upper else response_upper[:100]
            matches = re.findall(option_pattern, first_sentence)
            extracted_letter = matches[0] if matches else 'unknown'
    
    # Ground truth should be a single letter
    expected_letter = ground_truth.strip().upper()
    is_correct = (extracted_letter == expected_letter)
    
    return {
        "extracted_answer": extracted_letter,
        "ground_truth": expected_letter,
        "is_correct": is_correct
    }


class ValidationCallback(TrainerCallback):
    """
    Callback for validation with confidence token checking and calibration.
    
    Validation process:
    1. Generate answer + confidence token
    2. Parse answer and predicted confidence token
    3. Check if answer is correct
    4. Determine expected confidence token
    5. Compare predicted vs expected token → validation accuracy
    6. Extract probabilities → calibration error
    """
    
    def __init__(
        self,
        model: AutoModelForCausalLM,
        tokenizer: AutoTokenizer,
        val_datasets: Dict[str, Dataset],
        calib_beta: int = 100,
        calib_p_norm: str = '2',
        output_dir: str = "./checkpoints"
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.val_datasets = val_datasets
        self.calib_beta = calib_beta
        self.calib_p_norm = calib_p_norm
        self.output_dir = output_dir
        
        self.best_val_accuracy = 0.0
        self.best_calib_error = float('inf')
    
    def on_evaluate(self, args, state: TrainerState, control: TrainerControl, **kwargs):
        """Run validation after each evaluation."""
        
        print(f"\n{'='*70}")
        print(f"VALIDATION - Epoch {state.epoch}")
        print(f"{'='*70}")
        
        all_results = {}
        
        for dataset_name, val_dataset in self.val_datasets.items():
            print(f"\nValidating on {dataset_name}...")
            
            results = self._validate_dataset(
                val_dataset,
                dataset_name,
                max_samples=50  # Reduced for quick testing (use 500 for final training)
            )
            
            all_results[dataset_name] = results
            
            print(f"\n{dataset_name.upper()} Results:")
            print(f"  Validation Accuracy: {results['val_accuracy']:.2f}%")
            print(f"  Calibration Error: {results['calib_error']:.4f}")
            print(f"  Samples evaluated: {results['num_samples']}")
            print(f"  Missing tokens: {results['missing_tokens']} ({results['missing_token_rate']:.1f}%)")
        
        # Aggregate metrics
        avg_val_acc = np.mean([r['val_accuracy'] for r in all_results.values()])
        avg_calib_err = np.mean([r['calib_error'] for r in all_results.values()])
        
        print(f"\n{'='*70}")
        print(f"OVERALL VALIDATION METRICS:")
        print(f"  Average Validation Accuracy: {avg_val_acc:.2f}%")
        print(f"  Average Calibration Error: {avg_calib_err:.4f}")
        print(f"{'='*70}\n")
        
        # Save best model
        if avg_val_acc > self.best_val_accuracy:
            self.best_val_accuracy = avg_val_acc
            best_path = os.path.join(self.output_dir, "best_accuracy")
            print(f"✓ New best validation accuracy! Saving to {best_path}")
            self.model.save_pretrained(best_path)
            self.tokenizer.save_pretrained(best_path)
        
        if avg_calib_err < self.best_calib_error:
            self.best_calib_error = avg_calib_err
            best_path = os.path.join(self.output_dir, "best_calibration")
            print(f"✓ New best calibration error! Saving to {best_path}")
            self.model.save_pretrained(best_path)
            self.tokenizer.save_pretrained(best_path)
        
        # Save reliability data
        reliability_data = {
            "epoch": state.epoch,
            "results": all_results,
            "avg_val_accuracy": float(avg_val_acc),
            "avg_calib_error": float(avg_calib_err)
        }
        
        reliability_path = os.path.join(
            self.output_dir,
            f"reliability_epoch_{int(state.epoch)}.json"
        )
        with open(reliability_path, 'w') as f:
            json.dump(reliability_data, f, indent=2)
        
        return control
    
    def _validate_dataset(
        self,
        val_dataset: Dataset,
        dataset_name: str,
        max_samples: int = 500
    ) -> Dict:
        """Validate on a single dataset."""
        
        self.model.eval()
        
        confidence_scores = []
        correctness = []
        token_matches = []
        missing_tokens = 0  # Track how many samples had no confidence token
        
        # Limit samples for speed
        num_samples = min(len(val_dataset), max_samples)
        
        for i in tqdm(range(num_samples), desc=f"Validating {dataset_name}"):
            example = val_dataset[i]
            
            # Get ground truth
            if dataset_name == "boolq":
                question = example['question']
                ground_truth = example['answer']
                domain_tokens = ("<C_READ>", "<U_READ>")
            else:  # medqa
                question = example['question']
                ground_truth = example['answer_idx'] if 'answer_idx' in example else example['answer']
                domain_tokens = ("<C_MED>", "<U_MED>")
            
            # Generate prediction
            prompt = f"Question: {question}\n\nAnswer:"
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=100,
                    do_sample=False,  # Greedy decoding
                    pad_token_id=self.tokenizer.pad_token_id
                )
            
            generated_text = self.tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=False)
            
            # Parse answer and confidence token
            predicted_token = None
            for token in CONFIDENCE_TOKENS:
                if token in generated_text:
                    predicted_token = token
                    break
            
            # If no confidence token found, treat as WRONG
            if predicted_token is None:
                # Model failed to output confidence token - count as incorrect
                token_matches.append(False)
                missing_tokens += 1
                # Skip calibration for this sample (no confidence to measure)
                continue
            
            # Remove confidence token to get answer
            answer_text = generated_text.replace(predicted_token, '').strip()
            
            # Check if answer is correct
            if dataset_name == "boolq":
                result = extract_boolq_answer(answer_text, ground_truth)
            else:  # medqa
                result = extract_medqa_answer(answer_text, ground_truth)
            
            is_correct = result['is_correct']
            
            # Determine expected confidence token
            expected_token = domain_tokens[0] if is_correct else domain_tokens[1]
            
            # Validation accuracy: does predicted token match expected?
            token_match = (predicted_token == expected_token)
            token_matches.append(token_match)
            
            # Get probability for calibration
            # Re-run forward pass to get logits
            full_input = prompt + answer_text
            full_inputs = self.tokenizer(full_input, return_tensors="pt").to(self.model.device)
            
            with torch.no_grad():
                forward_outputs = self.model(**full_inputs)
                logits = forward_outputs.logits[0, -1, :]  # Last position
            
            # Extract probability over domain tokens
            c_token_id = self.tokenizer.convert_tokens_to_ids(domain_tokens[0])
            u_token_id = self.tokenizer.convert_tokens_to_ids(domain_tokens[1])
            
            c_logit = logits[c_token_id]
            u_logit = logits[u_token_id]
            
            # Softmax over only these 2 tokens
            p_confident = (torch.exp(c_logit) / (torch.exp(c_logit) + torch.exp(u_logit))).item()
            
            confidence_scores.append(p_confident)
            correctness.append(1 if is_correct else 0)
        
        # Calculate metrics
        val_accuracy = (sum(token_matches) / len(token_matches) * 100) if token_matches else 0.0
        
        calib_error = calib_err(
            np.array(confidence_scores),
            np.array(correctness),
            p=self.calib_p_norm,
            beta=self.calib_beta
        ) if confidence_scores else 0.0
        
        return {
            "val_accuracy": val_accuracy,
            "calib_error": calib_error,
            "num_samples": len(token_matches),
            "missing_tokens": missing_tokens,
            "missing_token_rate": (missing_tokens / len(token_matches) * 100) if token_matches else 0.0,
            "confidence_scores": confidence_scores,
            "correctness": correctness
        }


def setup_lora(model: AutoModelForCausalLM, args) -> AutoModelForCausalLM:
    """Setup LoRA configuration using standard PEFT."""
    
    print("\nConfiguring LoRA...")
    print(f"  rank (r): {args.lora_r}")
    print(f"  alpha: {args.lora_alpha}")
    print(f"  dropout: {args.lora_dropout}")
    
    # Prepare model for k-bit training
    model = prepare_model_for_kbit_training(model)
    
    # Configure LoRA
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                       "gate_proj", "up_proj", "down_proj"],
        bias="none",
        task_type="CAUSAL_LM",
    )
    
    # Apply LoRA
    model = get_peft_model(model, lora_config)
    
    # Enable gradient checkpointing
    model.enable_input_require_grads()
    
    # Print trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nTrainable parameters: {trainable_params:,} / {total_params:,} ({trainable_params/total_params*100:.2f}%)")
    
    return model


def main():
    """Main training function."""
    
    args = parse_arguments()
    
    # Set seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_short = args.model_name.split('/')[-1].lower()
    output_dir = os.path.join(args.output_dir, f"{model_short}_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\n{'='*70}")
    print(f"LoRA Fine-tuning for Domain-Specific Confidence Tokens")
    print(f"{'='*70}")
    print(f"Model: {args.model_name}")
    print(f"Training data: {args.train_data}")
    print(f"Output directory: {output_dir}")
    print(f"{'='*70}\n")
    
    # Initialize model with special tokens
    model, tokenizer = initialize_model_with_tokens(
        args.model_name,
        args.max_seq_length
    )
    
    # Setup LoRA
    model = setup_lora(model, args)
    
    # Load training data
    train_dataset = ConfidenceDataset(
        args.train_data,
        tokenizer,
        args.max_seq_length
    )
    
    # Load validation datasets
    print("\nLoading validation datasets...")
    val_datasets = {}
    
    try:
        medqa_val = load_dataset(args.val_medqa_split, split="validation")
        val_datasets["medqa"] = medqa_val
        print(f"✓ Loaded MedQA validation: {len(medqa_val)} examples")
    except Exception as e:
        print(f"⚠ Warning: Could not load MedQA validation: {e}")
    
    try:
        boolq_val = load_dataset(args.val_boolq_split, split="validation")
        val_datasets["boolq"] = boolq_val
        print(f"✓ Loaded BoolQ validation: {len(boolq_val)} examples")
    except Exception as e:
        print(f"⚠ Warning: Could not load BoolQ validation: {e}")
    
    # Data collator
    data_collator = ConfidenceTokenCollator(tokenizer)
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=args.num_epochs,
        max_steps=args.max_steps if args.max_steps else -1,  # -1 means use num_epochs
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        lr_scheduler_type="cosine",
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        max_grad_norm=args.max_grad_norm,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        eval_steps=args.eval_steps,
        save_total_limit=3,
        bf16=args.bf16,
        fp16=args.fp16,
        optim="adamw_8bit",
        report_to=["tensorboard"],
        logging_dir=os.path.join(output_dir, "logs"),
        seed=args.seed,
    )
    
    # Validation callback
    val_callback = ValidationCallback(
        model,
        tokenizer,
        val_datasets,
        args.calib_beta,
        args.calib_p_norm,
        output_dir
    )
    
    # Trainer (use small eval dataset to trigger callbacks)
    # We'll use first 10 examples as dummy eval dataset just to trigger evaluation
    dummy_eval_dataset = torch.utils.data.Subset(train_dataset, range(min(10, len(train_dataset))))
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=dummy_eval_dataset,  # Dummy dataset to trigger evaluation
        data_collator=data_collator,
        callbacks=[val_callback],
    )
    
    # Train
    print(f"\n{'='*70}")
    print("Starting training...")
    print(f"{'='*70}\n")
    
    trainer.train()
    
    # Run final validation manually if not already done
    print(f"\n{'='*70}")
    print("Running final validation...")
    print(f"{'='*70}")
    val_callback.on_evaluate(
        training_args,
        trainer.state,
        trainer.control
    )
    
    # Save final model
    final_path = os.path.join(output_dir, "final")
    print(f"\n{'='*70}")
    print(f"Saving final model to: {final_path}")
    model.save_pretrained(final_path)
    tokenizer.save_pretrained(final_path)
    
    # Push to HuggingFace Hub if specified
    if args.hf_repo:
        print(f"Pushing to HuggingFace Hub: {args.hf_repo}")
        try:
            model.push_to_hub(args.hf_repo)
            tokenizer.push_to_hub(args.hf_repo)
            print("✓ Successfully pushed to Hub")
        except Exception as e:
            print(f"⚠ Warning: Could not push to Hub: {e}")
    
    print(f"\n{'='*70}")
    print("Training complete!")
    print(f"Best validation accuracy: {val_callback.best_val_accuracy:.2f}%")
    print(f"Best calibration error: {val_callback.best_calib_error:.4f}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()

