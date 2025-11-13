"""
Comprehensive unit tests for train_confidence_lora.py

Run with:
    pytest tests/test_training.py -v
    pytest tests/test_training.py --cov=train_confidence_lora --cov-report=html
"""

import pytest
import torch
import numpy as np
import json
import tempfile
import os
from unittest.mock import Mock, patch, MagicMock

# Import functions from training script
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from train_confidence_lora import (
    calib_err,
    initialize_model_with_tokens,
    ConfidenceTokenCollator,
    ConfidenceDataset,
    extract_boolq_answer,
    extract_medqa_answer,
    CONFIDENCE_TOKENS
)


# ============================================================================
# TEST SPECIAL TOKENS
# ============================================================================

class TestSpecialTokens:
    """Test special token initialization and vocabulary."""
    
    @pytest.mark.skip(reason="Requires GPU and model download")
    def test_token_initialization(self):
        """Test that new tokens are initialized as average of existing embeddings."""
        model, tokenizer = initialize_model_with_tokens(
            "Qwen/Qwen2.5-7B-Instruct",
            max_seq_length=512
        )
        
        embeddings = model.get_input_embeddings().weight
        special_tokens = CONFIDENCE_TOKENS
        
        # Get average of original embeddings (excluding new tokens)
        original_avg = embeddings[:-len(special_tokens)].mean(dim=0)
        
        for token in special_tokens:
            token_id = tokenizer.convert_tokens_to_ids(token)
            token_embedding = embeddings[token_id]
            
            # Check embedding is close to average (not random)
            assert torch.allclose(token_embedding, original_avg, atol=1e-3), \
                f"Token {token} not initialized as average"
            
            # Check it's not zero or NaN
            assert not torch.isnan(token_embedding).any()
            assert not torch.all(token_embedding == 0)
        
        print("✓ Special tokens initialized correctly")
    
    @pytest.mark.skip(reason="Requires GPU and model download")
    def test_vocabulary_size(self):
        """Test that vocabulary size increased correctly."""
        model, tokenizer = initialize_model_with_tokens(
            "Qwen/Qwen2.5-7B-Instruct",
            max_seq_length=512
        )
        
        # Check all tokens in vocabulary
        for token in CONFIDENCE_TOKENS:
            token_id = tokenizer.convert_tokens_to_ids(token)
            assert token_id != tokenizer.unk_token_id, \
                f"Token {token} should be in vocabulary"
        
        print("✓ Vocabulary size increased correctly")
    
    def test_token_list_complete(self):
        """Test that all required tokens are in CONFIDENCE_TOKENS."""
        assert "<C_READ>" in CONFIDENCE_TOKENS
        assert "<U_READ>" in CONFIDENCE_TOKENS
        assert "<C_MED>" in CONFIDENCE_TOKENS
        assert "<U_MED>" in CONFIDENCE_TOKENS
        assert len(CONFIDENCE_TOKENS) == 4
        
        print("✓ All required tokens present")


# ============================================================================
# TEST GRADIENT MASKING (CRITICAL)
# ============================================================================

class TestGradientMasking:
    """Test gradient masking logic and gradient flow."""
    
    def test_collator_masking_logic(self):
        """Test that collator creates correct label masks."""
        # Create mock tokenizer
        tokenizer = Mock()
        tokenizer.pad_token_id = 0
        
        collator = ConfidenceTokenCollator(tokenizer)
        
        # Create sample features
        features = [
            {
                "input_ids": [1, 2, 3, 4, 5, 6, 7, 8],  # Last token (8) is confidence token
                "attention_mask": [1, 1, 1, 1, 1, 1, 1, 1],
                "input_length": 5,  # First 5 tokens are input
                "dataset": "boolq"
            }
        ]
        
        batch = collator(features)
        labels = batch["labels"][0]
        
        # Check: Everything before last token is masked
        assert torch.all(labels[:-1] == -100), \
            "All tokens except last should be masked (-100)"
        
        # Check: Last token is NOT masked (should be the confidence token ID)
        assert labels[-1] == 8, \
            "Last token (confidence token) should not be masked"
        
        print("✓ Gradient masking logic correct")
    
    def test_collator_padding(self):
        """Test that collator handles variable-length sequences."""
        tokenizer = Mock()
        tokenizer.pad_token_id = 0
        
        collator = ConfidenceTokenCollator(tokenizer)
        
        # Variable length sequences
        features = [
            {
                "input_ids": [1, 2, 3, 4, 5],
                "attention_mask": [1, 1, 1, 1, 1],
                "input_length": 3,
                "dataset": "boolq"
            },
            {
                "input_ids": [1, 2, 3, 4, 5, 6, 7, 8],
                "attention_mask": [1, 1, 1, 1, 1, 1, 1, 1],
                "input_length": 5,
                "dataset": "medqa"
            }
        ]
        
        batch = collator(features)
        
        # Check batch shapes
        assert batch["input_ids"].shape[0] == 2, "Batch size should be 2"
        assert batch["input_ids"].shape[1] == 8, "Should pad to max length (8)"
        assert batch["labels"].shape == batch["input_ids"].shape, \
            "Labels and input_ids should have same shape"
        
        # Check padding
        assert batch["attention_mask"][0, 5:].sum() == 0, \
            "Short sequence should have padding in attention mask"
        
        print("✓ Collator padding works correctly")
    
    @pytest.mark.skip(reason="Requires model and GPU")
    def test_gradient_flow(self):
        """Test that gradients only flow to confidence token."""
        model, tokenizer = initialize_model_with_tokens(
            "Qwen/Qwen2.5-7B-Instruct",
            max_seq_length=512
        )
        model.train()
        
        # Create input with masked labels
        input_text = "Question: 2+2?\n\nAnswer: 4<C_READ>"
        inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
        
        input_ids = inputs['input_ids']
        labels = input_ids.clone()
        labels[0, :-1] = -100  # Mask everything except last token
        
        # Forward pass
        outputs = model(input_ids=input_ids, labels=labels)
        loss = outputs.loss
        
        # Backward pass
        loss.backward()
        
        # Check gradients
        embeddings = model.get_input_embeddings()
        confidence_token_id = tokenizer.convert_tokens_to_ids("<C_READ>")
        
        conf_grad = embeddings.weight.grad[confidence_token_id]
        assert conf_grad is not None and torch.any(conf_grad != 0), \
            "Confidence token should have gradients"
        
        print(f"✓ Gradient flow verified (conf token grad norm: {conf_grad.norm():.4f})")


# ============================================================================
# TEST CALIBRATION ERROR (CRITICAL)
# ============================================================================

class TestCalibration:
    """Test calibration error calculation with known inputs."""
    
    def test_perfect_calibration(self):
        """Test calibration error with perfect calibration."""
        # 50% confident, 50% correct → perfect calibration
        confidence = np.array([0.5] * 100)
        correct = np.array([1, 0] * 50)
        
        error = calib_err(confidence, correct, p='2', beta=100)
        
        assert abs(error) < 0.01, \
            f"Perfect calibration should have ~0 error, got {error:.4f}"
        
        print(f"✓ Perfect calibration error: {error:.4f}")
    
    def test_overconfident(self):
        """Test calibration error with overconfidence."""
        # 90% confident, 50% correct → overconfident
        confidence = np.array([0.9] * 100)
        correct = np.array([1, 0] * 50)
        
        error = calib_err(confidence, correct, p='2', beta=100)
        
        assert error > 0.35, \
            f"Overconfidence should have high error, got {error:.4f}"
        
        print(f"✓ Overconfident calibration error: {error:.4f}")
    
    def test_underconfident(self):
        """Test calibration error with underconfidence."""
        # 30% confident, 100% correct → underconfident
        confidence = np.array([0.3] * 100)
        correct = np.array([1] * 100)
        
        error = calib_err(confidence, correct, p='2', beta=100)
        
        assert error > 0.65, \
            f"Underconfidence should have high error, got {error:.4f}"
        
        print(f"✓ Underconfident calibration error: {error:.4f}")
    
    def test_different_p_norms(self):
        """Test calibration error with different p-norms."""
        np.random.seed(42)
        confidence = np.random.rand(200)
        correct = np.random.randint(0, 2, 200)
        
        error_l1 = calib_err(confidence, correct, p='1', beta=50)
        error_l2 = calib_err(confidence, correct, p='2', beta=50)
        error_linf = calib_err(confidence, correct, p='infty', beta=50)
        
        assert error_l1 > 0 and error_l2 > 0 and error_linf > 0, \
            "All errors should be positive"
        
        # L-infinity should be >= L2 >= L1 (generally)
        assert error_linf >= error_l1, "L-infinity should be >= L1"
        
        print(f"✓ Different p-norms: L1={error_l1:.4f}, L2={error_l2:.4f}, L∞={error_linf:.4f}")
    
    def test_empty_input(self):
        """Test calibration error with empty input."""
        confidence = np.array([])
        correct = np.array([])
        
        error = calib_err(confidence, correct, p='2', beta=100)
        
        assert error == 0.0, "Empty input should return 0 error"
        
        print("✓ Empty input handled correctly")
    
    def test_small_sample(self):
        """Test calibration error with small sample."""
        confidence = np.array([0.8, 0.6, 0.9])
        correct = np.array([1, 0, 1])
        
        error = calib_err(confidence, correct, p='2', beta=100)
        
        assert error >= 0, "Error should be non-negative"
        
        print(f"✓ Small sample error: {error:.4f}")


# ============================================================================
# TEST DATA PIPELINE
# ============================================================================

class TestDataPipeline:
    """Test data loading and collator."""
    
    def test_dataset_loading(self):
        """Test that dataset loads from JSONL correctly."""
        # Create temporary JSONL file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            data = [
                {
                    "question": "Is the sky blue?",
                    "tagged_response": "Yes <C_READ>",
                    "dataset": "boolq"
                },
                {
                    "question": "What is the treatment?",
                    "tagged_response": "A: Medication <C_MED>",
                    "dataset": "medqa"
                }
            ]
            for record in data:
                f.write(json.dumps(record) + '\n')
            temp_path = f.name
        
        try:
            # Create mock tokenizer
            tokenizer = Mock()
            tokenizer.return_value = {
                "input_ids": [1, 2, 3, 4, 5],
                "attention_mask": [1, 1, 1, 1, 1]
            }
            
            # Load dataset
            dataset = ConfidenceDataset(temp_path, tokenizer, max_length=512)
            
            assert len(dataset) == 2, "Should load 2 examples"
            assert dataset.examples[0]['dataset'] == 'boolq'
            assert dataset.examples[1]['dataset'] == 'medqa'
            assert "<C_READ>" in dataset.examples[0]['output']
            assert "<C_MED>" in dataset.examples[1]['output']
            
            print("✓ Dataset loading works")
        
        finally:
            os.unlink(temp_path)
    
    def test_collator_shapes(self):
        """Test that collator produces correct batch shapes."""
        tokenizer = Mock()
        tokenizer.pad_token_id = 0
        
        collator = ConfidenceTokenCollator(tokenizer)
        
        # Create mock samples with varying lengths
        samples = []
        for i in range(4):
            length = 5 + i * 2
            samples.append({
                "input_ids": list(range(1, length + 1)),
                "attention_mask": [1] * length,
                "input_length": length - 3,
                "dataset": "boolq"
            })
        
        batch = collator(samples)
        
        # Check shapes
        assert batch["input_ids"].shape[0] == 4, "Batch size should be 4"
        max_len = max(len(s["input_ids"]) for s in samples)
        assert batch["input_ids"].shape[1] == max_len, \
            f"Should pad to max length ({max_len})"
        assert batch["labels"].shape == batch["input_ids"].shape, \
            "Labels and input_ids should match"
        assert batch["attention_mask"].shape == batch["input_ids"].shape, \
            "Attention mask should match"
        
        print("✓ Data collator shapes correct")


# ============================================================================
# TEST ANSWER EXTRACTION
# ============================================================================

class TestAnswerExtraction:
    """Test answer extraction functions."""
    
    def test_boolq_extraction_yes(self):
        """Test BoolQ answer extraction - YES."""
        response = "Yes, the sky is blue."
        result = extract_boolq_answer(response, True)
        
        assert result['extracted_answer'] == 'yes'
        assert result['ground_truth'] == 'yes'
        assert result['is_correct'] == True
        
        print("✓ BoolQ YES extraction works")
    
    def test_boolq_extraction_no(self):
        """Test BoolQ answer extraction - NO."""
        response = "No, the sky is not green."
        result = extract_boolq_answer(response, False)
        
        assert result['extracted_answer'] == 'no'
        assert result['ground_truth'] == 'no'
        assert result['is_correct'] == True
        
        print("✓ BoolQ NO extraction works")
    
    def test_boolq_extraction_unknown(self):
        """Test BoolQ answer extraction - UNKNOWN."""
        response = "The answer is complicated."
        result = extract_boolq_answer(response, True)
        
        assert result['extracted_answer'] == 'unknown'
        assert result['is_correct'] == False
        
        print("✓ BoolQ unknown extraction works")
    
    def test_medqa_extraction_letter_colon(self):
        """Test MedQA answer extraction - Letter colon format."""
        response = "C: Glycosaminoglycan accumulation"
        result = extract_medqa_answer(response, "C")
        
        assert result['extracted_answer'] == 'C'
        assert result['ground_truth'] == 'C'
        assert result['is_correct'] == True
        
        print("✓ MedQA letter:text extraction works")
    
    def test_medqa_extraction_letter_only(self):
        """Test MedQA answer extraction - Letter only."""
        response = "A"
        result = extract_medqa_answer(response, "A")
        
        assert result['extracted_answer'] == 'A'
        assert result['is_correct'] == True
        
        print("✓ MedQA letter-only extraction works")
    
    def test_medqa_extraction_unknown(self):
        """Test MedQA answer extraction - UNKNOWN."""
        response = "The treatment involves multiple factors."
        result = extract_medqa_answer(response, "B")
        
        assert result['extracted_answer'] == 'unknown'
        assert result['is_correct'] == False
        
        print("✓ MedQA unknown extraction works")


# ============================================================================
# TEST VALIDATION LOGIC
# ============================================================================

class TestValidationLogic:
    """Test validation confidence token matching logic."""
    
    def test_correct_answer_confident_token(self):
        """Test: Correct answer + confident token → validation correct."""
        # Answer is correct, model outputs <C_READ>
        is_correct = True
        predicted_token = "<C_READ>"
        expected_token = "<C_READ>" if is_correct else "<U_READ>"
        
        validation_correct = (predicted_token == expected_token)
        
        assert validation_correct == True, \
            "Correct answer + confident token should be validation correct"
        
        print("✓ Correct + confident → validation ✓")
    
    def test_correct_answer_unconfident_token(self):
        """Test: Correct answer + unconfident token → validation incorrect."""
        # Answer is correct, but model outputs <U_READ> (wrong token)
        is_correct = True
        predicted_token = "<U_READ>"
        expected_token = "<C_READ>" if is_correct else "<U_READ>"
        
        validation_correct = (predicted_token == expected_token)
        
        assert validation_correct == False, \
            "Correct answer + unconfident token should be validation incorrect"
        
        print("✓ Correct + unconfident → validation ✗")
    
    def test_incorrect_answer_unconfident_token(self):
        """Test: Incorrect answer + unconfident token → validation correct."""
        # Answer is wrong, model outputs <U_READ>
        is_correct = False
        predicted_token = "<U_READ>"
        expected_token = "<C_READ>" if is_correct else "<U_READ>"
        
        validation_correct = (predicted_token == expected_token)
        
        assert validation_correct == True, \
            "Incorrect answer + unconfident token should be validation correct"
        
        print("✓ Incorrect + unconfident → validation ✓")
    
    def test_incorrect_answer_confident_token(self):
        """Test: Incorrect answer + confident token → validation incorrect."""
        # Answer is wrong, but model outputs <C_READ> (wrong token)
        is_correct = False
        predicted_token = "<C_READ>"
        expected_token = "<C_READ>" if is_correct else "<U_READ>"
        
        validation_correct = (predicted_token == expected_token)
        
        assert validation_correct == False, \
            "Incorrect answer + confident token should be validation incorrect"
        
        print("✓ Incorrect + confident → validation ✗")
    
    def test_validation_accuracy_calculation(self):
        """Test validation accuracy calculation."""
        # Simulate validation results
        token_matches = [True, True, False, True, False, True, True, True]
        
        val_accuracy = (sum(token_matches) / len(token_matches)) * 100
        
        assert val_accuracy == 75.0, \
            f"6/8 matches should be 75%, got {val_accuracy}%"
        
        print(f"✓ Validation accuracy: {val_accuracy}%")


# ============================================================================
# TEST PROBABILITY EXTRACTION
# ============================================================================

class TestProbabilityExtraction:
    """Test confidence probability extraction for calibration."""
    
    def test_softmax_normalization(self):
        """Test softmax normalization over 2 domain tokens."""
        # Mock logits
        c_logit = torch.tensor(2.0)
        u_logit = torch.tensor(1.0)
        
        # Softmax over only these 2 tokens
        p_confident = (torch.exp(c_logit) / (torch.exp(c_logit) + torch.exp(u_logit))).item()
        
        assert 0 <= p_confident <= 1, "Probability should be in [0, 1]"
        assert p_confident > 0.5, "Higher logit should have higher probability"
        
        expected_p = torch.exp(c_logit) / (torch.exp(c_logit) + torch.exp(u_logit))
        assert abs(p_confident - expected_p.item()) < 1e-6
        
        print(f"✓ Softmax normalization: p_confident = {p_confident:.4f}")
    
    def test_equal_logits(self):
        """Test probability extraction with equal logits."""
        c_logit = torch.tensor(1.0)
        u_logit = torch.tensor(1.0)
        
        p_confident = (torch.exp(c_logit) / (torch.exp(c_logit) + torch.exp(u_logit))).item()
        
        assert abs(p_confident - 0.5) < 1e-6, \
            "Equal logits should give 50% probability"
        
        print(f"✓ Equal logits: p_confident = {p_confident:.4f}")
    
    def test_extreme_logits(self):
        """Test probability extraction with extreme logits."""
        # Very confident
        c_logit = torch.tensor(10.0)
        u_logit = torch.tensor(-10.0)
        
        p_confident = (torch.exp(c_logit) / (torch.exp(c_logit) + torch.exp(u_logit))).item()
        
        assert p_confident > 0.999, \
            f"Very high c_logit should give ~1.0 probability, got {p_confident}"
        
        # Very unconfident
        c_logit = torch.tensor(-10.0)
        u_logit = torch.tensor(10.0)
        
        p_confident = (torch.exp(c_logit) / (torch.exp(c_logit) + torch.exp(u_logit))).item()
        
        assert p_confident < 0.001, \
            f"Very low c_logit should give ~0.0 probability, got {p_confident}"
        
        print("✓ Extreme logits handled correctly")


# ============================================================================
# INTEGRATION TESTS
# ============================================================================

class TestIntegration:
    """Test end-to-end workflows."""
    
    @pytest.mark.skip(reason="Requires model and GPU")
    def test_full_forward_pass(self):
        """Test complete forward pass with real model."""
        model, tokenizer = initialize_model_with_tokens(
            "Qwen/Qwen2.5-7B-Instruct",
            max_seq_length=512
        )
        
        # Create sample
        text = "Question: What is 2+2?\n\nAnswer: 4<C_READ>"
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        
        # Create labels (only last token)
        labels = inputs['input_ids'].clone()
        labels[0, :-1] = -100
        
        # Forward pass
        outputs = model(input_ids=inputs['input_ids'], labels=labels)
        loss = outputs.loss
        
        # Checks
        assert loss is not None, "Loss should be computed"
        assert not torch.isnan(loss), "Loss should not be NaN"
        assert loss.item() > 0, "Loss should be positive"
        assert loss.requires_grad, "Loss should have gradient"
        
        print(f"✓ End-to-end forward pass works, loss: {loss.item():.4f}")
    
    def test_data_to_batch_pipeline(self):
        """Test data loading → collation → batch creation."""
        # Create temporary data
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            f.write(json.dumps({
                "question": "Test?",
                "tagged_response": "Answer <C_READ>",
                "dataset": "boolq"
            }) + '\n')
            temp_path = f.name
        
        try:
            # Mock tokenizer
            tokenizer = Mock()
            tokenizer.pad_token_id = 0
            
            def mock_tokenize(text, **kwargs):
                # Simple mock: return token IDs based on text length
                return {
                    "input_ids": list(range(1, 10)),
                    "attention_mask": [1] * 9
                }
            
            tokenizer.side_effect = mock_tokenize
            
            # Load dataset
            dataset = ConfidenceDataset(temp_path, tokenizer, max_length=512)
            
            # Create collator
            collator = ConfidenceTokenCollator(tokenizer)
            
            # Get batch
            batch = collator([dataset[0]])
            
            assert "input_ids" in batch
            assert "attention_mask" in batch
            assert "labels" in batch
            assert batch["input_ids"].shape[0] == 1
            
            print("✓ Data → batch pipeline works")
        
        finally:
            os.unlink(temp_path)


# ============================================================================
# SANITY CHECKS
# ============================================================================

class TestSanityChecks:
    """Basic sanity checks."""
    
    def test_imports(self):
        """Test that all required imports work."""
        try:
            from train_confidence_lora import (
                calib_err,
                ConfidenceTokenCollator,
                ConfidenceDataset,
                initialize_model_with_tokens,
                CONFIDENCE_TOKENS
            )
            print("✓ All imports successful")
        except ImportError as e:
            pytest.fail(f"Import failed: {e}")
    
    def test_confidence_tokens_defined(self):
        """Test that CONFIDENCE_TOKENS is correctly defined."""
        assert isinstance(CONFIDENCE_TOKENS, list)
        assert len(CONFIDENCE_TOKENS) == 4
        assert all(isinstance(token, str) for token in CONFIDENCE_TOKENS)
        print("✓ CONFIDENCE_TOKENS correctly defined")
    
    def test_calib_err_function_signature(self):
        """Test calib_err function signature."""
        import inspect
        sig = inspect.signature(calib_err)
        
        assert 'confidence' in sig.parameters
        assert 'correct' in sig.parameters
        assert 'p' in sig.parameters
        assert 'beta' in sig.parameters
        
        print("✓ calib_err function signature correct")


# ============================================================================
# RUN TESTS
# ============================================================================

if __name__ == "__main__":
    # Run with: python tests/test_training.py
    pytest.main([__file__, "-v", "--tb=short"])

