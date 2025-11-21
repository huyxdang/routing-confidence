"""
Configuration file for routing confidence project.
Centralizes token definitions, dataset configurations, and constants.
"""

# ============================================================================
# Confidence Tokens
# ============================================================================
# Format: <C_DOMAIN> for correct, <U_DOMAIN> for incorrect
# Used in pipeline/tagging.py and train_sft.py

CONFIDENCE_TOKENS = {
    'medqa': {
        'correct': '<C_MED>',
        'incorrect': '<U_MED>',
        'domain': 'MED'
    },
    'boolq': {
        'correct': '<C_READ>',
        'incorrect': '<U_READ>',
        'domain': 'READ'
    },
    'math': {
        'correct': '<C_MATH>',
        'incorrect': '<U_MATH>',
        'domain': 'MATH'
    }
}

# Alternative format used in eval/run_judge_datasets.py
# Format: <CN_DOMAIN> for correct, <UN_DOMAIN> for incorrect
# NOTE: This is inconsistent with the above format. Consider standardizing.
JUDGE_TOKENS = {
    'MED': {
        'correct': '<CN_MED>',
        'incorrect': '<UN_MED>'
    },
    'READ': {
        'correct': '<CN_READ>',
        'incorrect': '<UN_READ>'
    },
    'MATH': {
        'correct': '<CN_MATH>',
        'incorrect': '<UN_MATH>'
    }
}

# All special tokens used in training
ALL_SPECIAL_TOKENS = [
    '<C_READ>', '<U_READ>',
    '<C_MED>', '<U_MED>',
    '<C_MATH>', '<U_MATH>',
    # Alternative format (if needed)
    '<CN_READ>', '<UN_READ>',
    '<CN_MED>', '<UN_MED>',
    '<CN_MATH>', '<UN_MATH>',
]


# ============================================================================
# Dataset Configurations
# ============================================================================

DATASET_CONFIGS = {
    'math': {
        'hf_path': 'huyxdang/math-split',
        'question_field': 'problem',
        'answer_field': 'solution',
        'domain': 'MATH',
        'max_tokens': 1024,
    },
    'medqa': {
        'hf_path': 'huyxdang/medqa-split',
        'question_field': 'question',
        'answer_field': 'answer_idx',  # Note: sometimes 'answer'
        'domain': 'MED',
        'max_tokens': 512,
        'has_options': True,
        'options_field': 'options'
    },
    'boolq': {
        'hf_path': 'huyxdang/boolq-split',
        'question_field': 'question',
        'answer_field': 'answer',
        'domain': 'READ',
        'max_tokens': 512,
        'has_passage': True,
        'passage_field': 'passage'
    }
}


# ============================================================================
# Helper Functions
# ============================================================================

def get_confidence_token(dataset_name: str, is_correct: bool) -> str:
    """
    Get confidence token for a dataset and correctness.
    
    Args:
        dataset_name: Name of dataset ('medqa', 'boolq', 'math')
        is_correct: Whether the prediction is correct
    
    Returns:
        Confidence token string (e.g., '<C_MED>', '<U_READ>')
    """
    if dataset_name not in CONFIDENCE_TOKENS:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    token_type = 'correct' if is_correct else 'incorrect'
    return CONFIDENCE_TOKENS[dataset_name][token_type]


def get_judge_token(domain: str, is_correct: bool) -> str:
    """
    Get judge token for a domain and correctness (alternative format).
    
    Args:
        domain: Domain name ('MED', 'READ', 'MATH')
        is_correct: Whether the prediction is correct
    
    Returns:
        Judge token string (e.g., '<CN_MED>', '<UN_READ>')
    """
    if domain not in JUDGE_TOKENS:
        raise ValueError(f"Unknown domain: {domain}")
    
    token_type = 'correct' if is_correct else 'incorrect'
    return JUDGE_TOKENS[domain][token_type]


def get_dataset_config(dataset_name: str) -> dict:
    """Get configuration for a dataset."""
    if dataset_name not in DATASET_CONFIGS:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    return DATASET_CONFIGS[dataset_name]

