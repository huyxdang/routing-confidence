import json
import torch
import numpy as np
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    TrainerCallback,
)
from peft import LoraConfig, get_peft_model
import random




# ===============================
# Config
# ===============================

MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"
TRAIN_PATH = "merged_data/full_train_data.json"
OUTPUT_DIR = "./conf_sft_qwen3b"

SPECIAL_TOKENS = ["<C_READ>", "<U_READ>", "<C_MED>", "<U_MED>"]

# ===============================
# Load dataset
# ===============================

import json
from datasets import Dataset

print("Loading dataset...")

# Load the JSON dictionary
with open(TRAIN_PATH, 'r') as f:
    data_dict = json.load(f)

print(f"Loaded {len(data_dict)} raw samples.")

# Convert to list, keeping ONLY the fields we need
data_list = []
for key, value in data_dict.items():
    # Extract only the fields needed for training
    clean_sample = {
        'question': value['question'],
        'tagged_response': value['tagged_response'],
        'correct': bool(value['correct']),  # Ensure it's a boolean
        'dataset': value['dataset']
    }
    
    data_list.append(clean_sample)

# Create HuggingFace Dataset
ds = Dataset.from_list(data_list)

print(f"Created dataset with {len(ds)} samples.")
print(f"Columns: {ds.column_names}")

# Show domain distribution
domains = {}
for item in data_list:
    domain = item['dataset']
    domains[domain] = domains.get(domain, 0) + 1

print(f"\nDomain distribution:")
for domain, count in domains.items():
    print(f"  - {domain}: {count}")

# Verify required fields
required_fields = ['question', 'tagged_response', 'correct', 'dataset']
missing = [f for f in required_fields if f not in ds.column_names]

if missing:
    raise ValueError(f"Dataset missing required fields: {missing}")

print("\n✅ Dataset loaded successfully!")
print(f"\nFirst sample:")
print(f"  Domain: {ds[0]['dataset']}")
print(f"  Question: {ds[0]['question'][:100]}...")
print(f"  Tagged response ends with: ...{ds[0]['tagged_response'][-50:]}")
print(f"  Correct: {ds[0]['correct']}")


# ===============================
# Calibration Error Function
# ===============================

def calibration_error(probs, correct, beta=100, p="2"):
    probs = np.array(probs)
    correct = np.array(correct)

    idxs = np.argsort(probs)
    probs = probs[idxs]
    correct = correct[idxs]

    n = len(probs)
    bins = [(i, min(i+beta, n)) for i in range(0, n, beta)]

    errors, weights = [], []

    for start, end in bins:
        p_bin = probs[start:end]
        c_bin = correct[start:end]
        if len(p_bin) == 0:
            continue

        diff = abs(p_bin.mean() - c_bin.mean())
        errors.append(diff)
        weights.append(len(p_bin) / n)

    if p == "1":
        return sum(w * e for w, e in zip(weights, errors))

    if p == "2":
        return np.sqrt(sum(w * (e**2) for w, e in zip(weights, errors)))

    return max(errors) if errors else 0.0


# ===============================
# Quick Evaluation Callback
# ===============================

class QuickEvalCallback(TrainerCallback):
    def __init__(self, model, tokenizer, eval_data, every=400):
        self.model = model
        self.tokenizer = tokenizer
        self.eval_data = eval_data
        self.every = every

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step == 0 or state.global_step % self.every != 0:
            return

        print("\n===== QUICK EVAL =====")
        self.model.eval()

        ids = random.sample(range(len(self.eval_data)), 6)
        token_present = 0
        correct_domain = 0

        for idx in ids:
            e = self.eval_data[idx]
            domain = e.get("dataset", "").lower()

            prompt = f"Question: {e['question']}\n\nAnswer:"
            inp = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
            out = self.model.generate(**inp, max_new_tokens=80, do_sample=False)
            text = self.tokenizer.decode(out[0], skip_special_tokens=False)

            print("\n--- SAMPLE OUTPUT ---")
            print(text)

            found = None
            for tok in SPECIAL_TOKENS:
                if tok in text:
                    found = tok
                    token_present += 1
                    break

            # domain correctness
            if found:
                if domain == "boolq" and "READ" in found:
                    correct_domain += 1
                if domain == "medqa" and "MED" in found:
                    correct_domain += 1

        print(f"\nToken present: {token_present}/6")
        print(f"Correct domain token: {correct_domain}/6")
        print("======================\n")

        self.model.train()


# ===============================
# Load tokenizer & model
# ===============================

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

num_added = tokenizer.add_special_tokens({"additional_special_tokens": SPECIAL_TOKENS})
print(f"Added {num_added} special tokens:", SPECIAL_TOKENS)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    device_map="auto",
    torch_dtype="auto",
)

model.resize_token_embeddings(len(tokenizer))


# Init token embeddings
emb = model.get_input_embeddings().weight.data
avg_vec = emb[:emb.shape[0] - num_added].mean(dim=0)

for tok in SPECIAL_TOKENS:
    tok_id = tokenizer.convert_tokens_to_ids(tok)
    emb[tok_id] = avg_vec.clone()

print("Initialized special token embeddings.\n")


# ===============================
# Load dataset
# ===============================

ds = load_dataset("json", data_files=TRAIN_PATH, split="train")
print(f"Loaded {len(ds)} samples.")


# ===============================
# Preprocess
# ===============================

def format_example(e):
    text = (
        f"Question: {e['question']}\n\n"
        f"Answer: {e['tagged_response']}"
    )
    toks = tokenizer(text, truncation=True, max_length=2048)
    toks["labels"] = toks["input_ids"].copy()
    return toks

ds = ds.map(format_example)
ds = ds.shuffle(seed=42)


# ===============================
# Setup LoRA
# ===============================

lora = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules=["q_proj", "v_proj"],
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, lora)


# ===============================
# Validation function (full)
# ===============================

def full_validate(model, tokenizer, data, max_samples=300):
    print("\n===== FULL VALIDATION (CALIBRATION) =====")
    model.eval()
    
    conf_probs = []
    correctness = []
    token_matches = []
    
    for i in range(min(len(data), max_samples)):
        e = data[i]
        domain = e.get("dataset", "").lower()
        prompt = f"Question: {e['question']}\n\nAnswer:"
        
        # Generate
        inp = tokenizer(prompt, return_tensors="pt").to(model.device)
        out = model.generate(**inp, max_new_tokens=80, do_sample=False)
        gen_text = tokenizer.decode(out[0], skip_special_tokens=False)
        
        # Find confidence token
        predicted_tok = None
        for tok in SPECIAL_TOKENS:
            if tok in gen_text:
                predicted_tok = tok
                break
        
        if predicted_tok is None:
            continue
        
        # Domain correctness
        correct_domain = (
            ("READ" in predicted_tok and domain == "boolq") or
            ("MED" in predicted_tok and domain == "medqa")
        )
        token_matches.append(int(correct_domain))
        
        # Extract answer text
        answer_start = gen_text.find("Answer:") + len("Answer:")
        token_pos = gen_text.find(predicted_tok)
        answer_text = gen_text[answer_start:token_pos].strip()
        
        # Correct or wrong?
        is_correct = bool(e.get("correct", False))
        correctness.append(1 if is_correct else 0)
        
        # Build prefix for logits
        full_input = f"Question: {e['question']}\n\nAnswer: {answer_text}"
        toks = tokenizer(full_input, return_tensors="pt").to(model.device)
        
        # Compute logits
        with torch.no_grad():
            logits = model(**toks).logits[0, -1, :]
        
        # Domain-specific ids
        if domain == "boolq":
            C_id = tokenizer.convert_tokens_to_ids("<C_READ>")
            U_id = tokenizer.convert_tokens_to_ids("<U_READ>")
        else:
            C_id = tokenizer.convert_tokens_to_ids("<C_MED>")
            U_id = tokenizer.convert_tokens_to_ids("<U_MED>")
        
        c_logit = logits[C_id].item()
        u_logit = logits[U_id].item()
        
        # p(confident)
        p_conf = torch.softmax(torch.tensor([c_logit, u_logit]), dim=0)[0].item()
        conf_probs.append(p_conf)
    
    # Final metrics
    token_rate = np.mean(token_matches) if token_matches else 0
    cal_err = calibration_error(conf_probs, correctness, beta=50, p="2")
    
    print(f"Samples evaluated: {len(conf_probs)}/{max_samples}")
    print(f"Token Domain Accuracy: {token_rate*100:.2f}%")
    print(f"Calibration Error (L2): {cal_err:.4f}")
    print("==========================================\n")
    
    return token_rate, cal_err


# ===============================
# Training
# ===============================

args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=2,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    learning_rate=2e-5,
    warmup_ratio=0.03,
    logging_steps=10,
    save_steps=1000,
    save_total_limit=2,
    bf16=True,
    report_to="none",
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=ds,
    callbacks=[QuickEvalCallback(model, tokenizer, ds, every=400)]
)

trainer.train()

# Final validation
full_validate(model, tokenizer, ds, max_samples=500)


# ===============================
# Save model
# ===============================

model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print("Training complete!")