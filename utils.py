# Standard library
import io
import json
import random
import sys

# Third-party libraries - data processing
import numpy as np
from datasets import Dataset
from tqdm import tqdm

# Third-party libraries - deep learning / transformers
import torch
from bert_score import score
from transformers import DataCollatorForLanguageModeling, Trainer, TrainerCallback

#Define a function to format the dataset examples into a prompt
#The prompt will include the context, question, and answer
def make_prompt(example):
    context = example["context"]
    question = example["question"]
    answer = example["answers"]["text"][0] if example["answers"]["text"] else "No answer"

    prompt = f"[INST] Given the context, answer the question.\n\nContext: {context}\n\nQuestion: {question} [/INST] {answer}"
    return {"prompt": prompt, "reference": answer}
    
def tokenize(example, tokenizer):
    return tokenizer(
        example["prompt"],
        truncation=True,
        padding="max_length",
        max_length=256
    )
    
# Modified BERTScore function with complete output suppression
def silent_bert_score(cands, refs, lang="en"):
    """BERTScore calculation with all output suppressed"""
    old_stdout = sys.stdout
    old_stderr = sys.stderr

    sys.stdout = io.StringIO()
    sys.stderr = io.StringIO()

    try:
        P, R, F1 = score(cands, refs, lang=lang, verbose=False)
        return P, R, F1
    finally:
        sys.stdout = old_stdout
        sys.stderr = old_stderr

# Custom Early Stopping based on Training Loss
class TrainingLossEarlyStoppingCallback(TrainerCallback):
    def __init__(self, patience=3, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float('inf')
        self.wait_count = 0

    def on_log(self, args, state, control, model=None, logs=None, **kwargs):
        if logs is not None and 'train_loss' in logs:
            current_loss = logs['train_loss']

            if current_loss < self.best_loss - self.min_delta:
                self.best_loss = current_loss
                self.wait_count = 0
                print(f"📈 Training loss improved to {current_loss:.4f}")
            else:
                self.wait_count += 1
                print(f"📊 No improvement in training loss ({self.wait_count}/{self.patience})")

                if self.wait_count >= self.patience:
                    print(f"🛑 Early stopping triggered! Best loss: {self.best_loss:.4f}")
                    control.should_training_stop = True

# Fixed Custom Trainer class with BERTScore loss
class BERTScoreTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """
        Custom loss function using BERTScore - completely silent
        """
        labels = inputs.get("labels")

        # Forward pass
        outputs = model(**inputs)

        # Generate predictions for BERTScore
        with torch.no_grad():
            input_ids = inputs["input_ids"]
            attention_mask = inputs["attention_mask"]

            # Generate text
            try:
                generated = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=50,
                    do_sample=False,
                    pad_token_id=self.processing_class.eos_token_id
                )

                # Decode predictions and references
                pred_texts = self.processing_class.batch_decode(generated, skip_special_tokens=True)
                ref_texts = self.processing_class.batch_decode(labels, skip_special_tokens=True)

                # Calculate BERTScore with completely silent function
                P, R, F1 = silent_bert_score(pred_texts, ref_texts, lang="en")
                bert_f1 = F1.mean().item()

                # Convert BERTScore to loss
                bert_loss = torch.tensor(1.0 - bert_f1, requires_grad=True, device=input_ids.device)
            except Exception as e:
                # Fallback to standard loss if BERTScore fails
                bert_loss = outputs.loss

        # Combine with standard language modeling loss
        standard_loss = outputs.loss
        combined_loss = 0.7 * standard_loss + 0.3 * bert_loss

        return (combined_loss, outputs) if return_outputs else combined_loss

# Data preparation function
def prepare_training_data(tokenized_dataset, tokenizer):
    """Prepare data for training"""

    def add_labels(example):
        example["labels"] = example["input_ids"].copy()
        return example

    # Only prepare train split
    train_dataset = tokenized_dataset["train"].map(add_labels)

    # Keep only necessary columns
    keep_keys = ["input_ids", "attention_mask", "labels"]
    train_dataset = train_dataset.remove_columns(
        [col for col in train_dataset.column_names if col not in keep_keys]
    )

    return {"train": train_dataset}

# Main training function
def train_model(model, tokenized_data, tokenizer, train_args):
    """Training function with BERTScore and early stopping"""

    # Prepare data
    prepared_data = prepare_training_data(tokenized_data, tokenizer)

    # Setup data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
        return_tensors="pt"
    )

    # Custom early stopping based on training loss
    early_stopping_callback = TrainingLossEarlyStoppingCallback(
        patience=5,
        min_delta=0.01
    )

    # Initialize BERTScore Trainer
    trainer = BERTScoreTrainer(
        model=model,
        args=train_args,
        train_dataset=prepared_data["train"],
        data_collator=data_collator,
        processing_class=tokenizer,
        callbacks=[early_stopping_callback],
    )

    # Start training
    print("Starting training with BERTScore optimization...")
    print("Early stopping based on training loss improvement")

    trainer.train()

    # Save model
    trainer.save_model("./phi3-squad2-final")
    print("✅ Model saved to ./phi3-squad2-final")

    return trainer

def evaluate_test_set_with_examples(model, tokenizer, dataset, make_prompt_func, num_examples=10):
    """
    Comprehensive evaluation on test set with detailed prediction examples
    """
    print("="*60)
    print("STARTING TEST SET EVALUATION WITH EXAMPLES")
    print("="*60)

    # Prepare test data
    print("Preparing test prompts...")
    test_prompts = dataset["test"].map(make_prompt_func)

    # Tokenize test prompts
    print("Tokenizing test data...")
    tokenized_test = test_prompts.map(
        lambda x: tokenizer(x["prompt"], truncation=True, padding="max_length", max_length=512),
        batched=True
    )

    # Set model to evaluation mode
    model.eval()

    # Initialize lists for predictions and references
    preds = []
    refs = []
    raw_outputs = []
    prompts_list = []

    print(f"Generating predictions for {len(test_prompts)} test examples...")

    # Generate predictions
    for example in tqdm(test_prompts, desc="Evaluating"):
        # Tokenize input
        inputs = tokenizer(
            example["prompt"],
            return_tensors="pt",
            truncation=True,
            padding="max_length",
            max_length=512
        ).to(model.device)

        # Generate response
        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=50,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )

        # Decode output
        decoded = tokenizer.decode(output[0], skip_special_tokens=True)

        # Extract answer (everything after [/INST])
        if '[/INST]' in decoded:
            answer = decoded.split('[/INST]')[-1].strip()
        else:
            answer = decoded.strip()

        # Store results
        preds.append(answer)
        refs.append(example.get("reference", example.get("answer", "No answer")))
        raw_outputs.append(decoded)
        prompts_list.append(example["prompt"])

    print("Predictions generated! Computing metrics...")

    # Compute BERTScore
    print("Computing BERTScore...")
    try:
        P, R, F1 = score(preds, refs, lang="en", verbose=False)
        bert_scores = {
            "precision": P.mean().item(),
            "recall": R.mean().item(),
            "f1": F1.mean().item()
        }
    except Exception as e:
        print(f"BERTScore computation failed: {e}")
        bert_scores = {"precision": 0.0, "recall": 0.0, "f1": 0.0}
        P = R = F1 = [0.0] * len(preds)

    # Compute exact match accuracy
    exact_matches = []
    for pred, ref in zip(preds, refs):
        if ref.lower().strip() in pred.lower().strip():
            exact_matches.append(1)
        else:
            exact_matches.append(0)

    exact_match_score = np.mean(exact_matches)

    # Compute answer length statistics
    pred_lengths = [len(pred.split()) for pred in preds]
    ref_lengths = [len(ref.split()) for ref in refs]

    # Print comprehensive results
    print("="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    print(f"Test Set Size: {len(preds)}")
    print("-"*60)
    print("BERTScore Metrics:")
    print(f"  Precision: {bert_scores['precision']:.4f}")
    print(f"  Recall:    {bert_scores['recall']:.4f}")
    print(f"  F1 Score:  {bert_scores['f1']:.4f}")
    print("-"*60)
    print("Other Metrics:")
    print(f"  Exact Match: {exact_match_score:.4f}")
    print("-"*60)
    print("Answer Length Statistics:")
    print(f"  Avg Prediction Length: {np.mean(pred_lengths):.2f} words")
    print(f"  Avg Reference Length:  {np.mean(ref_lengths):.2f} words")
    print("="*60)

    # Show detailed examples
    print("\n" + "="*80)
    print("DETAILED PREDICTION EXAMPLES")
    print("="*80)

    # Select diverse examples: best, worst, and random
    f1_scores = [f.item() if hasattr(f, 'item') else f for f in F1]

    # Get indices for different categories
    sorted_indices = sorted(range(len(f1_scores)), key=lambda i: f1_scores[i], reverse=True)

    best_indices = sorted_indices[:3]  # Top 3
    worst_indices = sorted_indices[-3:]  # Bottom 3
    random_indices = random.sample(range(len(preds)), min(4, len(preds)))  # Random 4

    example_categories = [
        ("BEST PREDICTIONS", best_indices),
        ("WORST PREDICTIONS", worst_indices),
        ("RANDOM PREDICTIONS", random_indices)
    ]

    for category_name, indices in example_categories:
        print(f"\n{category_name}:")
        print("-" * 80)

        for i, idx in enumerate(indices):
            print(f"\nExample {i+1} (Index {idx}):")
            print(f"BERTScore F1: {f1_scores[idx]:.4f}")
            print(f"Exact Match: {'✓' if exact_matches[idx] else '✗'}")

            # Extract question from prompt
            prompt = prompts_list[idx]
            if "Question:" in prompt:
                question = prompt.split("Question:")[-1].split("Answer:")[0].strip()
                print(f"Question: {question}")
            else:
                print(f"Prompt: {prompt[:200]}...")

            print(f"Reference Answer: {refs[idx]}")
            print(f"Model Prediction: {preds[idx]}")

            # Analysis
            pred_words = len(preds[idx].split())
            ref_words = len(refs[idx].split())
            print(f"Length: Pred={pred_words} words, Ref={ref_words} words")

            # Simple similarity check
            pred_lower = preds[idx].lower()
            ref_lower = refs[idx].lower()
            common_words = set(pred_lower.split()) & set(ref_lower.split())
            print(f"Common words: {len(common_words)}")

            print("-" * 50)

    # Show questions by category if available
    if "category" in test_prompts.column_names or "topic" in test_prompts.column_names:
        print(f"\n{'='*80}")
        print("PERFORMANCE BY CATEGORY")
        print("="*80)

        category_field = "category" if "category" in test_prompts.column_names else "topic"
        categories = {}

        for i, example in enumerate(test_prompts):
            cat = example.get(category_field, "Unknown")
            if cat not in categories:
                categories[cat] = {"f1_scores": [], "exact_matches": [], "indices": []}
            categories[cat]["f1_scores"].append(f1_scores[i])
            categories[cat]["exact_matches"].append(exact_matches[i])
            categories[cat]["indices"].append(i)

        for cat, data in categories.items():
            avg_f1 = np.mean(data["f1_scores"])
            avg_em = np.mean(data["exact_matches"])
            count = len(data["f1_scores"])
            print(f"{cat}: F1={avg_f1:.4f}, EM={avg_em:.4f}, Count={count}")

            # Show one example from each category
            best_idx_in_cat = data["indices"][np.argmax(data["f1_scores"])]
            print(f"  Best example: {prompts_list[best_idx_in_cat][:100]}...")
            print(f"  Prediction: {preds[best_idx_in_cat]}")
            print()

    # Create results dictionary
    results = {
        "test_size": len(preds),
        "bert_score": bert_scores,
        "exact_match": exact_match_score,
        "avg_prediction_length": np.mean(pred_lengths),
        "avg_reference_length": np.mean(ref_lengths),
        "predictions": preds,
        "references": refs,
        "prompts": prompts_list,
        "individual_scores": {
            "bert_f1": f1_scores,
            "exact_match": exact_matches
        }
    }

    # Save detailed results
    print(f"\n{'='*60}")
    print("SAVING RESULTS")
    print("="*60)

    # Save main results
    with open("test_evaluation_results.json", "w") as f:
        json.dump({k: v for k, v in results.items() if k != "prompts"}, f, indent=2)

    # Save detailed examples
    with open("detailed_predictions.txt", "w", encoding="utf-8") as f:
        f.write("DETAILED TEST SET PREDICTIONS\n")
        f.write("="*80 + "\n\n")

        for i, (prompt, pred, ref, f1_score, em) in enumerate(zip(prompts_list, preds, refs, f1_scores, exact_matches)):
            f.write(f"Example {i+1}:\n")
            f.write(f"BERTScore F1: {f1_score:.4f}\n")
            f.write(f"Exact Match: {'✓' if em else '✗'}\n")

            if "Question:" in prompt:
                question = prompt.split("Question:")[-1].split("Answer:")[0].strip()
                f.write(f"Question: {question}\n")
            else:
                f.write(f"Prompt: {prompt}\n")

            f.write(f"Reference: {ref}\n")
            f.write(f"Prediction: {pred}\n")
            f.write("-" * 50 + "\n\n")

    print("Results saved to:")
    print("  - test_evaluation_results.json")
    print("  - detailed_predictions.txt")
    print("="*60)

    return results


def generate_new_dataset(prompts, predictions):
    records = []
    for prompt, pred in zip(prompts, predictions):
        parts = prompt.split("Context:")[-1].split("Question:")
        context = parts[0].strip()
        question = parts[1].split("[/INST]")[0].strip()
        records.append({
            "context": context,
            "question": question,
            "answers": {"text": [pred]}
        })
    return Dataset.from_list(records)


def generate_dataset_from_model(model, tokenizer, dataset, max_examples=None):
    formatted = dataset.map(make_prompt)
    if max_examples is not None:
        formatted = formatted.select(range(max_examples))

    model.eval()
    preds = []
    prompts_list = []

    print(f"Generating synthetic answers for {len(formatted)} examples...")
    for example in formatted:
        inputs = tokenizer(
            example["prompt"],
            return_tensors="pt",
            truncation=True,
            padding="max_length",
            max_length=512
        ).to(model.device)

        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=50,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )

        decoded = tokenizer.decode(output[0], skip_special_tokens=True)
        if '[/INST]' in decoded:
            answer = decoded.split('[/INST]')[-1].strip()
        else:
            answer = decoded.strip()

        preds.append(answer)
        prompts_list.append(example["prompt"])

    new_dataset = generate_new_dataset(prompts_list, preds)
    return new_dataset