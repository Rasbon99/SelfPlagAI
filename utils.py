# Standard library
import io
import json
import random
import sys
import shutil
import os
import re

# Third-party libraries - data processing
import numpy as np
from datasets import Dataset, DatasetDict
from tqdm import tqdm

# Third-party libraries - deep learning / transformers
import torch
from accelerate import Accelerator
from bert_score import score
from transformers import DataCollatorForLanguageModeling, Trainer, TrainerCallback

#Define a function to format the dataset examples into a prompt taht will include the context, question, and answer
def make_prompt(example):
    context = example["context"]
    question = example["question"]
    answer = example["answers"]["text"][0] if example["answers"]["text"] else "No answer"

    prompt = f"[INST] Given the context, answer the question. If you do not find the answer in the context, answer with \"No answer\"\n\nContext: {context}\n\nQuestion: {question} [/INST] Answer: {answer}"
    return {"prompt": prompt, "reference": answer}
    
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

# Define tokenization function based on data structure
def tokenize(example, tokenizer):
    # Check if example has 'prompt' key (formatted data) or needs to be created
    if 'prompt' in example:
        text_to_tokenize = example["prompt"]
    else:
        # Create prompt from raw data using make_prompt logic
        context = example["context"]
        question = example["question"]
        
        # Handle different answer formats
        if "answers" in example:
            answers = example["answers"]
            # Check if answers is a dict with 'text' key (original format)
            if isinstance(answers, dict) and "text" in answers:
                answer = answers["text"][0] if answers["text"] else "No answer"
            # Check if answers is a list (some synthetic data format)
            elif isinstance(answers, list) and len(answers) > 0:
                # If it's a list of strings
                if isinstance(answers[0], str):
                    answer = answers[0]
                # If it's a list of dicts with 'text' key
                elif isinstance(answers[0], dict) and "text" in answers[0]:
                    answer = answers[0]["text"]
                else:
                    answer = "No answer"
            else:
                answer = "No answer"
        else:
            # Fallback: check if there's a 'reference' field
            answer = example.get("reference", "No answer")
        
        text_to_tokenize = f"[INST] Given the context, answer the question.\n\nContext: {context}\n\nQuestion: {question} [/INST] Answer: {answer}"
    
    return tokenizer(
        text_to_tokenize,
        truncation=True,
        padding="max_length",
        max_length=512
    )
 

# Fixed Custom Trainer class with BERTScore loss
class BERTScoreTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.label_names = ["labels"]
        
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """
        Custom loss function using BERTScore - completely silent
        Extracts only the answer part after "Answer: " for BERTScore computation
        """
        labels = inputs.get("labels")
        
        # Temporarily disable cache for forward pass
        model.config.use_cache = False
        
        # Forward pass
        outputs = model(**inputs)
        
        # Generate predictions for BERTScore
        with torch.no_grad():
            input_ids = inputs["input_ids"]
            attention_mask = inputs["attention_mask"]
            
            # Re-enable cache for generation
            model.config.use_cache = True
            
            # Generate text
            try:
                generated = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=50,
                    do_sample=False,
                    pad_token_id=self.processing_class.eos_token_id,
                    use_cache=True
                )
                
                # Decode predictions and references
                pred_texts = self.processing_class.batch_decode(generated, skip_special_tokens=True)
                ref_texts = self.processing_class.batch_decode(labels, skip_special_tokens=True)
                
                # Extract only the answer part from both predictions and references
                extracted_preds = []
                extracted_refs = []
                
                for pred_text in pred_texts:
                    # Extract answer after "Answer: " in prediction
                    if "Answer: " in pred_text:
                        answer_part = pred_text.split("Answer: ")[-1].strip()
                    elif "[/INST]" in pred_text:
                        # Fallback: extract after [/INST] if "Answer: " not found
                        answer_part = pred_text.split("[/INST]")[-1].strip()
                        # Remove "Answer: " prefix if it exists
                        if answer_part.startswith("Answer: "):
                            answer_part = answer_part[8:].strip()
                    else:
                        answer_part = pred_text.strip()
                    
                    extracted_preds.append(answer_part)
                
                for ref_text in ref_texts:
                    # Extract answer after "Answer: " in reference
                    if "Answer: " in ref_text:
                        answer_part = ref_text.split("Answer: ")[-1].strip()
                    elif "[/INST]" in ref_text:
                        # Fallback: extract after [/INST] if "Answer: " not found
                        answer_part = ref_text.split("[/INST]")[-1].strip()
                        # Remove "Answer: " prefix if it exists
                        if answer_part.startswith("Answer: "):
                            answer_part = answer_part[8:].strip()
                    else:
                        answer_part = ref_text.strip()
                    
                    extracted_refs.append(answer_part)
                
                # Calculate BERTScore with completely silent function on extracted answers only
                P, R, F1 = silent_bert_score(extracted_preds, extracted_refs, lang="en")
                bert_f1 = F1.mean().item()
                
                # Convert BERTScore to loss
                bert_loss = torch.tensor(1.0 - bert_f1, requires_grad=True, device=input_ids.device)
            except Exception as e:
                # Fallback to standard loss if BERTScore fails
                bert_loss = outputs.loss
            finally:
                # Disable cache again for gradient checkpointing compatibility
                model.config.use_cache = False
        
        # Combine with standard language modeling loss
        standard_loss = outputs.loss
        combined_loss = 0.7 * standard_loss + 0.3 * bert_loss
        
        return (combined_loss, outputs) if return_outputs else combined_loss

# Data preparation function
def prepare_training_data(tokenized_dataset):
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

def cleanup_checkpoints(output_dir):
    """Remove checkpoint directories and files"""
    if os.path.exists(output_dir):
        # Find all checkpoint directories
        checkpoint_dirs = [d for d in os.listdir(output_dir) if d.startswith('checkpoint-')]
        
        for checkpoint_dir in checkpoint_dirs:
            checkpoint_path = os.path.join(output_dir, checkpoint_dir)
            if os.path.isdir(checkpoint_path):
                print(f"🗑️ Removing checkpoint: {checkpoint_path}")
                shutil.rmtree(checkpoint_path)
        
        # Remove any other checkpoint-related files
        checkpoint_files = [f for f in os.listdir(output_dir) if 'checkpoint' in f.lower()]
        for checkpoint_file in checkpoint_files:
            file_path = os.path.join(output_dir, checkpoint_file)
            if os.path.isfile(file_path):
                print(f"🗑️ Removing checkpoint file: {file_path}")
                os.remove(file_path)
        
        print("✅ Checkpoint cleanup completed!")

def configure_model_for_training(model):
    """Configure model for training with proper cache settings"""
    
    # Disable use_cache for training compatibility with gradient checkpointing
    if hasattr(model.config, 'use_cache'):
        model.config.use_cache = False
        print("✅ Set use_cache=False for gradient checkpointing compatibility")
    
    # Enable gradient checkpointing if available
    if hasattr(model, 'gradient_checkpointing_enable'):
        model.gradient_checkpointing_enable()
        print("✅ Enabled gradient checkpointing for memory efficiency")
    
    return model

# Main training function
def train_model(model, tokenized_data, tokenizer, train_args):
    """Training function with BERTScore and early stopping"""
    
    # Configure model for training
    model = configure_model_for_training(model)
    
    # Prepare data
    prepared_data = prepare_training_data(tokenized_data)
    
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
    print("Cache disabled for gradient checkpointing compatibility")
    
    trainer.train()
    
    # Re-enable cache for inference after training
    if hasattr(model.config, 'use_cache'):
        model.config.use_cache = True
        print("✅ Re-enabled use_cache for inference")
    
    # Save model
    final_model_path = "./phi3-squad2-final"
    trainer.save_model(final_model_path)
    print(f"✅ Model saved to {final_model_path}")
    
    # Clean up checkpoints after saving the final model
    print("\n🧹 Cleaning up checkpoints...")
    if hasattr(train_args, 'output_dir') and train_args.output_dir:
        cleanup_checkpoints(train_args.output_dir)
    
    # Also clean up from the final model directory if it has checkpoints
    cleanup_checkpoints(final_model_path)
    
    # Clean up any checkpoint directories in the current working directory
    current_dir_checkpoints = [d for d in os.listdir('.') if d.startswith('checkpoint-')]
    for checkpoint_dir in current_dir_checkpoints:
        if os.path.isdir(checkpoint_dir):
            print(f"🗑️ Removing checkpoint: {checkpoint_dir}")
            shutil.rmtree(checkpoint_dir)
    
    print("🎉 Training completed and checkpoints cleaned up!")
    
    return trainer

def generate_synthetic_answers(model, tokenizer, formatted_dataset, device, generation_num=1):
    """
    Generate synthetic answers using the fine-tuned causal language model.
    Takes formatted_dataset with prompts as input.
    Updated to handle "Answer: " format and write "No answer" when appropriate.
    """
    
    print(f"Generating synthetic answers (Generation {generation_num})...")
    
    synthetic_data = []
    
    # Use the train split from formatted_dataset
    train_dataset = formatted_dataset['train']
    
    # Enhanced progress bar with statistics
    progress_bar = tqdm(
        train_dataset, 
        desc=f"🤖 Gen {generation_num} - Generating answers",
        unit="examples",
        position=1,
        leave=False,
        dynamic_ncols=True
    )
    
    # Statistics tracking
    successful_generations = 0
    failed_generations = 0
    total_examples = len(train_dataset)
    
    for idx, example in enumerate(progress_bar):
        # Get the prompt that was created by make_prompt function
        prompt = example['prompt']
        
        # For your format: "[INST] ... [/INST] Answer: {answer}"
        # We need to generate starting from just before "Answer:"
        if '[/INST] Answer:' in prompt:
            # Remove the existing answer to create generation prompt
            generation_prompt = prompt.split(' Answer:')[0] + ' Answer:'
        elif '[/INST]' in prompt:
            # Fallback: if no "Answer:" found, add it
            generation_prompt = prompt.split('[/INST]')[0] + '[/INST] Answer:'
        else:
            # Fallback: use the full prompt
            generation_prompt = prompt
        
        try:
            # Tokenize input - increased max_length to handle full context
            inputs = tokenizer(
                generation_prompt,
                max_length=512,  # Increased from 50 to handle full context
                truncation=True,
                padding=True,
                return_tensors="pt"
            ).to(device)
            
            # Generate answer using causal LM
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=50,   # Reduced for concise answers
                    do_sample=True,      # Keep diversity
                    temperature=0.7,     # Controlled randomness
                    top_p=0.9,          # Nucleus sampling
                    pad_token_id=tokenizer.eos_token_id,
                    eos_token_id=tokenizer.eos_token_id
                )
            
            # Decode the generated text
            generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract only the answer part after "Answer: "
            if 'Answer:' in generated_text:
                # Split by "Answer:" and take the last part (the generated answer)
                answer_parts = generated_text.split('Answer:')
                if len(answer_parts) > 1:
                    synthetic_answer = answer_parts[-1].strip()
                else:
                    synthetic_answer = "No answer"
            else:
                # Fallback: extract after [/INST] if "Answer:" not found
                if '[/INST]' in generated_text:
                    synthetic_answer = generated_text.split('[/INST]')[-1].strip()
                    # Remove "Answer:" prefix if it exists
                    if synthetic_answer.startswith('Answer:'):
                        synthetic_answer = synthetic_answer[7:].strip()
                else:
                    synthetic_answer = "No answer"
            
            # Additional cleanup and validation
            if synthetic_answer:
                # Stop at first sentence if answer is too long
                sentences = synthetic_answer.split('.')
                if len(sentences) > 1 and len(sentences[0]) > 5:
                    synthetic_answer = sentences[0].strip()
                
                # Remove any remaining formatting artifacts
                synthetic_answer = synthetic_answer.replace('\n', ' ').strip()
                
                # Check if answer is reasonable (not empty, not too long)
                if len(synthetic_answer) < 2 or len(synthetic_answer) > 200:
                    synthetic_answer = "No answer"
                    failed_generations += 1
                else:
                    # Check if the answer exists in the context (if context is available)
                    context = example.get('context', '')
                    if context and synthetic_answer.lower() not in context.lower():
                        # Answer not found in context, mark as "No answer"
                        synthetic_answer = "No answer"
                        failed_generations += 1
                    else:
                        successful_generations += 1
            else:
                synthetic_answer = "No answer"
                failed_generations += 1
            
        except Exception as e:
            # Handle any generation errors
            synthetic_answer = "No answer"
            failed_generations += 1
            print(f"\nWarning: Generation failed for example {idx}: {str(e)}")
        
        # Create new example with synthetic answer
        new_example = example.copy()
        
        # Update the prompt to include the generated answer in correct format
        # Format: "[INST] ... [/INST] Answer: {synthetic_answer}"
        if '[/INST]' in prompt:
            base_prompt = prompt.split('[/INST]')[0] + '[/INST] Answer:'
            new_example['prompt'] = base_prompt + ' ' + synthetic_answer
        else:
            new_example['prompt'] = prompt + ' ' + synthetic_answer
        
        # Update the reference to the synthetic answer
        new_example['reference'] = synthetic_answer
        
        # If original data has structured fields, preserve them and update answers
        if 'answers' in example:
            if synthetic_answer != "No answer":
                # Try to find answer in context if context exists
                context = example.get('context', '')
                answer_start = context.find(synthetic_answer) if context else 0
                if answer_start == -1:
                    answer_start = 0
                
                new_example['answers'] = {
                    'text': [synthetic_answer],
                    'answer_start': [answer_start]
                }
            else:
                # No answer case - follow SQuAD v2 format
                new_example['answers'] = {
                    'text': [],
                    'answer_start': []
                }
        
        # Add generation metadata
        new_example['generation_num'] = generation_num
        new_example['synthetic'] = True
        
        synthetic_data.append(new_example)
        
        # Update progress bar with statistics
        success_rate = (successful_generations / (idx + 1)) * 100
        progress_bar.set_postfix({
            'Success': f'{successful_generations}/{idx + 1}',
            'Rate': f'{success_rate:.1f}%',
            'Failed': failed_generations,
            'No Answer': failed_generations
        })
        
        # Update description every 100 examples
        if (idx + 1) % 100 == 0:
            progress_bar.set_description(
                f"🤖 Gen {generation_num} - Generated {idx + 1}/{total_examples}"
            )
    
    # Close progress bar
    progress_bar.close()
    
    # Print final statistics
    print(f"✅ Generation {generation_num} completed:")
    print(f"   📊 Total examples processed: {total_examples}")
    print(f"   ✅ Successful generations: {successful_generations}")
    print(f"   ❌ No answer cases: {failed_generations}")
    print(f"   📈 Success rate: {(successful_generations/total_examples)*100:.1f}%")
    print(f"   📈 No answer rate: {(failed_generations/total_examples)*100:.1f}%")
    
    # Create a new formatted dataset with the synthetic data
    print("📦 Creating synthetic dataset...")
    with tqdm(total=1, desc="📦 Building Dataset", position=1, leave=False) as dataset_pbar:
        synthetic_dataset = Dataset.from_list(synthetic_data)
        dataset_pbar.update(1)
    
    # Return in the same format as input
    return {
        'train': synthetic_dataset
    }


def save_synthetic_dataset(dataset_dir, synthetic_formatted_dataset, generation_num):
    """
    Save the synthetic formatted dataset to a new subdirectory.
    
    Args:
        dataset_dir: Base dataset directory
        synthetic_formatted_dataset: Formatted dataset with synthetic answers
        generation_num: Generation number
    """
    
    # Create new subdirectory
    new_dir = os.path.join(dataset_dir, f"generation_{generation_num}")
    os.makedirs(new_dir, exist_ok=True)
    
    # Save the train split
    synthetic_formatted_dataset['train'].save_to_disk(os.path.join(new_dir, "train"))
    
    # Also save as JSON for inspection
    json_file = os.path.join(new_dir, "synthetic_data.json")
    synthetic_formatted_dataset['train'].to_json(json_file)
    
    # Save metadata
    metadata = {
        "generation_number": generation_num,
        "total_examples": len(synthetic_formatted_dataset['train']),
        "generated_from": "fine_tuned_model",
        "description": f"Synthetic answers generated using fine-tuned model (Generation {generation_num})",
        "format": "formatted_dataset_with_prompts"
    }
    
    with open(os.path.join(new_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Saved synthetic formatted dataset to {new_dir}")
    return new_dir

def extract_model_nick(model_path):
    # Extract the part after the first "/"
    model_name = model_path.split("/")[1]
    
    # Match common patterns like "phi-3" or "Mistral-7B"
    match = re.match(r"([A-Za-z0-9\-]+?)(?=-\d|-[a-zA-Z])", model_name)
    
    return match.group(1) if match else model_name
    