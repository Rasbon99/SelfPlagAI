# Standard library
import io
import json
import random
import sys

# Third‑party libraries — data processing
import numpy as np
from datasets import Dataset
from tqdm import tqdm

# Third‑party libraries — deep learning / Transformers
import torch
from bert_score import score
from transformers import DataCollatorForLanguageModeling, Trainer, TrainerCallback

# Local modules — database utilities
from db_utils import (
    get_mongo_client,
    insert_dataframe_to_mongo,
    drop_collection,
    read_collection,
)

def read_original_and_create_subset(
    client: MongoClient,
    original_coll: str,
    subset_coll: str,
    db_name: str = "squadv2",
    sample_frac: float = 0.005,
    random_state: int = 42
) -> pd.DataFrame:
    """
    Legge il dataset SQuAD originale da MongoDB, ne estrae un subset e lo salva in una nuova collezione.
    
    Args:
        client: istanza MongoClient autenticata.
        original_coll: nome della collezione di partenza (es. 'squadv2_original').
        subset_coll: nome della collezione di destinazione (es. 'squad_subset_0_5pct').
        db_name: nome del database MongoDB.
        sample_frac: frazione di esempi da campionare (default 0.005 → 0.5%).
        random_state: seme per riproducibilità del campionamento.
    
    Returns:
        sample_df: DataFrame contenente il subset estratto.
    """
    # 1) Leggo tutto da original_coll
    df = read_collection(client, original_coll, db_name=db_name, as_dataframe=True)
    
    # 2) Campiono la frazione desiderata
    sample_df = df.sample(frac=sample_frac, random_state=random_state).reset_index(drop=True)
    
    # 3) Svuoto (se esiste) e riscrivo subset_coll
    drop_collection(client, subset_coll, db_name=db_name)
    insert_dataframe_to_mongo(client, sample_df, subset_coll, db_name=db_name)
    
    print(f"Sottinsieme creato: {len(sample_df)} esempi in '{db_name}.{subset_coll}'")
    return sample_df


def load_dataset_from_mongo(
    client: MongoClient,
    train_coll: str,
    test_coll: str,
    db_name: str = "squadv2",
    projection: dict = {"_id": 0}
) -> DatasetDict:
    """
    Carica da MongoDB il train e il test come HuggingFace DatasetDict.
    
    Args:
        client: istanza MongoClient autenticata.
        train_coll: collezione con il train set.
        test_coll: collezione con il test set.
        db_name: nome del database MongoDB.
        projection: come proiettare i campi (per es. escludere _id).
    
    Returns:
        dataset: DatasetDict con split 'train' e 'test'.
    """
    # Leggi train e test in DataFrame
    train_df = read_collection(client, train_coll, db_name=db_name, as_dataframe=True, projection=projection)
    test_df  = read_collection(client, test_coll,  db_name=db_name, as_dataframe=True, projection=projection)
    
    # Converto in Hugging Face Dataset
    train_ds = Dataset.from_pandas(train_df)
    test_ds  = Dataset.from_pandas(test_df)
    
    return DatasetDict({"train": train_ds, "test": test_ds})

def make_prompt(example):
    context = example["context"]
    question = example["question"]
    answer = example["answers"]["text"][0] if example["answers"]["text"] else "No answer"

    prompt = f"[INST] Given the context, answer the question. If you do not find the answer in the context, answer with \"No answer\"\n\nContext: {context}\n\nQuestion: {question} [/INST] Answer: {answer}"
    return {"prompt": prompt, "reference": answer}


'''
TRAINING UTILITY FUNCTIONS
'''

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

# Data preparation function - removed tokenizer parameter since it's not used
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


'''
SYNTHETIC DATASET GENERATION
'''
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


def save_synthetic_dataset_to_mongo(
    client: MongoClient,
    synthetic_dataset: DatasetDict,
    gen_number: int,
    train_coll: str,
    metadata_coll: str = "synthetic_metadata",
    db_name: str = "squadv2"
) -> None:
    """
    Inserisce il train sintetico su MongoDB e ne salva i metadati.
    
    Args:
        client: istanza MongoClient autenticata.
        synthetic_dataset: DatasetDict prodotto dal modello, con split 'train'.
        gen_number: numero di generazione corrente.
        train_coll: collezione di destinazione per il train sintetico.
        metadata_coll: collezione per i metadata (default 'synthetic_metadata').
        db_name: nome del database MongoDB.
    """
    # 1) Estrai pandas.DataFrame dal Dataset HF
    train_df = synthetic_dataset["train"].to_pandas().reset_index(drop=True)
    
    # 2) Svuota e reinserisci
    drop_collection(client, train_coll, db_name=db_name)
    insert_dataframe_to_mongo(client, train_df, train_coll, db_name=db_name)
    
    # 3) Prepara e inserisci metadata
    metadata = {
        "generation_number":     gen_number,
        "total_examples":        len(train_df),
        "timestamp":             pd.Timestamp.now().isoformat(),
        "description":           f"Synthetic train generation {gen_number}"
    }
    # Upsert basato su generation_number
    coll_meta = client[db_name][metadata_coll]
    coll_meta.update_one(
        {"generation_number": gen_number},
        {"$set": metadata},
        upsert=True
    )
    print(f"Train sintetico e metadata (gen {gen_number}) salvati su MongoDB.")


def extract_model_nick(model_path):
    # Extract the part after the first "/"
    model_name = model_path.split("/")[1]
    
    # Match common patterns like "phi-3" or "Mistral-7B"
    match = re.match(r"([A-Za-z0-9\-]+?)(?=-\d|-[a-zA-Z])", model_name)
    
    return match.group(1) if match else model_name

'''
ITERATIVE TRAINING
'''
def iterative_training_and_generation(
    base_dataset_dir="squad_v2_01percent",
    model_path="microsoft/Phi-3-mini-128k-instruct",
    num_generations=3,
    start_generation=1,
    device="cuda" if torch.cuda.is_available() else "cpu"
):
    """
    Perform iterative training and synthetic data generation.
    
    Args:
        base_dataset_dir: Directory containing the original dataset
        model_path: Base model path or fine-tuned model path
        num_generations: Number of generations to create
        start_generation: Starting generation number (1 for base model)
        device: Device for inference
    """
    
    # Accelerator setup
    accelerator = Accelerator()
    
    # Quantization config
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        llm_int8_enable_fp32_cpu_offload=True
    )

    # Determine if starting from base model or fine-tuned model
    is_base_model = start_generation == 1
    
    # Load tokenizer once at the beginning
    if is_base_model:
        base_model_name = model_path
        model_nick = extract_model_nick(base_model_name)
    else:
        # For fine-tuned models, extract base model name for tokenizer
        # Assume model_path format: "./phi3-squad2-gen{X}-final"
        base_model_name = "microsoft/Phi-3-mini-128k-instruct"  # Default fallback
        model_nick = extract_model_nick(base_model_name)
    
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    print(f"Starting iterative training and generation for {num_generations} generations...")
    
    # Used to load dataset
    dataset = None
    formatted_dataset = None
    
    # Main progress bar for generations
    generation_progress = tqdm(
        range(start_generation, start_generation + num_generations), 
        desc="🔄 Overall Progress", 
        unit="generation",
        position=0,
        leave=True
    )
    
    for generation in generation_progress:
        torch.cuda.empty_cache()
        generation_progress.set_description(f"🔄 Generation {generation}/{start_generation+num_generations-1}")
        
        print(f"\n{'='*50}")
        print(f"GENERATION {generation}")
        print(f"{'='*50}")
        
        # Step 1: Fine-tune model with current training dataset
        print("Step 1: Fine-tuning model...")
        
        # Prepare training arguments for this generation
        train_config = {
            "bf16": True,
            "do_eval": False,  # Disable evaluation completely
            "learning_rate": 1.0e-05,
            "log_level": "info",
            "logging_steps": 10,
            "logging_strategy": "steps",
            "lr_scheduler_type": "cosine",
            "num_train_epochs": 3,
            "max_steps": -1,
            "output_dir": f"./phi3-squad2-gen{generation}",  # Update for each generation
            "overwrite_output_dir": True,
            "per_device_train_batch_size": 4,
            "remove_unused_columns": True,
            "save_steps": 50,
            "save_total_limit": 2,
            "seed": 42,
            "gradient_checkpointing": True,
            "gradient_checkpointing_kwargs": {"use_reentrant": False},
            "gradient_accumulation_steps": 2,
            "warmup_ratio": 0.05,
            "save_strategy": "steps",
            "load_best_model_at_end": False,  # No evaluation, so no "best" model
            "disable_tqdm": False,  # Enable tqdm progress bars for training
        }

        train_args = TrainingArguments(**train_config)

        offload_cache_dir = "./offload_cache"
        os.makedirs(offload_cache_dir, exist_ok=True)
        
        # Load model for this generation
        print("📥 Loading model...")
        with tqdm(total=1, desc="🤖 Model Loading", position=1, leave=False) as model_pbar:
            if generation == start_generation and is_base_model:
                # First generation: load base model with quantization
                model = AutoModelForCausalLM.from_pretrained(
                    base_model_name,
                    device_map="auto",
                    quantization_config=bnb_config
                )
                # Load the very first train/test from MongoDB
                ds = load_dataset_from_mongo(
                    client=client,
                    train_coll=current_train_coll,   # inizialmente il subset creato
                    test_coll=args.test_coll,
                    db_name=args.db,
                    projection={"_id": 0}
                )
                formatted_dataset = DatasetDict({
                    split: ds[split].map(make_prompt)
                    for split in ds
                })

            elif generation == start_generation and not is_base_model:
                # Starting from fine-tuned model: load without quantization
                model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    device_map="auto",
                    torch_dtype=torch.float16,        # Use float16 to save memory
                    low_cpu_mem_usage=True           # Enable low CPU memory usage
                )
                # Prepare dataset based on start generation
                if start_generation == 1:
                    dataset = load_squad_subset(base_dataset_dir)
                    formatted_dataset = {
                        split: dataset[split].map(make_prompt)
                        for split in dataset.keys()
                    }
                else:
                    # Use the dedicated function to load synthetic data
                    synthetic_data_dir = os.path.join(base_dataset_dir, f"generation_{start_generation - 1}", "train")
                    # Load last synthetic train + fixed original test
                    ds = load_dataset_from_mongo(
                        client=client,
                        train_coll=args.synthetic_coll,
                        test_coll=args.test_coll,
                        db_name=args.db,
                        projection={"_id": 0}
                    )
                    formatted_dataset = ds  # se ds è già prompt-formatted, altrimenti mappa make_prompt
                  
            else:
                # Subsequent generations: load previous model - FIXED VERSION
                previous_model_path = f"./phi3-squad2-gen{generation-1}-final"
                
                # Load the model without offloading to avoid dispatch issues
                model = AutoModelForCausalLM.from_pretrained(
                    previous_model_path,
                    torch_dtype=torch.float16,
                    low_cpu_mem_usage=True,
                    device_map="auto"
                )
                
                # Manually move to device after loading
                if torch.cuda.is_available():
                    model = model.to("cuda")
                
                # Use the dedicated function to load synthetic data
                synthetic_data_dir = os.path.join(base_dataset_dir, f"generation_{generation - 1}", "train")
                dataset = load_synthetic_train(base_dataset_dir, synthetic_data_dir)
                formatted_dataset = dataset  # Already formatted

            model_pbar.update(1)
        
        # Apply PEFT configuration
        print("🔧 Applying PEFT configuration...")
        with tqdm(total=1, desc="⚙️ PEFT Setup", position=1, leave=False) as peft_pbar:
            peft_config = {
                "r": 8,  # Reduced from 16 to 8 (fewer parameters)
                "lora_alpha": 16,  # Reduced from 32 to 16
                "lora_dropout": 0.1,  # Slightly increased dropout
                "bias": "none",
                "task_type": "CAUSAL_LM",
                "target_modules": "all-linear",
                "modules_to_save": None,
            }
            lora_config = LoraConfig(**peft_config)
            model = get_peft_model(model, lora_config)
            peft_pbar.update(1)
        
        # Define tokenization function based on data structure
        def tokenize(example):
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
        
        # Tokenize current training dataset
        print("🔤 Tokenizing dataset...")
        print(f"🔍 Checking dataset structure...")
        
        # Check the structure of the first example to debug
        sample_example = formatted_dataset['train'][0]
        print(f"📋 Available columns in training data: {list(sample_example.keys())}")
        
        # Debug: Check the structure of answers field
        if 'answers' in sample_example:
            print(f"🔍 Answers field type: {type(sample_example['answers'])}")
            print(f"🔍 Answers field content: {sample_example['answers']}")
        
        if generation == 1 or (generation == start_generation and is_base_model):
            # For first generation or starting from base model, tokenize normally
            tokenized = {
                split: formatted_dataset[split].map(tokenize, batched=False)
                for split in formatted_dataset.keys()
            }
        else:
            # For subsequent generations, data is already formatted with prompts
            tokenized = {
                split: formatted_dataset[split].map(tokenize, batched=False)
                for split in formatted_dataset.keys()
            }
        
        # Fine-tune the model
        print("🚀 Starting training...")
        trainer = train_model(model, tokenized, tokenizer, train_args)
        
        # Save the fine-tuned model for this generation
        print("💾 Saving model...")
        with tqdm(total=1, desc="💾 Saving Model", position=1, leave=False) as save_pbar:
            final_model_path = f"./phi3-squad2-gen{generation}-final"
            trainer.save_model(final_model_path)
            save_pbar.update(1)
        print(f"✅ Generation {generation} model saved to {final_model_path}")
        
        # Step 2: Generate synthetic answers using the fine-tuned model
        print("Step 2: Generating synthetic answers...")
        model.eval()  # Set to evaluation mode
        
        # For synthetic generation, ensure we have proper formatted dataset structure
        if generation == 1 or (generation == start_generation and is_base_model):
            generation_dataset = formatted_dataset
        else:
            # Ensure synthetic data has the right format for generation
            # If it doesn't have 'prompt' column, create it
            if 'prompt' not in formatted_dataset['train'].column_names:
                print("🔧 Reformatting synthetic data for generation...")
                formatted_dataset['train'] = formatted_dataset['train'].map(make_prompt)
            generation_dataset = formatted_dataset
        
        synthetic_dataset = generate_synthetic_answers(
            model, tokenizer, generation_dataset, device, generation
        )
        
        # Step 3: Save synthetic dataset
        print("Step 3: Saving synthetic dataset...")
        with tqdm(total=1, desc="💾 Saving Dataset", position=1, leave=False) as dataset_save_pbar:
            save_synthetic_dataset_to_mongo(
            client=client,
            synthetic_dataset=synthetic_dataset,
            gen_number=generation,
            train_coll=args.synthetic_coll,
            metadata_coll=args.metadata_coll,
            db_name=args.db
            )
            print(f"✅ Synthetic dataset for generation {generation} saved in MongoDB collection '{args.synthetic_coll}'")
            dataset_save_pbar.update(1)
        
        print(f"Model saved to: {final_model_path}")
        
        # Clean up GPU memory
        del model
        del trainer
        torch.cuda.empty_cache()
    
    generation_progress.close()
    print(f"\n🎉 All {num_generations} generations completed!")
    print("Final models and datasets are ready for use.")
    

'''
EVALUTATION FUNCTIONS
'''

def evaluate_model(model, tokenizer, dataset, device="cuda" if torch.cuda.is_available() else "cpu", num_examples=None):
    """
    Comprehensive evaluation on test set with minimal console output.
    All detailed analysis is written to files instead of console.
    """    
    # Prepare test data using make_prompt function
    test_prompts = dataset.map(make_prompt)
    
    # Set model to evaluation mode
    model.eval()
    
    # Initialize lists for predictions and references
    preds = []
    refs = []
    raw_outputs = []
    prompts_list = []
    questions = []
    contexts = []
    
    # Limit examples if specified
    if num_examples:
        test_prompts = test_prompts.select(range(min(num_examples, len(test_prompts))))
    
    # Generate predictions with progress bar
    for example in tqdm(test_prompts, desc="🔍 Evaluating", leave=False):
        # Get prompt without answer (remove answer part from make_prompt output)
        full_prompt = example["prompt"]
        if '[/INST]' in full_prompt:
            prompt_without_answer = full_prompt.split('[/INST]')[0] + '[/INST] Answer:'
        else:
            prompt_without_answer = full_prompt
        
        # Tokenize input
        inputs = tokenizer(
            prompt_without_answer,
            return_tensors="pt",
            truncation=True,
            padding="max_length",
            max_length=512
        ).to(device)
        
        # FIXED: Store input length to extract only new tokens
        input_length = inputs['input_ids'].shape[1]
        
        # Generate response
        with torch.no_grad():
            output = model.generate(
                **inputs, 
                max_new_tokens=50,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )
        
        # FIXED: Extract only the newly generated tokens (not the entire input + output)
        generated_tokens = output[0][input_length:]  # Only the new tokens
        decoded = tokenizer.decode(generated_tokens, skip_special_tokens=True)
        
        # Extract answer - now we're working with only the generated part
        answer = decoded.strip()
        
        # Clean up answer - remove any trailing instruction markers or unwanted text
        if "Answer:" in answer:
            answer = answer.split("Answer:")[-1].strip()
        
        # Take only first line to avoid multi-line responses and remove any artifacts
        answer = answer.split('\n')[0].strip()
        
        # Remove any remaining instruction tokens that might have been generated
        answer = answer.replace('[/INST]', '').replace('[INST]', '').strip()
        
        # Store results
        preds.append(answer)
        refs.append(example["reference"])
        # Store the full decoded output for debugging
        raw_outputs.append(tokenizer.decode(output[0], skip_special_tokens=True))
        prompts_list.append(example["prompt"])
        
        # Extract question and context for detailed analysis
        if "Question:" in example["prompt"]:
            question_part = example["prompt"].split("Question:")[-1].split("[/INST]")[0].strip()
            questions.append(question_part)
        else:
            questions.append("")
            
        if "Context:" in example["prompt"]:
            context_part = example["prompt"].split("Context:")[-1].split("Question:")[0].strip()
            contexts.append(context_part[:200] + "..." if len(context_part) > 200 else context_part)
        else:
            contexts.append("")
    
    # Rest of the function remains the same...
    # [Include all the remaining code for metrics computation and file writing]

    # Compute BERTScore using silent function
    try:
        P, R, F1 = silent_bert_score(preds, refs, lang="en")
        bert_scores = {
            "precision": P.mean().item(),
            "recall": R.mean().item(),
            "f1": F1.mean().item()
        }
        bert_f1_scores = [f.item() if hasattr(f, 'item') else f for f in F1]
    except Exception as e:
        bert_scores = {"precision": 0.0, "recall": 0.0, "f1": 0.0}
        bert_f1_scores = [0.0] * len(preds)

    # Compute exact match accuracy - improved logic for "No answer" cases
    exact_matches = []
    for pred, ref in zip(preds, refs):
        pred_clean = pred.lower().strip()
        ref_clean = ref.lower().strip()
        
        # Handle "No answer" cases
        if ref_clean == "no answer":
            # For "no answer" references, check if prediction indicates no answer
            no_answer_indicators = ["no answer", "cannot answer", "not provided", "no information", "unknown"]
            match = any(indicator in pred_clean for indicator in no_answer_indicators)
            exact_matches.append(1 if match else 0)
        else:
            # For regular answers, check if reference is contained in prediction
            match = ref_clean in pred_clean or pred_clean in ref_clean
            exact_matches.append(1 if match else 0)

    exact_match_score = np.mean(exact_matches)

    # Compute F1 score (token overlap)
    f1_scores = []
    for pred, ref in zip(preds, refs):
        pred_tokens = set(pred.lower().split())
        ref_tokens = set(ref.lower().split())
        
        if len(pred_tokens) == 0 and len(ref_tokens) == 0:
            f1_scores.append(1.0)
        elif len(pred_tokens) == 0 or len(ref_tokens) == 0:
            f1_scores.append(0.0)
        else:
            common = len(pred_tokens & ref_tokens)
            precision = common / len(pred_tokens) if len(pred_tokens) > 0 else 0
            recall = common / len(ref_tokens) if len(ref_tokens) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            f1_scores.append(f1)

    f1_score = np.mean(f1_scores)

    # Compute semantic similarity (simple word overlap)
    semantic_similarities = []
    for pred, ref in zip(preds, refs):
        pred_words = set(pred.lower().split())
        ref_words = set(ref.lower().split())
        
        if len(pred_words) == 0 and len(ref_words) == 0:
            semantic_similarities.append(1.0)
        elif len(pred_words | ref_words) == 0:
            semantic_similarities.append(0.0)
        else:
            jaccard = len(pred_words & ref_words) / len(pred_words | ref_words)
            semantic_similarities.append(jaccard)

    semantic_similarity = np.mean(semantic_similarities)

    # Compute answer length statistics
    pred_lengths = [len(pred.split()) for pred in preds]
    ref_lengths = [len(ref.split()) for ref in refs]

    # Create results dictionary
    results = {
        "test_size": len(preds),
        "exact_match": exact_match_score,
        "f1_score": f1_score,
        "bert_score_f1": bert_scores["f1"],
        "semantic_similarity": semantic_similarity,
        "avg_prediction_length": np.mean(pred_lengths),
        "avg_reference_length": np.mean(ref_lengths),
        "predictions": preds,
        "references": refs,
        "questions": questions,
        "contexts": contexts,
        "individual_scores": {
            "bert_f1": bert_f1_scores,
            "token_f1": f1_scores,
            "exact_match": exact_matches,
            "semantic_similarity": semantic_similarities
        }
    }

    # Write detailed analysis to file instead of console
    with open("detailed_evaluation_analysis.txt", "w", encoding="utf-8") as f:
        f.write("DETAILED TEST SET EVALUATION ANALYSIS\n")
        f.write("="*80 + "\n\n")
        
        # Summary metrics
        f.write("EVALUATION RESULTS SUMMARY\n")
        f.write("-"*60 + "\n")
        f.write(f"Test Set Size: {len(preds)}\n")
        f.write(f"Exact Match: {exact_match_score:.4f}\n")
        f.write(f"F1 Score: {f1_score:.4f}\n")
        f.write(f"BERTScore F1: {bert_scores['f1']:.4f}\n")
        f.write(f"Semantic Similarity: {semantic_similarity:.4f}\n")
        f.write(f"Avg Prediction Length: {np.mean(pred_lengths):.2f} words\n")
        f.write(f"Avg Reference Length: {np.mean(ref_lengths):.2f} words\n\n")
        
        # Example analysis
        f.write("DETAILED PREDICTION EXAMPLES\n")
        f.write("="*80 + "\n\n")
        
        # Select diverse examples: best, worst, and random
        if bert_f1_scores and len(bert_f1_scores) > 0:
            sorted_indices = sorted(range(len(bert_f1_scores)), key=lambda i: bert_f1_scores[i], reverse=True)
            
            best_indices = sorted_indices[:3]  # Top 3
            worst_indices = sorted_indices[-3:]  # Bottom 3
            random_indices = random.sample(range(len(preds)), min(4, len(preds)))  # Random 4
            
            example_categories = [
                ("BEST PREDICTIONS", best_indices),
                ("WORST PREDICTIONS", worst_indices),
                ("RANDOM PREDICTIONS", random_indices)
            ]
            
            for category_name, indices in example_categories:
                f.write(f"\n{category_name}:\n")
                f.write("-" * 80 + "\n")
                
                for i, idx in enumerate(indices):
                    if idx >= len(preds):
                        continue
                        
                    f.write(f"\nExample {i+1} (Index {idx}):\n")
                    f.write(f"BERTScore F1: {bert_f1_scores[idx]:.4f}\n")
                    f.write(f"Token F1: {f1_scores[idx]:.4f}\n")
                    f.write(f"Exact Match: {'✓' if exact_matches[idx] else '✗'}\n")
                    
                    if idx < len(questions) and questions[idx]:
                        f.write(f"Question: {questions[idx]}\n")
                    if idx < len(contexts) and contexts[idx]:
                        f.write(f"Context: {contexts[idx]}\n")
                    
                    f.write(f"Reference Answer: {refs[idx]}\n")
                    f.write(f"Model Prediction: {preds[idx]}\n")
                    
                    # Analysis
                    pred_words = len(preds[idx].split())
                    ref_words = len(refs[idx].split())
                    f.write(f"Length: Pred={pred_words} words, Ref={ref_words} words\n")
                    
                    # Word overlap analysis
                    pred_lower = preds[idx].lower()
                    ref_lower = refs[idx].lower()
                    common_words = set(pred_lower.split()) & set(ref_lower.split())
                    f.write(f"Common words: {len(common_words)}\n")
                    
                    f.write("-" * 50 + "\n")
        
        # Full predictions list
        f.write(f"\n\nFULL PREDICTIONS LIST\n")
        f.write("="*80 + "\n\n")
        
        for i, (prompt, pred, ref, bert_f1, token_f1, em) in enumerate(zip(
            prompts_list, preds, refs, bert_f1_scores, f1_scores, exact_matches
        )):
            f.write(f"Example {i+1}:\n")
            f.write(f"BERTScore F1: {bert_f1:.4f}\n")
            f.write(f"Token F1: {token_f1:.4f}\n")
            f.write(f"Exact Match: {'✓' if em else '✗'}\n")
            
            if "Question:" in prompt:
                question = prompt.split("Question:")[-1].split("[/INST]")[0].strip()
                f.write(f"Question: {question}\n")
            
            f.write(f"Reference: {ref}\n")
            f.write(f"Prediction: {pred}\n")
            f.write("-" * 50 + "\n\n")

    return results

def evaluate_generation(base_model_name, generation_num, mongo_client, device, 
                       save_results=True, results_dir="evaluation_results_squad01", show_examples=True, num_examples=5):
    """
    Evaluate a specific generation model on the test set with optional result saving.
    Loads the model and tokenizer internally based on generation number.
    
    Args:
        base_model_name: Base model name (e.g., "microsoft/Phi-3-mini-128k-instruct")
        generation_num: Generation number to evaluate
        base_dataset_dir: Directory containing the original test dataset
        device: Device for inference
        save_results: Whether to save results to JSON file
        results_dir: Directory to save evaluation results
        show_examples: Whether to display example predictions
        num_examples: Number of examples to show
        
    Returns:
        dict: Dictionary containing evaluation metrics
    """
    torch.cuda.empty_cache()
    
    # Accelerator setup
    accelerator = Accelerator()
    
    # Quantization config
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        llm_int8_enable_fp32_cpu_offload=True
    )

    # Determine if starting from base model or fine-tuned model
    is_base_model = generation_num == 0
    
    # Extract model nickname for path construction
    model_nick = extract_model_nick(base_model_name)
    
    # Load tokenizer
    print("📥 Loading tokenizer...")
    with tqdm(total=1, desc="🔤 Tokenizer Loading", position=1, leave=False) as tokenizer_pbar:
        tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer_pbar.update(1)
    
    # Load model based on generation number
    print("📥 Loading model...")
    with tqdm(total=1, desc="🤖 Model Loading", position=1, leave=False) as model_pbar:
        if generation_num == 0:
            # Load base model
            model = AutoModelForCausalLM.from_pretrained(
                base_model_name,
                quantization_config=bnb_config,
                device_map="auto"
            )
        else:
            # Load fine-tuned model from specific generation
            model_path = f"./phi3-squad2-gen{generation_num}-final"
            # Create offload directory
            offload_cache_dir = "./offload_cache"
            os.makedirs(offload_cache_dir, exist_ok=True)
            
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model for generation {generation_num} not found at {model_path}")
            
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                device_map="auto",
                torch_dtype=torch.float16,
                offload_folder=offload_cache_dir,
                low_cpu_mem_usage=True
            )
        model_pbar.update(1)
    
    # Load original test dataset from MongoDB
    with tqdm(total=1, desc="📂 Loading Test Data", position=1, leave=False) as load_pbar:
        ds = load_dataset_from_mongo(
            client=mongo_client,
            train_coll="dummy_train_coll",    # non usato qui
            test_coll=args.test_coll,         # passa la collection del test
            db_name=args.db,
            projection={"_id": 0}
        )
        test_dataset = ds["test"]
        load_pbar.update(1)

    
    # Set model to evaluation mode
    model.eval()
    
    # Run evaluation
    with tqdm(total=1, desc="📊 Evaluating", position=1, leave=False) as eval_pbar:
        evaluation_results = evaluate_model(model, tokenizer, test_dataset, device)
        eval_pbar.update(1)
    
    # Add metadata to results
    evaluation_results.update({
        'generation': generation_num,
        'test_dataset_size': len(test_dataset),
        'base_dataset_dir': base_dataset_dir,
        'base_model_name': base_model_name,
        'model_nick': model_nick,
        'model_path': f"./phi3-squad2-gen{generation_num}-final" if generation_num > 0 else base_model_name,
    })
    
    # Create results directory and detailed report file
    os.makedirs(results_dir, exist_ok=True)
    detailed_report_file = os.path.join(results_dir, f"generation_{generation_num}_detailed_report.txt")
    
    # Write all detailed output to file instead of printing
    with open(detailed_report_file, 'w', encoding='utf-8') as report_file:
        # Redirect detailed output to file
        report_file.write(f"DETAILED EVALUATION REPORT - GENERATION {generation_num}\n")
        report_file.write("="*80 + "\n\n")
        
        # Basic metrics
        report_file.write(f"📊 Generation {generation_num} Evaluation Results:\n")
        report_file.write(f"   Exact Match: {evaluation_results['exact_match']:.3f}\n")
        report_file.write(f"   F1 Score: {evaluation_results['f1_score']:.3f}\n")
        report_file.write(f"   BERTScore F1: {evaluation_results['bert_score_f1']:.3f}\n")
        report_file.write(f"   Semantic Similarity: {evaluation_results['semantic_similarity']:.3f}\n\n")
    
    # Only print brief summary to console
    print(f"📊 Generation {generation_num} Evaluation Completed:")
    print(f"   Exact Match: {evaluation_results['exact_match']:.3f}")
    print(f"   F1 Score: {evaluation_results['f1_score']:.3f}")
    print(f"   BERTScore F1: {evaluation_results['bert_score_f1']:.3f}")
    print(f"   Semantic Similarity: {evaluation_results['semantic_similarity']:.3f}")
    
    # Save results if requested
    if save_results:
        results_file = os.path.join(results_dir, f"generation_{generation_num}_results.json")
        
        # Create a serializable version of results for JSON
        json_results = {}
        for key, value in evaluation_results.items():
            if isinstance(value, (list, dict, str, int, float, bool, type(None))):
                json_results[key] = value
            elif isinstance(value, np.ndarray):
                json_results[key] = value.tolist()
            else:
                json_results[key] = str(value)
        
        with open(results_file, 'w') as f:
            json.dump(json_results, f, indent=2)
        
        print(f"💾 Results saved to: {results_file}")
        print(f"📄 Detailed report saved to: {detailed_report_file}")
    
    # Clean up GPU memory
    del model
    torch.cuda.empty_cache()
    
    print(f"🎉 Evaluation completed for Generation {generation_num}!")
    
    return evaluation_results

def evaluate_multiple_generations(
    base_model_name="microsoft/Phi-3-mini-128k-instruct",
    mongo_client,
    start_generation=0,
    num_generations=4,
    device="cuda" if torch.cuda.is_available() else "cpu",
    save_results=True,
    results_dir="evaluation_results_squad01"
):
    """
    Evaluate multiple generations in sequence with progress tracking and result comparison.
    
    Args:
        base_model_name: Base model name for tokenizer and generation 0
        base_dataset_dir: Directory containing the original test dataset
        start_generation: Starting generation number (0 for base model)
        num_generations: Number of generations to evaluate
        device: Device for inference
        save_results: Whether to save results to JSON files
        results_dir: Directory to save all evaluation results
        
    Returns:
        dict: Dictionary containing all evaluation results by generation
    """
    
    print(f"🚀 Starting evaluation of {num_generations} generations...")
    print(f"📋 Generations: {start_generation} to {start_generation + num_generations - 1}")
    print(f"💾 Results will be saved to: {results_dir}")
    print("="*80)
    
    # Dictionary to store all results
    all_results = {}
    
    # Create results directory
    os.makedirs(results_dir, exist_ok=True)
    
    # Main progress bar for all generations
    generation_progress = tqdm(
        range(start_generation, start_generation + num_generations),
        desc="🔄 Evaluating Generations",
        unit="generation",
        position=0,
        leave=True
    )
    
    # Track metrics for comparison
    metrics_comparison = {
        'generations': [],
        'exact_match': [],
        'f1_score': [],
        'bert_score_f1': [],
        'semantic_similarity': []
    }
    
    for generation_num in generation_progress:
        generation_progress.set_description(f"🔄 Evaluating Generation {generation_num}")
        
        print(f"\n{'='*60}")
        print(f"EVALUATING GENERATION {generation_num}")
        print(f"{'='*60}")
        
        try:
            # Evaluate this generation
            results = evaluate_generation(
                base_model_name=base_model_name,
                generation_num=generation_num,
                base_dataset_dir=base_dataset_dir,
                device=device,
                save_results=save_results,
                results_dir=results_dir
            )
            
            # Store results
            all_results[f"generation_{generation_num}"] = results
            
            # Track metrics for comparison
            metrics_comparison['generations'].append(generation_num)
            metrics_comparison['exact_match'].append(results['exact_match'])
            metrics_comparison['f1_score'].append(results['f1_score'])
            metrics_comparison['bert_score_f1'].append(results['bert_score_f1'])
            metrics_comparison['semantic_similarity'].append(results['semantic_similarity'])
            
            # Update progress bar with current metrics
            generation_progress.set_postfix({
                'EM': f"{results['exact_match']:.3f}",
                'F1': f"{results['f1_score']:.3f}",
                'BERT': f"{results['bert_score_f1']:.3f}",
                'SIM': f"{results['semantic_similarity']:.3f}"
            })
            
        except Exception as e:
            print(f"❌ Error evaluating generation {generation_num}: {str(e)}")
            # Store error information
            all_results[f"generation_{generation_num}"] = {
                'error': str(e),
                'generation': generation_num,
                'status': 'failed'
            }
            continue
    
    generation_progress.close()
    
    # Create comprehensive comparison report
    print(f"\n{'='*80}")
    print("EVALUATION SUMMARY")
    print(f"{'='*80}")
    
    # Print comparison table
    print(f"{'Generation':<12} {'Exact Match':<12} {'F1 Score':<12} {'BERTScore F1':<14} {'Semantic Sim':<14}")
    print("-" * 70)
    
    for i, gen in enumerate(metrics_comparison['generations']):
        if f"generation_{gen}" in all_results and 'error' not in all_results[f"generation_{gen}"]:
            print(f"{gen:<12} {metrics_comparison['exact_match'][i]:<12.3f} "
                  f"{metrics_comparison['f1_score'][i]:<12.3f} "
                  f"{metrics_comparison['bert_score_f1'][i]:<14.3f} "
                  f"{metrics_comparison['semantic_similarity'][i]:<14.3f}")
        else:
            print(f"{gen:<12} {'FAILED':<12} {'FAILED':<12} {'FAILED':<14} {'FAILED':<14}")
    
    # Find best performing generation for each metric
    if metrics_comparison['generations']:
        print(f"\n📈 BEST PERFORMING GENERATIONS:")
        if metrics_comparison['exact_match']:
            best_em_idx = np.argmax(metrics_comparison['exact_match'])
            best_em_gen = metrics_comparison['generations'][best_em_idx]
            print(f"   Exact Match: Generation {best_em_gen} ({metrics_comparison['exact_match'][best_em_idx]:.3f})")
        
        if metrics_comparison['f1_score']:
            best_f1_idx = np.argmax(metrics_comparison['f1_score'])
            best_f1_gen = metrics_comparison['generations'][best_f1_idx]
            print(f"   F1 Score: Generation {best_f1_gen} ({metrics_comparison['f1_score'][best_f1_idx]:.3f})")
        
        if metrics_comparison['bert_score_f1']:
            best_bert_idx = np.argmax(metrics_comparison['bert_score_f1'])
            best_bert_gen = metrics_comparison['generations'][best_bert_idx]
            print(f"   BERTScore F1: Generation {best_bert_gen} ({metrics_comparison['bert_score_f1'][best_bert_idx]:.3f})")
        
        if metrics_comparison['semantic_similarity']:
            best_sim_idx = np.argmax(metrics_comparison['semantic_similarity'])
            best_sim_gen = metrics_comparison['generations'][best_sim_idx]
            print(f"   Semantic Similarity: Generation {best_sim_gen} ({metrics_comparison['semantic_similarity'][best_sim_idx]:.3f})")
    
    # Save comprehensive results
    if save_results:
        # Save all results in one file
        comprehensive_results_file = os.path.join(results_dir, "comprehensive_evaluation_results.json")
        
        # Create serializable version
        json_all_results = {}
        for gen_key, gen_results in all_results.items():
            json_gen_results = {}
            for key, value in gen_results.items():
                if isinstance(value, (list, dict, str, int, float, bool, type(None))):
                    json_gen_results[key] = value
                elif isinstance(value, np.ndarray):
                    json_gen_results[key] = value.tolist()
                else:
                    json_gen_results[key] = str(value)
            json_all_results[gen_key] = json_gen_results
        
        # Add comparison metrics
        json_all_results['comparison_metrics'] = metrics_comparison
        json_all_results['evaluation_metadata'] = {
            'base_model_name': base_model_name,
            'base_dataset_dir': base_dataset_dir,
            'start_generation': start_generation,
            'num_generations': num_generations,
            'total_generations_evaluated': len(metrics_comparison['generations']),
            'evaluation_timestamp': str(pd.Timestamp.now()) if 'pd' in globals() else "timestamp_not_available"
        }
        
        with open(comprehensive_results_file, 'w') as f:
            json.dump(json_all_results, f, indent=2)
        
        # Save comparison table as CSV
        comparison_csv_file = os.path.join(results_dir, "generations_comparison.csv")
        comparison_df_data = {
            'Generation': metrics_comparison['generations'],
            'Exact_Match': metrics_comparison['exact_match'],
            'F1_Score': metrics_comparison['f1_score'],
            'BERTScore_F1': metrics_comparison['bert_score_f1'],
            'Semantic_Similarity': metrics_comparison['semantic_similarity']
        }
        
        # Create simple CSV manually
        with open(comparison_csv_file, 'w') as f:
            f.write("Generation,Exact_Match,F1_Score,BERTScore_F1,Semantic_Similarity\n")
            for i in range(len(metrics_comparison['generations'])):
                f.write(f"{metrics_comparison['generations'][i]},"
                       f"{metrics_comparison['exact_match'][i]},"
                       f"{metrics_comparison['f1_score'][i]},"
                       f"{metrics_comparison['bert_score_f1'][i]},"
                       f"{metrics_comparison['semantic_similarity'][i]}\n")
        
        print(f"\n💾 Comprehensive results saved to:")
        print(f"   📊 {comprehensive_results_file}")
        print(f"   📋 {comparison_csv_file}")
    
    print(f"\n🎉 Evaluation of {len(metrics_comparison['generations'])} generations completed!")
    
    return all_results

'''
EXPORT PREDICTIONS
'''

def export_predictions_to_json(model, tokenizer, test_dataset, generation_num, device, output_dir="predictions_export_squad05"):
    """
    Export all questions, model predictions, and reference answers to a JSON file.
    
    Args:
        model: The loaded model for generating predictions
        tokenizer: The tokenizer for the model
        test_dataset: The test dataset containing questions and reference answers
        generation_num: Generation number (0 for base model)
        device: Device for inference
        output_dir: Directory to save the JSON file
        
    Returns:
        str: Path to the saved JSON file
    """
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Prepare test data using make_prompt function
    test_prompts = test_dataset.map(make_prompt)
    
    # Set model to evaluation mode
    model.eval()
    
    # Initialize results list
    results = []
    
    print(f"🔍 Generating predictions for Generation {generation_num}...")
    
    # Generate predictions with progress bar
    for idx, example in enumerate(tqdm(test_prompts, desc=f"📝 Gen {generation_num} - Processing", leave=False)):
        # Get prompt without answer (remove answer part from make_prompt output)
        full_prompt = example["prompt"]
        if '[/INST]' in full_prompt:
            prompt_without_answer = full_prompt.split('[/INST]')[0] + '[/INST] Answer:'
        else:
            prompt_without_answer = full_prompt
        
        # Tokenize input
        inputs = tokenizer(
            prompt_without_answer,
            return_tensors="pt",
            truncation=True,
            padding="max_length",
            max_length=512
        ).to(device)
        
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
        
        # Extract answer - look for content after [/INST] or after "Answer:" if present
        if '[/INST]' in decoded:
            answer_part = decoded.split('[/INST]')[-1].strip()
            
            # If there's an "Answer:" pattern in the generated text, extract after it
            if "Answer:" in answer_part:
                prediction = answer_part.split("Answer:")[-1].strip()
            else:
                prediction = answer_part
        else:
            # Fallback: look for "Answer:" pattern in the entire decoded text
            if "Answer:" in decoded:
                prediction = decoded.split("Answer:")[-1].strip()
            else:
                prediction = decoded.strip()
        
        # Clean up prediction - remove any trailing instruction markers or unwanted text
        prediction = prediction.split('\n')[0].strip()  # Take only first line to avoid multi-line responses
        
        # Extract context and question from the original example
        context = example.get("context", "")
        question = example.get("question", "")
        reference = example.get("reference", "")
        
        # Create result entry
        result_entry = {
            "id": idx,
            "context": context,
            "question": question,
            "reference_answer": reference,
            "model_prediction": prediction,
            "generation": generation_num,
            "full_prompt": full_prompt,
            "raw_model_output": decoded
        }
        
        results.append(result_entry)
    
    # Save to JSON file
    json_filename = f"generation_{generation_num}_predictions.json"
    json_filepath = os.path.join(output_dir, json_filename)
    
    # Create metadata
    metadata = {
        "generation": generation_num,
        "total_examples": len(results),
        "model_type": "base_model" if generation_num == 0 else "fine_tuned",
        "export_timestamp": str(pd.Timestamp.now()) if 'pd' in globals() else "timestamp_not_available",
        "description": f"Predictions from generation {generation_num} model on test dataset"
    }
    
    # Final JSON structure
    final_json = {
        "metadata": metadata,
        "predictions": results
    }
    
    # Save to file
    with open(json_filepath, 'w', encoding='utf-8') as f:
        json.dump(final_json, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Exported {len(results)} predictions to: {json_filepath}")
    return json_filepath


def export_all_generations_predictions(
    base_model_name,
    mongo_client,
    db_name,
    test_coll,
    start_generation=0,
    end_generation=5,
    device="cuda",
    output_dir="predictions_export_squad05"
):
    """
    Export predictions from all generations to JSON, reading test set from MongoDB.
    """
    # … logging …
    # Load tokenizer (unchanged)
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    tokenizer.pad_token = tokenizer.eos_token

    # **NEW**: load test set from Mongo
    ds = load_dataset_from_mongo(
        client=mongo_client,
        train_coll="unused",
        test_coll=test_coll,
        db_name=db_name,
        projection={"_id": 0}
    )
    test_dataset = ds["test"]
    print(f"   Test dataset size: {len(test_dataset)} examples")
    
    # Main progress bar for all generations
    generation_progress = tqdm(
        range(start_generation, end_generation + 1),
        desc="🔄 Exporting Generations",
        unit="generation",
        position=0,
        leave=True
    )
    
    for generation_num in generation_progress:
        generation_progress.set_description(f"🔄 Exporting Generation {generation_num}")
        
        print(f"\n{'='*60}")
        print(f"EXPORTING GENERATION {generation_num}")
        print(f"{'='*60}")
        
        try:
            torch.cuda.empty_cache()
            
            # Load model based on generation number
            print("📥 Loading model...")
            with tqdm(total=1, desc="🤖 Model Loading", position=1, leave=False) as model_pbar:
                if generation_num == 0:
                    # Load base model
                    model = AutoModelForCausalLM.from_pretrained(
                        base_model_name,
                        quantization_config=bnb_config,
                        device_map="auto"
                    )
                else:
                    # Load fine-tuned model from specific generation
                    model_path = f"./phi3-squad2-gen{generation_num}-final"
                    
                    if not os.path.exists(model_path):
                        print(f"❌ Model for generation {generation_num} not found at {model_path}")
                        continue
                    
                    # Create offload directory
                    offload_cache_dir = "./offload_cache"
                    os.makedirs(offload_cache_dir, exist_ok=True)
                    
                    model = AutoModelForCausalLM.from_pretrained(
                        model_path,
                        device_map="auto",
                        torch_dtype=torch.float16,
                        offload_folder=offload_cache_dir,
                        low_cpu_mem_usage=True
                    )
                model_pbar.update(1)
            
            # Export predictions for this generation
            json_filepath = export_predictions_to_json(
                model=model,
                tokenizer=tokenizer,
                test_dataset=test_dataset,
                generation_num=generation_num,
                device=device,
                output_dir=output_dir
            )
            
            # Store the file path
            exported_files[generation_num] = json_filepath
            
            # Update progress bar
            generation_progress.set_postfix({
                'Exported': f"{len(exported_files)}/{end_generation - start_generation + 1}",
                'Current': f"Gen {generation_num}"
            })
            
            # Clean up GPU memory
            del model
            torch.cuda.empty_cache()
            
            print(f"✅ Generation {generation_num} export completed!")
            
        except Exception as e:
            print(f"❌ Error exporting generation {generation_num}: {str(e)}")
            exported_files[generation_num] = f"ERROR: {str(e)}"
            continue
    
    generation_progress.close()
    
    # Create summary file
    summary_file = os.path.join(output_dir, "export_summary.json")
    summary_data = {
        "export_metadata": {
            "base_model_name": base_model_name,
            "base_dataset_dir": base_dataset_dir,
            "start_generation": start_generation,
            "end_generation": end_generation,
            "total_generations": end_generation - start_generation + 1,
            "successful_exports": len([f for f in exported_files.values() if not f.startswith("ERROR:")]),
            "failed_exports": len([f for f in exported_files.values() if f.startswith("ERROR:")]),
            "export_timestamp": str(pd.Timestamp.now()) if 'pd' in globals() else "timestamp_not_available"
        },
        "exported_files": exported_files,
        "test_dataset_info": {
            "size": len(test_dataset),
            "source": base_dataset_dir
        }
    }
    
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary_data, f, indent=2, ensure_ascii=False)
    
    # Print final summary
    print(f"\n{'='*80}")
    print("EXPORT SUMMARY")
    print(f"{'='*80}")
    
    successful_exports = 0
    failed_exports = 0
    
    for gen_num, filepath in exported_files.items():
        if filepath.startswith("ERROR:"):
            print(f"❌ Generation {gen_num}: {filepath}")
            failed_exports += 1
        else:
            print(f"✅ Generation {gen_num}: {filepath}")
            successful_exports += 1
    
    print(f"\n📊 Export Statistics:")
    print(f"   Total generations: {end_generation - start_generation + 1}")
    print(f"   Successful exports: {successful_exports}")
    print(f"   Failed exports: {failed_exports}")
    print(f"   Test examples per file: {len(test_dataset)}")
    print(f"\n💾 Summary saved to: {summary_file}")
    print(f"📁 All files saved in: {output_dir}")
    
    print(f"\n🎉 Prediction export completed!")
    
    return exported_files