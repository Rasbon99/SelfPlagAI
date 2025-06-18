# Standard library
import json
import os
import argparse

# Third-party libraries - data processing
import pandas as pd
from datasets import load_dataset

# Third-party libraries - transformers and related
from transformers import (
    Accelerator,
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    LoraConfig,
    TrainingArguments,
    get_peft_model,
)

# Local project modules
from utils import (
    evaluate_test_set_with_examples,
    generate_dataset_from_model,
    make_prompt,
    train_model,
    tokenize,
)

def parse_args():
    parser = argparse.ArgumentParser(
        description="Recursive fine-tuning and degradation tracking for Phi-3 model"
    )
    parser.add_argument(
        "--n_iterations", type=int, default=5,
        help="Number of recursive fine-tuning iterations"
    )
    parser.add_argument(
        "--output_dir", type=str, default="./runs",
        help="Base directory for storing run outputs"
    )
    return parser.parse_args()


def prepare_initial_dataset():
    raw = load_dataset("squad_v2")
    return raw['train'], raw['validation']


def main(current_model_path="microsoft/phi-3-mini-128k-instruct"):
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        llm_int8_enable_fp32_cpu_offload=True
    )
    peft_cfg = LoraConfig(**{
        "r": 8,
        "lora_alpha": 16,
        "lora_dropout": 0.1,
        "bias": "none",
        "task_type": "CAUSAL_LM",
        "target_modules": "all-linear"
    })

    tokenizer = AutoTokenizer.from_pretrained(current_model_path)
    tokenizer.pad_token = tokenizer.eos_token

    train_dataset, test_dataset = prepare_initial_dataset()
    formatted_test = test_dataset.map(make_prompt)

    formatted_train = train_dataset.map(make_prompt)
    tokenized_train = formatted_train.map(lambda x: tokenize(x, tokenizer), batched=True)
    
    all_metrics = []

    for i in range(1, args.n_iterations + 1):
        print(f"\n=== Iteration {i}/{args.n_iterations} ===")
        run_dir = os.path.join(args.output_dir, f"run_{i}")
        os.makedirs(run_dir, exist_ok=True)

        # Definisco train_args dentro il ciclo per salvare ogni iterazione separatamente
        train_args = TrainingArguments(
            output_dir=run_dir,
            per_device_train_batch_size=8,
            num_train_epochs=1,
            logging_steps=10,
            save_strategy="no",
            evaluation_strategy="no",
            learning_rate=1e-4,
            weight_decay=0.01,
            warmup_steps=100,
            save_total_limit=1,
            fp16=True,
            report_to=[]
        )

        print("Loading model and applying LoRA...")
        model = AutoModelForCausalLM.from_pretrained(
            current_model_path,
            device_map="auto",
            quantization_config=bnb_config
        )
        model = get_peft_model(model, peft_cfg)

        print("Starting fine-tuning on training dataset...")
        trainer = train_model(model, tokenized_train, tokenizer, train_args)

        print("Evaluating on fixed test dataset...")
        eval_res = evaluate_test_set_with_examples(
            model, tokenizer,
            {"test": formatted_test},
            make_prompt,
            num_examples=len(formatted_test)
        )
        print(f"Iteration {i} BERT F1: {eval_res['bert_score']['f1']:.4f}, Exact Match: {eval_res['exact_match']:.4f}")

        all_metrics.append({
            "iteration": i,
            "bert_f1": eval_res["bert_score"]["f1"],
            "exact_match": eval_res["exact_match"]
        })

        print(f"Saving model checkpoint to {run_dir} ...")
        trainer.save_model(run_dir)
        current_model_path = run_dir

        print("Generating new synthetic training dataset from current model...")
        new_train_dataset = generate_dataset_from_model(model, tokenizer, train_dataset)

        formatted_train = new_train_dataset.map(make_prompt)
        tokenized_train = formatted_train.map(lambda x: tokenize(x, tokenizer), batched=True)

    metrics_path = os.path.join(args.output_dir, "degradation_metrics.json")
    with open(metrics_path, 'w') as f:
        json.dump(all_metrics, f, indent=2)
    print(f"\n📁 Saved degradation metrics to {metrics_path}")

    df_metrics = pd.DataFrame(all_metrics)
    csv_path = os.path.join(args.output_dir, "degradation_metrics.csv")
    df_metrics.to_csv(csv_path, index=False)
    print(f"📄 Saved degradation metrics CSV to {csv_path}")


if __name__ == "__main__":
    main()