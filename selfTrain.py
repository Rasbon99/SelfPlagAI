# Standard library
import os
import io
import argparse
import logging
import warnings

# Third‑party libraries — data processing & environment
import torch
import pandas as pd
from dotenv import load_dotenv
from huggingface_hub import login
from datasets import DatasetDict

# Local modules — database & pipeline utilities
from db_utils import (
    get_mongo_client,
    read_collection,
    insert_dataframe_to_mongo,
    drop_collection,
)
from mongo_pipeline_utils import (
    read_original_and_create_subset,
    load_dataset_from_mongo,
    save_synthetic_dataset_to_mongo,
)
from train_utils import make_prompt
from training_pipeline import (
    iterative_training_and_generation,
    evaluate_multiple_generations,
    export_all_generations_predictions,
)

def parse_args():
    parser = argparse.ArgumentParser(
        description="End‑to‑end recursive fine‑tuning via MongoDB"
    )
    parser.add_argument("--hf-model", type=str, required=True,
                        help="HuggingFace model (e.g. microsoft/phi-3-mini-128k-instruct)")
    parser.add_argument("--num-generations", type=int, default=5,
                        help="Number of synthetic generations to perform")
    parser.add_argument("--sample-frac", type=float, default=0.005,
                        help="Fraction of original SQuAD train to sample (e.g. 0.005 → 0.5%)")
    parser.add_argument("--db", type=str, default="squadv2",
                        help="MongoDB database name")
    parser.add_argument("--orig-train-coll", type=str, default="squadv2_original_train",
                        help="Original full SQuAD v2 train collection")
    parser.add_argument("--test-coll", type=str, default="squadv2_original_test",
                        help="Original full SQuAD v2 test collection")
    parser.add_argument("--subset-coll", type=str, default="squad_subset_0_5pct",
                        help="Collection for the sampled subset")
    parser.add_argument("--synthetic-coll", type=str, default="squad_synthetic_train",
                        help="Collection for synthetic train generations")
    parser.add_argument("--metadata-coll", type=str, default="synthetic_metadata",
                        help="Collection for synthetic metadata")
    return parser.parse_args()

def main():
    args = parse_args()

    # 1) Load credentials and connect to MongoDB
    load_dotenv("key.env")
    client = get_mongo_client(
        username=os.getenv("USERNAME"),
        password=os.getenv("PASSWORD")
    )

    # 2) Sample a small subset from the original train and store in Mongo
    print("▶ Creating initial subset in MongoDB...")
    read_original_and_create_subset(
        client=client,
        original_coll=args.orig_train_coll,
        subset_coll=args.subset_coll,
        db_name=args.db,
        sample_frac=args.sample_frac,
        random_state=42
    )

    # 3) Login to HF Hub & silence noisy logs
    hf_token = os.getenv("HUGGINGFACE_TOKEN")
    login(token=hf_token)
    logging.getLogger("transformers").setLevel(logging.CRITICAL)
    logging.getLogger("accelerate").setLevel(logging.CRITICAL)
    warnings.filterwarnings("ignore")
    torch.cuda.empty_cache()

    # 4) Initialize BERTScore silently
    print("▶ Initializing BERTScore...")
    with io.StringIO(), io.StringIO():
        from bert_score import score
        _ = score(["test"], ["test"], lang="en", verbose=False)
    print("BERTScore ready ✅\n")

    # 5) Iterative training & synthetic generation
    current_train_coll = args.subset_coll
    for gen in range(1, args.num_generations + 1):
        print(f"=== Generation {gen} ===")

        # 5.1) Load train + test from Mongo into HF DatasetDict
        ds = load_dataset_from_mongo(
            client=client,
            train_coll=current_train_coll,
            test_coll=args.test_coll,
            db_name=args.db,
            projection={"_id": 0}
        )

        # 5.2) Format prompts
        print("Formatting prompts...")
        formatted = DatasetDict({
            split: ds[split].map(make_prompt)
            for split in ds
        })

        # 5.3) Fine‑tune & generate synthetic train
        synthetic_ds = iterative_training_and_generation(
            train_dataset=formatted["train"],
            model_name_or_path=args.hf_model,
            generation_num=gen,
            device="cuda" if torch.cuda.is_available() else "cpu"
        )

        # 5.4) Save new synthetic train + metadata into Mongo
        print("Saving synthetic train and metadata to MongoDB...")
        save_synthetic_dataset_to_mongo(
            client=client,
            synthetic_dataset=synthetic_ds,
            gen_number=gen,
            train_coll=args.synthetic_coll,
            metadata_coll=args.metadata_coll,
            db_name=args.db
        )

        current_train_coll = args.synthetic_coll
        print()

    # 6) Evaluation of all generations
    print("▶ Evaluating all generations...")
    evaluate_multiple_generations(
        base_model_name=args.hf_model,
        num_generations=args.num_generations,
        device="cuda" if torch.cuda.is_available() else "cpu",
        mongo_client=client,
        db_name=args.db,
        test_coll=args.test_coll,
        synthetic_coll=args.synthetic_coll,
        save_results=True,
        results_dir="evaluation_results_squad05"
    )

    # 7) Export all predictions
    print("▶ Exporting predictions...")
    export_all_generations_predictions(
        base_model_name=args.hf_model,
        mongo_client=client,
        db_name=args.db,
        test_coll=args.test_coll,
        start_generation=1,
        end_generation=args.num_generations,
        device="cuda" if torch.cuda.is_available() else "cpu",
        output_dir="predictions_export"
    )

    client.close()
    print("\n✅ All done.")

if __name__ == "__main__":
    main()