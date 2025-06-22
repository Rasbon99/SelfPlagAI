# --- Standard Library ---
import os
import argparse
import logging
import warnings

# --- Third-party Libraries: Deep Learning & Environment ---
import torch
from dotenv import load_dotenv
from huggingface_hub import login

# --- Local Modules: Database & Pipeline Utilities ---
from db_utils import (
    get_mongo_client,
)

from train_utils import (
    read_original_and_create_subset,
    iterative_training_and_generation,
    evaluate_multiple_generations,
    export_all_generations_predictions,
    save_synthetic_dataset_to_mongo,
)

def parse_args():
    parser = argparse.ArgumentParser(
        description="End-to-end recursive fine-tuning via MongoDB"
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
    parser.add_argument("--train-subset-coll", type=str, default="squad_subset_0_5pct",
                        help="Collection for the sampled train subset")
    parser.add_argument("--test-subset-coll", type=str, default="squad_subset_0_5pct",
                        help="Collection for the sampled test subset")
    parser.add_argument("--synthetic-coll", type=str, default="squad_synthetic_train",
                        help="Collection for synthetic train generations")
    parser.add_argument("--metadata-coll", type=str, default="synthetic_metadata",
                        help="Collection for synthetic metadata")
    parser.add_argument("--predictions-coll-prefix", type=str, default="predictions_gen",
                        help="Prefix for MongoDB collections to store exported predictions")
    return parser.parse_args()

def main():
    args = parse_args()
    load_dotenv("key.env")

    username = os.getenv("MONGO_USERNAME")
    password = os.getenv("MONGO_PASSWORD")
    print(f"Connecting to MongoDB as {username}...")
    if not username or not password:
        print("❌ USERNAME or PASSWORD not found in .env file.")
        return

    try:
        client = get_mongo_client(username, password)
        client.admin.command("ping")
        print("✅ Connected to MongoDB")
    except Exception as e:
        print(f"❌ MongoDB connection failed: {e}")
        return

    print("▶ Creating initial subset in MongoDB...")
    read_original_and_create_subset(
        client=client,
        original_coll=args.orig_train_coll,
        subset_coll=args.train_subset_coll,
        db_name=args.db,
        sample_frac=args.sample_frac,
        random_state=42
    )

    read_original_and_create_subset(
        client=client,
        original_coll=args.test_coll,
        subset_coll=args.test_subset_coll,
        db_name=args.db,
        sample_frac=args.sample_frac,
        random_state=42
    )

    hf_token = os.getenv("HUGGINGFACE_TOKEN")
    login(token=hf_token)
    logging.getLogger("transformers").setLevel(logging.CRITICAL)
    logging.getLogger("accelerate").setLevel(logging.CRITICAL)
    warnings.filterwarnings("ignore")

    print("▶ Initializing BERTScore...")
    from bert_score import score
    _ = score(["test"], ["test"], lang="en", verbose=False)
    print("BERTScore ready ✅\n")

    current_train_coll = args.train_subset_coll

    for gen in range(1, args.num_generations + 1):
        print(f"=== Generation {gen} ===")

        iterative_training_and_generation(
            client=client,
            args=args,
            base_model_name=args.hf_model,
            start_generation=gen,
            num_generations=1,
            device="cuda" if torch.cuda.is_available() else "cpu"
        )

        current_train_coll = args.synthetic_coll  # aggiorna se serve

        print()

    # Add mongo_client to args object
    args.mongo_client = client
    

    print("▶ Evaluating all generations...")
    evaluate_multiple_generations(
        args=args,
        base_model_name=args.hf_model,
        start_generation=0,
        num_generations=args.num_generations + 1,  # +1 to include base model (gen 0)
        device="cuda" if torch.cuda.is_available() else "cpu",
        save_results=True,
        results_dir="evaluation_results"
    )

    print("▶ Exporting predictions...")
    export_all_generations_predictions(
        base_model_name=args.hf_model,
        mongo_client=client,
        db_name=args.db,
        test_coll=args.test_subset_coll,
        start_generation=1,
        end_generation=args.num_generations,
        device="cuda" if torch.cuda.is_available() else "cpu",
        output_dir="predictions_export",
        mongo_collection_prefix=args.predictions_coll_prefix
    )

    client.close()
    print("\n✅ All done.")

if __name__ == "__main__":
    main()