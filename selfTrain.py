# Standard library
import os
import argparse
import logging
import warnings

# Third‑party libraries — data processing & environment
import torch
from dotenv import load_dotenv
from huggingface_hub import login

# Local modules — database & pipeline utilities
from db_utils import (
    get_mongo_client,
    read_original_and_create_subset,
    save_synthetic_dataset_to_mongo,
)

from train_utils import (
    iterative_training_and_generation,
    evaluate_multiple_generations,
    export_all_generations_predictions,
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
    parser.add_argument("--subset-coll", type=str, default="squad_subset_0_5pct",
                        help="Collection for the sampled subset")
    parser.add_argument("--synthetic-coll", type=str, default="squad_synthetic_train",
                        help="Collection for synthetic train generations")
    parser.add_argument("--metadata-coll", type=str, default="synthetic_metadata",
                        help="Collection for synthetic metadata")
    return parser.parse_args()

def main():
    args = parse_args()
    load_dotenv("key.env")
    client = get_mongo_client(
        username=os.getenv("USERNAME"),
        password=os.getenv("PASSWORD")
    )

    print("▶ Creating initial subset in MongoDB...")
    read_original_and_create_subset(
        client=client,
        original_coll=args.orig_train_coll,
        subset_coll=args.subset_coll,
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

    current_train_coll = args.subset_coll

    for gen in range(1, args.num_generations + 1):
        print(f"=== Generation {gen} ===")

        # Iterative training & synthetic generation (internamente gestisce i dati)
        synthetic_ds = iterative_training_and_generation(
            client=client,
            args=args,
            base_model_name=args.hf_model,
            start_generation=gen,
            num_generations=1,
            device="cuda" if torch.cuda.is_available() else "cpu"
        )

        print("Saving synthetic train and metadata to MongoDB...")
        save_synthetic_dataset_to_mongo(
            client=client,
            synthetic_dataset=synthetic_ds,
            gen_number=gen,
            train_coll=args.synthetic_coll,
            metadata_coll=args.metadata_coll,
            db_name=args.db
        )

        current_train_coll = args.synthetic_coll  # se vuoi gestire meglio le generazioni cambia qui

        print()

    print("▶ Evaluating all generations...")
    evaluate_multiple_generations(
        args=args,
        base_model_name=args.hf_model,
        start_generation=1,
        num_generations=args.num_generations,
        device="cuda" if torch.cuda.is_available() else "cpu",
        mongo_client=client,
        db_name=args.db,
        test_coll=args.test_coll,
        synthetic_coll=args.synthetic_coll,
        save_results=True,
        results_dir="evaluation_results_squad05"
    )

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