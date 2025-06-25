import os
import json
import argparse
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv
from db_utils import get_mongo_client, insert_dataframe_to_mongo, drop_collection, read_collection

def load_json_files_to_dataframe(directory_path, file_pattern="*.json"):
    """
    Load all JSON files from a directory into a pandas DataFrame.
    
    Args:
        directory_path: Path to directory containing JSON files
        file_pattern: Pattern to match files (default: "*.json")
    
    Returns:
        pd.DataFrame: DataFrame containing all loaded data
    """
    directory = Path(directory_path)
    if not directory.exists():
        print(f"❌ Directory {directory_path} does not exist")
        return pd.DataFrame()
    
    json_files = list(directory.glob(file_pattern))
    if not json_files:
        print(f"❌ No JSON files found in {directory_path}")
        return pd.DataFrame()

    # Sort files to ensure proper order (especially for generation files)
    json_files.sort(key=lambda x: x.name)
    
    all_data = []
    
    for json_file in json_files:
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Handle different JSON structures
            if isinstance(data, dict):
                # Add metadata about the file
                data['_source_file'] = json_file.name
                data['_source_directory'] = str(directory_path)
                all_data.append(data)
                
            elif isinstance(data, list):
                # If it's a list, add metadata to each item
                for item in data:
                    if isinstance(item, dict):
                        item['_source_file'] = json_file.name
                        item['_source_directory'] = str(directory_path)
                        all_data.append(item)
                    else:
                        # Handle primitive types in list
                        record = {
                            'data': item,
                            '_source_file': json_file.name,
                            '_source_directory': str(directory_path)
                        }
                        all_data.append(record)
            else:
                # Handle primitive types
                record = {
                    'data': data,
                    '_source_file': json_file.name,
                    '_source_directory': str(directory_path)
                }
                all_data.append(record)
                
            print(f"✅ Loaded {json_file.name}")
            
        except json.JSONDecodeError as e:
            print(f"❌ Error decoding JSON from {json_file.name}: {e}")
        except Exception as e:
            print(f"❌ Error loading {json_file.name}: {e}")
    
    if all_data:
        return pd.DataFrame(all_data)
    else:
        return pd.DataFrame()

def load_evaluation_results(client, db_name, results_dir="evaluation_results", collection_name="evaluation_results", clear_existing=False):
    """
    Load evaluation results into MongoDB using db_utils functions.
    """
    print(f"▶ Loading evaluation results from {results_dir}...")
    
    if clear_existing:
        print(f"▶ Clearing existing collection {collection_name}...")
        drop_collection(client, collection_name, db_name)
    
    df = load_json_files_to_dataframe(results_dir)
    
    if df.empty:
        print(f"❌ No data loaded from {results_dir}")
        return 0
    
    print(f"▶ Inserting {len(df)} records into MongoDB collection {collection_name}...")
    insert_dataframe_to_mongo(client, df, collection_name, db_name)
    
    return len(df)

def load_predictions_export(client, db_name, predictions_dir="predictions_export", collection_name="predictions_export", clear_existing=False):
    """
    Load prediction exports into MongoDB using db_utils functions.
    """
    print(f"▶ Loading prediction exports from {predictions_dir}...")
    
    if clear_existing:
        print(f"▶ Clearing existing collection {collection_name}...")
        drop_collection(client, collection_name, db_name)
    
    df = load_json_files_to_dataframe(predictions_dir)
    
    if df.empty:
        print(f"❌ No data loaded from {predictions_dir}")
        return 0
    
    print(f"▶ Inserting {len(df)} records into MongoDB collection {collection_name}...")
    insert_dataframe_to_mongo(client, df, collection_name, db_name)
    
    return len(df)

def parse_args():
    parser = argparse.ArgumentParser(
        description="Load evaluation_results and predictions_export directories to MongoDB"
    )
    parser.add_argument("--db", type=str, default="squadv2",
                        help="MongoDB database name")
    parser.add_argument("--evaluation-dir", type=str, default="evaluation_results",
                        help="Path to evaluation results directory")
    parser.add_argument("--predictions-dir", type=str, default="predictions_export",
                        help="Path to predictions export directory")
    parser.add_argument("--evaluation-collection", type=str, default="evaluation_results",
                        help="Collection name for evaluation results")
    parser.add_argument("--predictions-collection", type=str, default="predictions_export",
                        help="Collection name for predictions export")
    parser.add_argument("--clear-existing", action="store_true",
                        help="Clear existing collections before loading")
    parser.add_argument("--load-evaluation", action="store_true", default=True,
                        help="Load evaluation results (default: True)")
    parser.add_argument("--load-predictions", action="store_true", default=True,
                        help="Load predictions export (default: True)")
    parser.add_argument("--skip-evaluation", action="store_true",
                        help="Skip loading evaluation results")
    parser.add_argument("--skip-predictions", action="store_true",
                        help="Skip loading predictions export")
    return parser.parse_args()

def main():
    args = parse_args()
    load_dotenv("key.env")
    
    username = os.getenv("MONGO_USERNAME")
    password = os.getenv("MONGO_PASSWORD")
    
    if not username or not password:
        print("❌ USERNAME or PASSWORD not found in .env file.")
        return
    
    print(f"Connecting to MongoDB as {username}...")
    try:
        client = get_mongo_client(username, password)
        client.admin.command("ping")
        print("✅ Connected to MongoDB")
    except Exception as e:
        print(f"❌ MongoDB connection failed: {e}")
        return
    
    total_loaded = 0
    
    # Load evaluation results
    if not args.skip_evaluation and os.path.exists(args.evaluation_dir):
        eval_count = load_evaluation_results(
            client=client,
            db_name=args.db,
            results_dir=args.evaluation_dir,
            collection_name=args.evaluation_collection,
            clear_existing=args.clear_existing
        )
        total_loaded += eval_count
    elif args.skip_evaluation:
        print("▶ Skipping evaluation results loading")
    else:
        print(f"❌ Evaluation directory {args.evaluation_dir} not found")
    
    # Load predictions export
    if not args.skip_predictions and os.path.exists(args.predictions_dir):
        pred_count = load_predictions_export(
            client=client,
            db_name=args.db,
            predictions_dir=args.predictions_dir,
            collection_name=args.predictions_collection,
            clear_existing=args.clear_existing
        )
        total_loaded += pred_count
    elif args.skip_predictions:
        print("▶ Skipping predictions export loading")
    else:
        print(f"❌ Predictions directory {args.predictions_dir} not found")
    
    print(f"\n✅ Total documents loaded: {total_loaded}")
    
    # Print collection stats using db_utils
    print("\n📊 Collection Statistics:")
    try:
        if not args.skip_evaluation:
            eval_data = read_collection(client, args.evaluation_collection, args.db)
            print(f"  - {args.evaluation_collection}: {len(eval_data)} documents")
        
        if not args.skip_predictions:
            pred_data = read_collection(client, args.predictions_collection, args.db)
            print(f"  - {args.predictions_collection}: {len(pred_data)} documents")
    except Exception as e:
        print(f"❌ Error reading collection stats: {e}")
    
    client.close()
    print("\n✅ All done.")

if __name__ == "__main__":
    main()