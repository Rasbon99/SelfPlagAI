"""
Type-Token Ratio Analysis Script for Multi-Generation Evaluation Results

This script analyzes the type-token ratio (TTR) of answers across different generations
from the evaluation results directory. It compares lexical diversity between generations
for the same question IDs.

Type-Token Ratio (TTR) = Number of unique tokens / Total number of tokens
Higher TTR indicates greater lexical diversity.
"""

import json
import os
import pandas as pd
import numpy as np
from pathlib import Path
from collections import Counter, defaultdict
from transformers import AutoTokenizer
import re
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm
from dotenv import load_dotenv
import bson

# Import the tokenizer setup from train_utils and db_utils
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from db_utils import get_mongo_client, insert_dataframe_to_mongo, drop_collection

# Load environment variables
load_dotenv('key.env')

class TypeTokenRatioAnalyzer:
    """Analyzes Type-Token Ratio across multiple generations of model predictions."""
    
    def __init__(self, 
                 evaluation_results_dir: str = "evaluation_results",
                 base_model_name: str = "microsoft/Phi-3-mini-128k-instruct",
                 mongo_client=None,
                 db_name: str = "squadv2"):
        """
        Initialize the analyzer.
        
        Args:
            evaluation_results_dir: Directory containing evaluation JSON files
            base_model_name: Base model name for tokenizer initialization
            mongo_client: MongoDB client instance (optional)
            db_name: MongoDB database name
        """
        self.results_dir = Path(evaluation_results_dir)
        self.base_model_name = base_model_name
        self.mongo_client = mongo_client
        self.db_name = db_name
        
        # Initialize tokenizer (same as in train_utils.py)
        print("🔤 Initializing tokenizer...")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(base_model_name)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            print("✅ Tokenizer initialized successfully")
            self.use_model_tokenizer = True
        except Exception as e:
            print(f"⚠️ Failed to load model tokenizer: {e}")
            print("   Falling back to simple word tokenization")
            self.tokenizer = None
            self.use_model_tokenizer = False
        
        self.generation_data = {}
        self.ttr_results = {}
        
    def load_evaluation_results(self) -> Dict:
        """Load all evaluation results from JSON files."""
        print("📂 Loading evaluation results...")
        
        # Try to load comprehensive results first
        comprehensive_file = self.results_dir / "comprehensive_evaluation_results.json"
        if comprehensive_file.exists():
            print("   📊 Loading comprehensive evaluation results...")
            with open(comprehensive_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Extract generation data
            for key, value in data.items():
                if key.startswith('generation_'):
                    gen_num = int(key.split('_')[1])
                    self.generation_data[gen_num] = value
                    
        else:
            # Load individual generation files
            print("   📄 Loading individual generation files...")
            for gen_file in self.results_dir.glob("evaluation_gen_*.json"):
                gen_num = int(gen_file.stem.split('_')[-1])
                print(f"      Loading generation {gen_num}...")
                
                with open(gen_file, 'r', encoding='utf-8') as f:
                    self.generation_data[gen_num] = json.load(f)
        
        if not self.generation_data:
            raise FileNotFoundError("No evaluation results found in the specified directory")
            
        print(f"✅ Loaded data for {len(self.generation_data)} generations: {sorted(self.generation_data.keys())}")
        return self.generation_data
    
    def clean_text(self, text: str) -> str:
        """Clean text for better tokenization."""
        if not isinstance(text, str):
            return ""
            
        # Remove "No answer" patterns and extra whitespace
        text = re.sub(r'\(.*?\)', '', text)  # Remove parentheses content
        text = re.sub(r'No answer.*', '', text, flags=re.IGNORECASE)
        text = re.sub(r'\s+', ' ', text)  # Normalize whitespace
        return text.strip()
    
    def calculate_ttr_for_text(self, text: str) -> Dict[str, float]:
        """
        Calculate various TTR metrics for a given text.
        
        Args:
            text: Input text to analyze
            
        Returns:
            Dictionary with TTR metrics
        """
        if not text or text.strip() == "":
            return {
                'ttr': 0.0,
                'total_tokens': 0,
                'unique_tokens': 0,
                'standardized_ttr': 0.0,
                'root_ttr': 0.0
            }
        
        # Clean text
        cleaned_text = self.clean_text(text)
        if not cleaned_text:
            return {
                'ttr': 0.0,
                'total_tokens': 0,
                'unique_tokens': 0,
                'standardized_ttr': 0.0,
                'root_ttr': 0.0
            }
        
        # Tokenize using the model's tokenizer or fallback to simple tokenization
        if self.use_model_tokenizer and self.tokenizer:
            tokens = self.tokenizer.tokenize(cleaned_text.lower())
            # Filter out special tokens
            tokens = [token for token in tokens if not token.startswith('<') and not token.startswith('[')]
        else:
            # Simple word tokenization as fallback
            import re
            tokens = re.findall(r'\b\w+\b', cleaned_text.lower())
            print("   Using simple word tokenization (fallback)")  # Debug message
        
        if len(tokens) == 0:
            return {
                'ttr': 0.0,
                'total_tokens': 0,
                'unique_tokens': 0,
                'standardized_ttr': 0.0,
                'root_ttr': 0.0
            }
        
        # Calculate metrics
        total_tokens = len(tokens)
        unique_tokens = len(set(tokens))
        
        # Standard TTR
        ttr = unique_tokens / total_tokens if total_tokens > 0 else 0.0
        
        # Standardized TTR (for comparison across different text lengths)
        # Uses a standard text length of 100 tokens
        standardized_length = min(100, total_tokens)
        if standardized_length > 0:
            sample_tokens = tokens[:standardized_length]
            standardized_ttr = len(set(sample_tokens)) / len(sample_tokens)
        else:
            standardized_ttr = 0.0
        
        # Root TTR (adjusted for text length)
        root_ttr = unique_tokens / np.sqrt(total_tokens) if total_tokens > 0 else 0.0
        
        return {
            'ttr': ttr,
            'total_tokens': total_tokens,
            'unique_tokens': unique_tokens,
            'standardized_ttr': standardized_ttr,
            'root_ttr': root_ttr
        }
    
    def analyze_generations(self) -> Dict:
        """Analyze TTR across all generations for each question ID."""
        print("🔍 Analyzing Type-Token Ratio across generations...")
        
        if not self.generation_data:
            self.load_evaluation_results()
        
        # Get the number of test examples (should be consistent across generations)
        test_size = next(iter(self.generation_data.values())).get('test_size', 0)
        print(f"   📊 Analyzing {test_size} examples across {len(self.generation_data)} generations")
        
        results = {
            'by_question': defaultdict(dict),  # TTR for each question across generations
            'by_generation': defaultdict(dict),  # Overall TTR stats per generation
            'summary_stats': {},
            'detailed_analysis': []
        }
        
        # Process each generation
        generation_progress = tqdm(
            sorted(self.generation_data.keys()), 
            desc="📈 Processing generations",
            unit="generation"
        )
        
        for gen_num in generation_progress:
            generation_progress.set_description(f"📈 Processing Generation {gen_num}")
            
            gen_data = self.generation_data[gen_num]
            predictions = gen_data.get('predictions', [])
            references = gen_data.get('references', [])
            
            gen_ttr_scores = []
            gen_token_counts = []
            gen_unique_counts = []
            
            # Process each prediction in this generation
            for idx, prediction in enumerate(predictions):
                ttr_metrics = self.calculate_ttr_for_text(prediction)
                
                # Store by question ID
                results['by_question'][idx][f'gen_{gen_num}'] = {
                    'prediction': prediction,
                    'reference': references[idx] if idx < len(references) else "N/A",
                    'ttr_metrics': ttr_metrics
                }
                
                # Collect generation-level statistics
                if ttr_metrics['total_tokens'] > 0:  # Only include non-empty responses
                    gen_ttr_scores.append(ttr_metrics['ttr'])
                    gen_token_counts.append(ttr_metrics['total_tokens'])
                    gen_unique_counts.append(ttr_metrics['unique_tokens'])
            
            # Calculate generation-level statistics
            if gen_ttr_scores:
                results['by_generation'][gen_num] = {
                    'mean_ttr': np.mean(gen_ttr_scores),
                    'std_ttr': np.std(gen_ttr_scores),
                    'median_ttr': np.median(gen_ttr_scores),
                    'min_ttr': np.min(gen_ttr_scores),
                    'max_ttr': np.max(gen_ttr_scores),
                    'mean_tokens': np.mean(gen_token_counts),
                    'mean_unique_tokens': np.mean(gen_unique_counts),
                    'total_predictions': len(gen_ttr_scores),
                    'empty_responses': len(predictions) - len(gen_ttr_scores)
                }
            
            generation_progress.set_postfix({
                'Predictions': len(predictions),
                'Valid TTR': len(gen_ttr_scores),
                'Avg TTR': f"{np.mean(gen_ttr_scores):.3f}" if gen_ttr_scores else "0.000"
            })
        
        generation_progress.close()
        
        # Calculate cross-generation statistics
        self._calculate_cross_generation_stats(results)
        
        self.ttr_results = results
        print("✅ TTR analysis completed!")
        return results
    
    def _calculate_cross_generation_stats(self, results: Dict):
        """Calculate statistics comparing TTR across generations."""
        print("📊 Calculating cross-generation statistics...")
        
        # Questions that have predictions in all generations
        all_generations = set(self.generation_data.keys())
        complete_questions = []
        
        for question_id, gen_data in results['by_question'].items():
            if set(gen_data.keys()) >= {f'gen_{g}' for g in all_generations}:
                complete_questions.append(question_id)
        
        print(f"   📋 {len(complete_questions)} questions have predictions in all {len(all_generations)} generations")
        
        # Calculate TTR evolution patterns
        ttr_evolution = []
        consistency_scores = []
        
        for question_id in complete_questions:
            question_ttrs = []
            for gen_num in sorted(all_generations):
                gen_key = f'gen_{gen_num}'
                if gen_key in results['by_question'][question_id]:
                    ttr = results['by_question'][question_id][gen_key]['ttr_metrics']['ttr']
                    question_ttrs.append(ttr)
            
            if len(question_ttrs) == len(all_generations):
                ttr_evolution.append(question_ttrs)
                # Calculate consistency (inverse of standard deviation)
                consistency = 1 / (np.std(question_ttrs) + 0.001)  # Add small epsilon
                consistency_scores.append(consistency)
        
        # Summary statistics
        results['summary_stats'] = {
            'total_questions': len(results['by_question']),
            'complete_questions': len(complete_questions),
            'total_generations': len(all_generations),
            'generations_analyzed': sorted(list(all_generations)),
            'avg_consistency': np.mean(consistency_scores) if consistency_scores else 0,
            'ttr_trends': self._analyze_ttr_trends(ttr_evolution, all_generations)
        }
    
    def _analyze_ttr_trends(self, ttr_evolution: List[List[float]], generations: List[int]) -> Dict:
        """Analyze trends in TTR across generations."""
        if not ttr_evolution:
            return {}
        
        ttr_matrix = np.array(ttr_evolution)
        trends = {}
        
        # Calculate mean TTR per generation
        mean_ttr_per_gen = np.mean(ttr_matrix, axis=0)
        
        # Calculate correlation with generation number
        from scipy.stats import pearsonr
        gen_numbers = sorted(list(generations))
        correlation, p_value = pearsonr(gen_numbers, mean_ttr_per_gen)
        
        trends = {
            'mean_ttr_per_generation': dict(zip(gen_numbers, mean_ttr_per_gen)),
            'overall_correlation': correlation,
            'correlation_p_value': p_value,
            'trend_direction': 'increasing' if correlation > 0.1 else 'decreasing' if correlation < -0.1 else 'stable'
        }
        
        return trends
    
    def save_detailed_results(self, output_file: str = "ttr_analysis_results.json"):
        """Save detailed TTR analysis results to JSON."""
        if not self.ttr_results:
            print("⚠️ No results to save. Run analyze_generations() first.")
            return
        
        print(f"💾 Saving detailed results to {output_file}...")
        
        # Convert defaultdict to regular dict for JSON serialization
        results_for_json = {
            'by_question': dict(self.ttr_results['by_question']),
            'by_generation': dict(self.ttr_results['by_generation']),
            'summary_stats': self.ttr_results['summary_stats'],
            'analysis_metadata': {
                'base_model_name': self.base_model_name,
                'evaluation_results_dir': str(self.results_dir),
                'analysis_timestamp': pd.Timestamp.now().isoformat(),
                'tokenizer_vocab_size': len(self.tokenizer.vocab) if (self.use_model_tokenizer and self.tokenizer and hasattr(self.tokenizer, 'vocab')) else 'Simple word tokenization',
                'tokenization_method': 'model_tokenizer' if self.use_model_tokenizer else 'simple_word_tokenization'
            }
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results_for_json, f, indent=2, ensure_ascii=False)
        
        print(f"✅ Results saved to {output_file}")
    
    def _sanitize_for_mongo(self, obj):
        """Sanitize data for MongoDB insertion by converting problematic types."""
        if isinstance(obj, dict):
            # Handle dictionary - recursively sanitize all values
            sanitized_dict = {}
            for key, value in obj.items():
                # Ensure key is string and valid for MongoDB
                clean_key = str(key) if not isinstance(key, str) else key
                # MongoDB doesn't allow keys starting with $ or containing .
                clean_key = clean_key.replace('$', '_dollar_').replace('.', '_dot_')
                sanitized_dict[clean_key] = self._sanitize_for_mongo(value)
            return sanitized_dict
            
        elif isinstance(obj, list):
            # Handle list - recursively sanitize all items
            return [self._sanitize_for_mongo(item) for item in obj]
            
        elif isinstance(obj, np.ndarray):
            # Convert numpy arrays to lists
            return obj.tolist()
            
        elif isinstance(obj, (np.int64, np.int32, np.int16, np.int8)):
            # Convert numpy integers to Python int
            return int(obj)
            
        elif isinstance(obj, (np.float64, np.float32, np.float16)):
            # Convert numpy floats to Python float, handle NaN
            float_val = float(obj)
            return None if (pd.isna(float_val) or np.isnan(float_val) or np.isinf(float_val)) else float_val
            
        elif isinstance(obj, np.bool_):
            # Convert numpy boolean to Python bool
            return bool(obj)
            
        elif isinstance(obj, pd.Timestamp):
            # Convert pandas timestamp to ISO string
            return obj.isoformat()
            
        elif pd.isna(obj):  # Handle NaN/NaT values from pandas
            return None
            
        elif isinstance(obj, (int, float, str, bool, type(None))):
            # Handle standard Python types
            if isinstance(obj, float) and (np.isnan(obj) or np.isinf(obj)):
                return None
            return obj
            
        elif hasattr(obj, 'item'):  # Handle numpy scalars
            try:
                return self._sanitize_for_mongo(obj.item())
            except (ValueError, AttributeError):
                return str(obj)
                
        else:
            # Convert other types to string as fallback
            try:
                return str(obj)
            except Exception:
                return "<<unable_to_serialize>>"

    def save_to_mongodb(self, collection_name: str = "ttr_analisys"):
        """Save TTR analysis results to MongoDB."""
        if not self.ttr_results:
            print("⚠️ No results to save. Run analyze_generations() first.")
            return
            
        if not self.mongo_client:
            print("⚠️ No MongoDB client available. Cannot save to MongoDB.")
            return
        
        print(f"💾 Saving TTR analysis results to MongoDB...")
        
        # Convert defaultdict to regular dict and sanitize for MongoDB
        raw_results = {
            'by_question': dict(self.ttr_results['by_question']),
            'by_generation': dict(self.ttr_results['by_generation']),
            'summary_stats': self.ttr_results['summary_stats'],
            'analysis_metadata': {
                'base_model_name': self.base_model_name,
                'evaluation_results_dir': str(self.results_dir),
                'analysis_timestamp': pd.Timestamp.now().isoformat(),
                'tokenizer_vocab_size': len(self.tokenizer.vocab) if (self.use_model_tokenizer and self.tokenizer and hasattr(self.tokenizer, 'vocab')) else 'Simple word tokenization',
                'tokenization_method': 'model_tokenizer' if self.use_model_tokenizer else 'simple_word_tokenization',
                'analysis_type': 'type_token_ratio',
                'document_id': f"ttr_analysis_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}"
            }
        }
        
        # Sanitize the entire document for MongoDB compatibility
        results_for_mongo = self._sanitize_for_mongo(raw_results)
        
        try:
            # Insert the document into MongoDB
            collection = self.mongo_client[self.db_name][collection_name]
            
            # Check if collection exists and print some debug info
            print(f"   📊 Target database: {self.db_name}")
            print(f"   📁 Target collection: {collection_name}")
            print(f"   📋 Document size (approx): {len(str(results_for_mongo))} characters")
            
            # Test document validity before insertion
            import bson
            try:
                bson.encode(results_for_mongo)
                print("   ✅ Document passed BSON validation")
            except Exception as bson_error:
                print(f"   ❌ BSON validation failed: {bson_error}")
                return None
            
            result = collection.insert_one(results_for_mongo)
            
            print(f"✅ TTR analysis results saved to MongoDB:")
            print(f"   📊 Database: {self.db_name}")
            print(f"   📁 Collection: {collection_name}")
            print(f"   🆔 Document ID: {result.inserted_id}")
            
            # Verify the document was inserted
            doc_count = collection.count_documents({})
            print(f"   📊 Total documents in collection: {doc_count}")
            
            return result.inserted_id
            
        except Exception as e:
            print(f"❌ Error saving to MongoDB: {str(e)}")
            print(f"   Error type: {type(e).__name__}")
            
            # Additional debug info for InvalidDocument errors
            if "InvalidDocument" in str(type(e)):
                print("   🔍 InvalidDocument error - checking data types...")
                self._debug_document_types(raw_results)
            
            import traceback
            traceback.print_exc()
            return None
    
    def _debug_document_types(self, obj, path=""):
        """Debug helper to find problematic data types in the document."""
        if isinstance(obj, dict):
            for key, value in obj.items():
                self._debug_document_types(value, f"{path}.{key}" if path else key)
        elif isinstance(obj, list):
            for i, item in enumerate(obj[:5]):  # Check first 5 items only
                self._debug_document_types(item, f"{path}[{i}]")
        else:
            obj_type = type(obj)
            if obj_type not in (int, float, str, bool, type(None)):
                print(f"   🔍 Problematic type at {path}: {obj_type} = {str(obj)[:100]}")
                if hasattr(obj, 'dtype'):
                    print(f"      NumPy dtype: {obj.dtype}")
                if pd.isna(obj):
                    print(f"      Value is NaN/NaT")
    
    def create_comparison_dataframe(self) -> pd.DataFrame:
        """Create a DataFrame for easy comparison of TTR across generations."""
        if not self.ttr_results:
            print("⚠️ No results available. Run analyze_generations() first.")
            return pd.DataFrame()
        
        print("📊 Creating comparison DataFrame...")
        
        rows = []
        for question_id, gen_data in self.ttr_results['by_question'].items():
            row = {'question_id': question_id}
            
            for gen_key, data in gen_data.items():
                gen_num = gen_key.split('_')[1]
                ttr_metrics = data['ttr_metrics']
                
                row[f'gen_{gen_num}_ttr'] = ttr_metrics['ttr']
                row[f'gen_{gen_num}_tokens'] = ttr_metrics['total_tokens']
                row[f'gen_{gen_num}_unique'] = ttr_metrics['unique_tokens']
                row[f'gen_{gen_num}_prediction'] = data['prediction'][:100] + "..." if len(data['prediction']) > 100 else data['prediction']
                
                if question_id == 0:  # Store reference only once
                    row['reference'] = data['reference'][:100] + "..." if len(data['reference']) > 100 else data['reference']
            
            rows.append(row)
        
        df = pd.DataFrame(rows)
        print(f"✅ DataFrame created with {len(df)} questions")
        return df
    
    def generate_summary_report(self) -> str:
        """Generate a comprehensive summary report."""
        if not self.ttr_results:
            return "No results available. Run analyze_generations() first."
        
        report = []
        report.append("=" * 80)
        report.append("TYPE-TOKEN RATIO ANALYSIS REPORT")
        report.append("=" * 80)
        report.append("")
        
        # Basic statistics
        summary = self.ttr_results['summary_stats']
        report.append("📊 SUMMARY STATISTICS")
        report.append("-" * 40)
        report.append(f"Total Questions Analyzed: {summary['total_questions']}")
        report.append(f"Questions with All Generations: {summary['complete_questions']}")
        report.append(f"Generations Analyzed: {summary['generations_analyzed']}")
        report.append(f"Average Consistency Score: {summary['avg_consistency']:.4f}")
        report.append("")
        
        # Generation-level statistics
        report.append("📈 GENERATION-LEVEL TTR STATISTICS")
        report.append("-" * 50)
        report.append(f"{'Gen':<4} {'Mean TTR':<10} {'Std TTR':<10} {'Median':<10} {'Avg Tokens':<12} {'Unique Tokens':<12}")
        report.append("-" * 70)
        
        for gen_num in sorted(self.ttr_results['by_generation'].keys()):
            stats = self.ttr_results['by_generation'][gen_num]
            report.append(f"{gen_num:<4} {stats['mean_ttr']:<10.4f} {stats['std_ttr']:<10.4f} "
                         f"{stats['median_ttr']:<10.4f} {stats['mean_tokens']:<12.1f} {stats['mean_unique_tokens']:<12.1f}")
        
        report.append("")
        
        # Trend analysis
        if 'ttr_trends' in summary and summary['ttr_trends']:
            trends = summary['ttr_trends']
            report.append("📊 TTR TREND ANALYSIS")
            report.append("-" * 30)
            report.append(f"Overall Trend: {trends['trend_direction'].upper()}")
            report.append(f"Correlation with Generation Number: {trends['overall_correlation']:.4f}")
            report.append(f"Statistical Significance (p-value): {trends['correlation_p_value']:.4f}")
            report.append("")
            
            report.append("Mean TTR by Generation:")
            for gen, mean_ttr in trends['mean_ttr_per_generation'].items():
                report.append(f"   Generation {gen}: {mean_ttr:.4f}")
            report.append("")
        
        # Top/Bottom performing questions
        if self.ttr_results['by_question']:
            report.append("🔍 NOTABLE EXAMPLES")
            report.append("-" * 25)
            
            # Find questions with highest/lowest TTR variation
            ttr_variations = []
            for q_id, gen_data in self.ttr_results['by_question'].items():
                ttrs = [data['ttr_metrics']['ttr'] for data in gen_data.values()]
                if len(ttrs) > 1:
                    variation = np.std(ttrs)
                    ttr_variations.append((q_id, variation, ttrs, gen_data))
            
            if ttr_variations:
                # Sort by variation (descending)
                ttr_variations.sort(key=lambda x: x[1], reverse=True)
                
                report.append("Questions with HIGHEST TTR variation across generations:")
                for i, (q_id, variation, ttrs, gen_data) in enumerate(ttr_variations[:3]):
                    report.append(f"\nQuestion {q_id} (Variation: {variation:.4f}):")
                    for gen_key, data in gen_data.items():
                        gen_num = gen_key.split('_')[1]
                        ttr = data['ttr_metrics']['ttr']
                        pred = data['prediction'][:80] + "..." if len(data['prediction']) > 80 else data['prediction']
                        report.append(f"   Gen {gen_num} (TTR: {ttr:.3f}): {pred}")
                
                report.append("\nQuestions with LOWEST TTR variation (most consistent):")
                for i, (q_id, variation, ttrs, gen_data) in enumerate(ttr_variations[-3:]):
                    report.append(f"\nQuestion {q_id} (Variation: {variation:.4f}):")
                    for gen_key, data in gen_data.items():
                        gen_num = gen_key.split('_')[1]
                        ttr = data['ttr_metrics']['ttr']
                        pred = data['prediction'][:80] + "..." if len(data['prediction']) > 80 else data['prediction']
                        report.append(f"   Gen {gen_num} (TTR: {ttr:.3f}): {pred}")
        
        report.append("")
        report.append("=" * 80)
        
        return "\n".join(report)


def main():
    """Main function to run the TTR analysis."""
    print("🚀 Starting Type-Token Ratio Analysis")
    print("=" * 50)
    
    # Initialize MongoDB client
    mongo_client = None
    try:
        print("🔍 Loading environment variables...")
        username = os.getenv('MONGO_USERNAME')
        password = os.getenv('MONGO_PASSWORD')
        print(f"   Username found: {'Yes' if username else 'No'}")
        print(f"   Password found: {'Yes' if password else 'No'}")
        
        if username and password:
            print("🔗 Connecting to MongoDB...")
            mongo_client = get_mongo_client(username, password)
            print("✅ MongoDB connection established")
            
            # Test the connection
            try:
                mongo_client.admin.command('ping')
                print("✅ MongoDB ping successful")
            except Exception as ping_error:
                print(f"⚠️ MongoDB ping failed: {str(ping_error)}")
        else:
            print("⚠️ MongoDB credentials not found in environment variables")
            print("   TTR results will be saved to local files only")
    except Exception as e:
        print(f"⚠️ MongoDB connection failed: {str(e)}")
        print("   TTR results will be saved to local files only")
        mongo_client = None
    
    # Initialize analyzer
    print("🔧 Initializing TTR analyzer...")
    analyzer = TypeTokenRatioAnalyzer(
        evaluation_results_dir="evaluation_results",
        base_model_name="microsoft/Phi-3-mini-128k-instruct",
        mongo_client=mongo_client,
        db_name="squadv2"
    )
    print("✅ TTR analyzer initialized successfully")
    
    try:
        # Load and analyze data
        analyzer.load_evaluation_results()
        results = analyzer.analyze_generations()
        
        # Save detailed results in ttr_analysis directory
        ttr_results_dir = Path("ttr_analysis")
        ttr_results_dir.mkdir(exist_ok=True)
        
        # Save detailed JSON results in ttr_analysis directory
        detailed_json_path = ttr_results_dir / "ttr_analysis_detailed.json"
        analyzer.save_detailed_results(str(detailed_json_path))
        
        # Save backup copy in current directory
        analyzer.save_detailed_results("ttr_analysis_detailed.json")
        
        # Save to MongoDB if available
        if mongo_client:
            mongodb_doc_id = analyzer.save_to_mongodb("ttr_results")
        
        # Create comparison DataFrame and save as CSV
        df = analyzer.create_comparison_dataframe()
        if not df.empty:
            csv_path = ttr_results_dir / "ttr_comparison_table.csv"
            df.to_csv(csv_path, index=False)
            df.to_csv("ttr_comparison_table.csv", index=False)  # Backup copy
            print(f"📊 Comparison table saved to {csv_path}")
        
        # Generate and save summary report
        report = analyzer.generate_summary_report()
        report_path = ttr_results_dir / "ttr_analysis_report.txt"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report)
        with open("ttr_analysis_report.txt", 'w', encoding='utf-8') as f:  # Backup copy
            f.write(report)
        print(f"📋 Summary report saved to {report_path}")
        
        # Print summary to console
        print("\n" + report)
        
        print("\n🎉 Type-Token Ratio analysis completed successfully!")
        print("📁 Output files:")
        print("   📊 ttr_analysis/ttr_analysis_detailed.json")
        print("   📋 ttr_analysis/ttr_analysis_report.txt")
        print("   📈 ttr_analysis/ttr_comparison_table.csv")
        if mongo_client:
            print("   🗄️ MongoDB: squadv2.ttr_results collection")
        print("   💾 Backup copies also saved in current directory")
        
    except Exception as e:
        print(f"❌ Error during analysis: {str(e)}")
        import traceback
        traceback.print_exc()
    finally:
        # Close MongoDB connection if it was established
        if mongo_client:
            try:
                mongo_client.close()
                print("🔐 MongoDB connection closed")
            except:
                pass


if __name__ == "__main__":
    main()
