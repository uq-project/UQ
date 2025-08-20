"""
Command-line interface for UQ Validator.
"""

import json
import argparse
import os
import sys
from typing import Dict
import concurrent.futures
import threading
from tqdm import tqdm

from .factory import JudgmentFactory
from .model_adapters import ConfigurationError

try:
    from utils.dataset_utils import load_uq_dataset
except ImportError:
    # Fallback for when utils is not available
    def load_uq_dataset():
        raise ImportError("utils.dataset_utils not found. Please ensure the UQ dataset utilities are available.")


def clean_evaluation_data(evaluation: Dict) -> Dict:
    """Remove verbose conversation data from evaluation results."""
    # Make a copy to avoid modifying the original
    cleaned = evaluation.copy()
    
    # Remove conversations
    if 'conversations' in cleaned:
        del cleaned['conversations']
    
    # Clean sample results recursively
    if 'sample_results' in cleaned:
        cleaned['sample_results'] = [clean_evaluation_data(r) for r in cleaned['sample_results']]
    
    # For sequential judgment, clean each step
    if 'step_results' in cleaned:
        cleaned['step_results'] = [clean_evaluation_data(r) for r in cleaned['step_results']]
    
    return cleaned


def evaluate_item(item_data, strategy, judge, file_lock, output_path):
    """Evaluates a single item and writes the result to the output file."""
    question_id, model_name, question, answer = item_data
    try:
        evaluation = strategy.judge(question, answer, judge)
        # Clean the evaluation data before saving
        cleaned_evaluation = clean_evaluation_data(evaluation)
        result = {
            'question_id': question_id,
            'model_name': model_name,
            'evaluation': cleaned_evaluation
        }
        
        # Acquire lock to safely write to the file
        with file_lock:
            with open(output_path, 'a', encoding='utf-8') as f:
                f.write(json.dumps(result, ensure_ascii=False) + '\n')
        return (question_id, model_name), True # Indicate success
    except Exception as e:
        print(f"Error evaluating question {question_id} (model: {model_name}): {e}")
        return (question_id, model_name), False # Indicate failure


def create_parser() -> argparse.ArgumentParser:
    """Create command line argument parser."""
    parser = argparse.ArgumentParser(
        description='Evaluate model answers using composable LLM judging strategies'
    )
    
    parser.add_argument('--input_file', required=True, 
                       help='Path to input JSONL file with model answers')
    parser.add_argument('--use_huggingface', action='store_true', default=True,
                       help='Use UQ dataset from Hugging Face (default: True)')
    parser.add_argument('--local_dataset', 
                       help='Path to local dataset file (overrides Hugging Face)')
    parser.add_argument('--strategy', 
                       choices=['relevance', 'cycle_consistency', 'fact_check', 'final_answer', 
                               'correctness', 'vanilla', 'sequential'], 
                       default='sequential', help='Judgment strategy')
    parser.add_argument('--model', default='gpt-4o', 
                       help='Judge model to use (e.g., gpt-4o, claude-3-5-sonnet-20241022, gemini-1.5-pro)')
    parser.add_argument('--samples', type=int, default=1, 
                       help='Number of samples for repeated sampling')
    parser.add_argument('--turns', type=int, default=1, 
                       help='Number of turns for multi-turn evaluation')
    parser.add_argument('--resampling_voting', choices=['majority', 'unanimous'], 
                       default='majority', help='Voting method for resampling')
    parser.add_argument('--multi_turn_voting', choices=['majority', 'unanimous'], 
                       default='majority', help='Voting method for multi-turn decisions')
    parser.add_argument('--sequential_strategies', nargs='+', 
                       default=['cycle_consistency', 'fact_check', 'correctness'],
                       help='Strategies to use in sequence (for sequential strategy)')
    parser.add_argument('--max_workers', type=int, default=16, 
                       help='Maximum number of threads for parallel execution')
    parser.add_argument('--output_dir', default='result',
                       help='Directory to save results')
    
    return parser


def load_data(input_file: str) -> list:
    """Load input data from JSONL file."""
    if not os.path.exists(input_file):
        print(f"❌ Input file not found: {input_file}")
        sys.exit(1)
        
    with open(input_file, 'r', encoding='utf-8') as f:
        return [json.loads(line) for line in f if line.strip()]


def load_questions_dict(use_huggingface: bool = True, local_dataset: str = None) -> dict:
    """Load questions dictionary from dataset."""
    if local_dataset:
        print(f"📁 Loading dataset from local file: {local_dataset}")
        if not os.path.exists(local_dataset):
            print(f"❌ Local dataset file not found: {local_dataset}")
            sys.exit(1)
            
        questions_dict = {}
        with open(local_dataset, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    q = json.loads(line)
                    questions_dict[q["question_id"]] = q
        return questions_dict
    
    elif use_huggingface:
        print("🤗 Loading UQ dataset from Hugging Face...")
        try:
            questions_list = load_uq_dataset()
            return {q["question_id"]: q for q in questions_list}
        except Exception as e:
            print(f"❌ Failed to load UQ dataset from Hugging Face: {e}")
            print("Try using --local_dataset to specify a local dataset file")
            sys.exit(1)
    
    else:
        print("❌ No dataset specified. Use --use_huggingface or --local_dataset")
        sys.exit(1)


def create_strategy_and_judge(args):
    """Create the appropriate strategy and judge based on arguments."""
    if args.strategy == 'sequential':
        return JudgmentFactory.create_sequential(
            args.model, args.sequential_strategies, args.samples, args.turns, 
            args.resampling_voting, args.multi_turn_voting
        )
    elif args.strategy == 'relevance':
        return JudgmentFactory.create_relevance(
            args.model, args.samples, args.turns, args.resampling_voting, args.multi_turn_voting
        )
    elif args.strategy == 'cycle_consistency':
        return JudgmentFactory.create_cycle_consistency(
            args.model, args.samples, args.turns, args.resampling_voting, args.multi_turn_voting
        )
    elif args.strategy == 'fact_check':
        return JudgmentFactory.create_fact_check(
            args.model, args.samples, args.turns, args.resampling_voting, args.multi_turn_voting
        )
    elif args.strategy == 'final_answer':
        return JudgmentFactory.create_final_answer(
            args.model, args.samples, args.turns, args.resampling_voting, args.multi_turn_voting
        )
    elif args.strategy == 'correctness':
        return JudgmentFactory.create_correctness(
            args.model, args.samples, args.turns, args.resampling_voting, args.multi_turn_voting
        )
    elif args.strategy == 'vanilla':
        return JudgmentFactory.create_vanilla(
            args.model, args.samples, args.turns, args.resampling_voting, args.multi_turn_voting
        )
    else:
        raise ValueError(f"Unknown strategy: {args.strategy}")


def main():
    """Main CLI entry point."""
    parser = create_parser()
    args = parser.parse_args()
    
    try:
        # Create the appropriate strategy based on arguments
        strategy, judge = create_strategy_and_judge(args)
        
        # Read input file
        data = load_data(args.input_file)
        
        # Load questions dictionary for metadata
        questions_dict = load_questions_dict(args.use_huggingface, args.local_dataset)
        
    except ConfigurationError as e:
        print(e)
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error during setup: {e}")
        sys.exit(1)
    
    # Create result directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Construct output filename from input filename and parameters
    output_filename = (f"{os.path.splitext(args.input_file)[0]}_strategy-{args.strategy}_"
                      f"model-{args.model}_resampling-{args.resampling_voting}_"
                      f"multi_turn-{args.multi_turn_voting}_turns-{args.turns}_"
                      f"samples-{args.samples}.jsonl")
    output_path = os.path.join(args.output_dir, output_filename)

    # Load existing results if file exists
    existing_results = {}
    if os.path.exists(output_path):
        with open(output_path, 'r', encoding='utf-8') as f:
            for line in f:
                result = json.loads(line)
                # Use (question_id, model_name) as key to handle multiple models per question
                result_key = (result['question_id'], result.get('model_name', ''))
                existing_results[result_key] = result

    # Prepare items to process
    items_to_process = []
    for item in data:
        question_id = item["question_id"]
        model_name = item.get("model_name", "")
        result_key = (question_id, model_name)
        
        if result_key in existing_results:
            print(f"Skipping question {question_id} (model: {model_name}) - already evaluated")
            continue
        
        if question_id not in questions_dict:
            print(f"Skipping question {question_id} (model: {model_name}) - "
                  f"question metadata not found in dataset")
            continue

        question = questions_dict[question_id]
        answer = item["answer"]
        if '</think>' in answer:
            answer = answer.split('</think>')[1]
        items_to_process.append((question_id, model_name, question, answer))

    if not items_to_process:
        print("No new questions to evaluate.")
        return

    # Process items with parallel execution
    file_lock = threading.Lock()
    processed_count = 0
    
    # Use ThreadPoolExecutor for parallel execution
    max_workers = min(args.max_workers, len(items_to_process))

    print(f"Starting evaluation for {len(items_to_process)} questions using up to {max_workers} workers...")
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit tasks
        future_to_qid = {
            executor.submit(evaluate_item, item_data, strategy, judge, file_lock, output_path): 
            (item_data[0], item_data[1]) for item_data in items_to_process
        }
        
        # Process results as they complete with tqdm progress bar
        for future in tqdm(concurrent.futures.as_completed(future_to_qid), 
                          total=len(items_to_process), desc="Evaluating"):
            qid_model = future_to_qid[future]
            try:
                result_key, success = future.result()
                if success:
                    processed_count += 1
                else:
                    # Error was already printed in evaluate_item
                    pass 
            except Exception as exc:
                print(f'Question {qid_model[0]} (model: {qid_model[1]}) generated an exception: {exc}')

    print(f"Evaluation complete. {processed_count}/{len(items_to_process)} "
          f"new questions evaluated. Results saved to {output_path}")


def cli():
    """Entry point for CLI when installed as package."""
    main()

if __name__ == "__main__":
    main()