from uq_validator.factory import JudgmentFactory
from uq_validator.model_adapters import ConfigurationError
import json
import argparse
import os
import sys
from typing import Dict
import concurrent.futures
import threading
from tqdm import tqdm
from utils.dataset_utils import load_uq_dataset

def clean_evaluation_data(evaluation: Dict) -> Dict:
    """Remove verbose conversation data and prompts from evaluation results."""
    # Make a copy to avoid modifying the original
    cleaned = evaluation.copy()
    
    # Remove conversations and prompts
    if 'conversations' in cleaned:
        del cleaned['conversations']
    if 'prompt' in cleaned:
        del cleaned['prompt']
    
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

def main():
    parser = argparse.ArgumentParser(description='Evaluate model answers using composable LLM judging strategies')
    parser.add_argument('--input_file', required=True, help='Path to input JSONL file')
    parser.add_argument('--strategy', choices=['relevance', 'cycle_consistency', 'fact_check', 'final_answer', 
                        'correctness', 'vanilla', 'sequential'], default='sequential', help='Judgment strategy')
    parser.add_argument('--model', default='o3', help='Model to use')
    parser.add_argument('--samples', type=int, default=1, help='Number of samples')
    parser.add_argument('--turns', type=int, default=1, help='Number of turns')
    parser.add_argument('--resampling_voting', choices=['majority', 'unanimous'], default='majority', help='Voting method for resampling')
    parser.add_argument('--multi_turn_voting', choices=['majority', 'unanimous'], default='majority', 
                        help='Voting method for multi-turn decisions')
    parser.add_argument('--sequential_strategies', nargs='+', 
                        default=['cycle_consistency', 'fact_check', 'correctness'],
                        help='Strategies to use in sequence (for sequential strategy)')
    parser.add_argument('--max_workers', type=int, default=32, help='Maximum number of threads for parallel execution')
    args = parser.parse_args()
    
    # Create the appropriate strategy based on arguments
    if args.strategy == 'sequential':
        strategy, judge = JudgmentFactory.create_sequential(
            args.model, args.sequential_strategies, args.samples, args.turns, 
            args.resampling_voting, args.multi_turn_voting
        )
    elif args.strategy == 'relevance':
        strategy, judge = JudgmentFactory.create_relevance(
            args.model, args.samples, args.turns, args.resampling_voting, args.multi_turn_voting
        )
    elif args.strategy == 'cycle_consistency':
        strategy, judge = JudgmentFactory.create_cycle_consistency(
            args.model, args.samples, args.turns, args.resampling_voting, args.multi_turn_voting
        )
    elif args.strategy == 'fact_check':
        strategy, judge = JudgmentFactory.create_fact_check(
            args.model, args.samples, args.turns, args.resampling_voting, args.multi_turn_voting
        )
    elif args.strategy == 'final_answer':
        strategy, judge = JudgmentFactory.create_final_answer(
            args.model, args.samples, args.turns, args.resampling_voting, args.multi_turn_voting
        )
    elif args.strategy == 'correctness':
        strategy, judge = JudgmentFactory.create_correctness(
            args.model, args.samples, args.turns, args.resampling_voting, args.multi_turn_voting
        )
    elif args.strategy == 'vanilla':
        strategy, judge = JudgmentFactory.create_vanilla(
            args.model, args.samples, args.turns, args.resampling_voting, args.multi_turn_voting
        )
    
    # Read input file
    if not os.path.exists(args.input_file):
        print(f"❌ Input file not found: {args.input_file}")
        sys.exit(1)
        
    with open(args.input_file, 'r', encoding='utf-8') as f:
        data = [json.loads(line) for line in f if line.strip()]
    
    # Load questions from Hugging Face dataset
    questions_list = load_uq_dataset()
    questions_dict = {q["question_id"]: q for q in questions_list}
    
    # Create result directory if it doesn't exist
    os.makedirs("result", exist_ok=True)
    # Construct output filename from input filename and parameters (use only the basename)
    input_basename = os.path.basename(os.path.splitext(args.input_file)[0])
    output_filename = f"{input_basename}_strategy-{args.strategy}_model-{args.model}_resampling-{args.resampling_voting}_multi_turn-{args.multi_turn_voting}_turns-{args.turns}_samples-{args.samples}.jsonl"
    output_path = f"result/{output_filename}"

    # Load existing results if file exists
    existing_results = {}
    if os.path.exists(output_path):
        with open(output_path, 'r', encoding='utf-8') as f:
            for line in f:
                result = json.loads(line)
                # Use (question_id, model_name) as key to handle multiple models per question
                result_key = (result['question_id'], result.get('model_name', ''))
                existing_results[result_key] = result

    # Evaluate each question-answer pair and append new results
    
    items_to_process = []
    for item in data:
        question_id = item["question_id"]
        model_name = item.get("model_name", "")
        result_key = (question_id, model_name)
        
        if result_key in existing_results:
            print(f"Skipping question {question_id} (model: {model_name}) - already evaluated")
            continue
        
        if question_id not in questions_dict:
            print(f"Skipping question {question_id} (model: {model_name}) - question metadata not found in UQ dataset")
            continue

        question = questions_dict[question_id]
        answer = item["answer"]
        if '</think>' in answer:
            answer = answer.split('</think>')[1]
        items_to_process.append((question_id, model_name, question, answer))

    if not items_to_process:
        print("No new questions to evaluate.")
        return


    file_lock = threading.Lock()
    processed_count = 0
    
    # Use ThreadPoolExecutor for parallel execution
    max_workers = min(args.max_workers, len(items_to_process))

    print(f"Starting evaluation for {len(items_to_process)} questions using up to {max_workers} workers...")
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit tasks
        future_to_qid = {executor.submit(evaluate_item, item_data, strategy, judge, file_lock, output_path): (item_data[0], item_data[1]) for item_data in items_to_process}
        
        # Process results as they complete with tqdm progress bar
        for future in tqdm(concurrent.futures.as_completed(future_to_qid), total=len(items_to_process), desc="Evaluating"):
            qid_model = future_to_qid[future]
            try:
                result_key, success = future.result()
                if success:
                    processed_count += 1
                    # Optional: print progress within the loop if needed, though tqdm handles overall progress
                    # print(f"Evaluated question {result_key[0]} (model: {result_key[1]})")
                else:
                    # Error was already printed in evaluate_item
                    pass 
            except Exception as exc:
                print(f'Question {qid_model[0]} (model: {qid_model[1]}) generated an exception: {exc}')

    print(f"Evaluation complete. {processed_count}/{len(items_to_process)} new questions evaluated. Results saved to {output_path}")

if __name__ == "__main__":
    main()