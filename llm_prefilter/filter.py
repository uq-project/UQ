import pandas as pd
import os
import openai
import time
from tqdm import tqdm
import argparse
import json
import re
import threading
import concurrent.futures
from dotenv import load_dotenv
load_dotenv("key.env")

# Configure OpenAI API key
client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
# Thread-local storage for OpenAI client
thread_local = threading.local()


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate and filter questions using OpenAI API')
    parser.add_argument('--input', type=str, default='__collector_questions.jsonl',
                        help='Path to input JSONL file containing questions')
    parser.add_argument('--output_all', type=str, default='all_evaluated_questions.jsonl',
                        help='Path to output JSONL file for all evaluated questions')
    parser.add_argument('--output_filtered', type=str, default='filtered_questions.jsonl',
                        help='Path to output JSONL file for filtered questions')
    parser.add_argument('--answer_model', type=str, default='gpt-4o',
                        help='OpenAI model to use for answering questions')
    parser.add_argument('--eval_model', type=str, default='o4-mini',
                        help='OpenAI model to use for evaluating difficulty')
    parser.add_argument('--temperature', type=float, default=0.3,
                        help='Temperature for LLM responses (lower = more deterministic)')
    parser.add_argument('--num_samples', type=int, default=3,
                        help='Number of evaluation samples for each question')
    parser.add_argument('--max_workers', type=int, default=48,
                        help='Maximum number of worker threads')
    return parser.parse_args()

def get_client():
    """Get a thread-local OpenAI client"""
    if not hasattr(thread_local, 'client'):
        thread_local.client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    return thread_local.client

def safe_json_loads(text):
    """
    Safely parse JSON by handling common issues with LLM outputs
    """
    # Try direct parsing first
    try:
        if "```json" in text and "```" in text:
            start = text.find("```json") + 7
            text = text[start:]
            end = text.rfind("```")
            text = text[:end].strip()
        text = text.strip()
        if not text.startswith("{"):
            text = text[text.find("{"):]
        if text.count("}") > 1:
            text = text[:text.rfind("}") + 1]
        text = text.replace("\n", "").replace("    ", "")
        return json.loads(text)
    except Exception as e:
        print(f"Failed to parse response.")
        return None

def evaluate_single_sample(question, model_answer, eval_model, sample_id=1):
    """Evaluate a question with a single sample"""
    eval_prompt = f"""
You are evaluating whether a question can be used for a benchmark of challenging questions. 
This benchmark aims at evaluating the most powerful LLMs' capabilities of solving the most difficult questions that are unsolved by human experts.
We only select questions that are difficult and even unsolvable by human experts.

Please evaluate the following question according to the criteria. You are also given an answer to this question provided by an LLM. If this LLM can answer the question correctly, it means the question is not difficult.

QUESTION TITLE: {question.get('title', '')}
QUESTION BODY: {question.get('body', '')}
TAGS: {question.get('tags', '')}
SITE: {question.get('source', '')}

MODEL ANSWER: {model_answer}

Evaluate based on the following criteria:
1. Answer_Correctness: (0-100%)
- Evaluate the probability that the model's answer is correct and completely solves the question.
- High score (80-100%) means the answer is comprehensive, accurate, and addresses all aspects of the problem.
- Medium score (40-79%) means the answer has minor errors or omissions but is mostly correct.
- Low score (20-39%) means the answer is partial, contains significant errors, lacks necessary references
- Very low score (0-19%) means the answer is mostly incorrect, doesn't touch the core of the question, very high-level, or merely states that the problem is unsolved/open.
- Consider both factual accuracy and solution completeness. Be strict in your evaluation.

2. Expert_Solve_Probability: (0-100%)
- Probability that domain experts (PhDs, Professors, Top Researchers) could solve this question correctly and completely.
- A low score (0-29%) means the question is extremely difficult even for domain experts.
- A medium score (30-69%) means experts would find it challenging but potentially solvable.
- A high score (70-100%) means experts is highly likely to solve it correctly and completely.
- Consider the depth of specialized knowledge and analytical skills required.

3. Answerable: (Yes or No)
- Can this question be answered with a definitive, verifiable solution, at least in principle?
- The question must have a well-defined problem statement and be logically sound.
- Answer "No" if it's fundamentally ill-posed, self-contradictory, based on demonstrably false premises or definitions, or requires information that cannot possibly be obtained.
- Answer "Yes" only if the question is valid and potentially solvable, even if no known answer currently exists.

4. Clear: (Yes or No)
- Is the question clearly stated with a well-defined objective without any ambiguity and missing information?
- Answer "No" if the question has multiple reasonable interpretations.
- Answer "No" if the question misses critical context, contains undefined variables, uses vague terminology, or has any other clarity issues.
- Answer "Yes" only if a domain expert would understand exactly what is being asked without any ambiguity.

5. Unambiguous_Answer: (Yes or No)
- Does this question have a definitive correct answer that can be objectively verified?
- Answer "No" to questions that have subjective answers like asking for reasons, opinions, or preferences.
- Answer "No" if the answer cannot be marked correct/incorrect without debate or subjective judgment.
- Answer "Yes" only if there exists a clear standard by which to judge the correctness of an answer.

Please be as strict and objective as possible.
"""

    max_retries = 3
    retry_delay = 5  # seconds
    client = get_client()

    for attempt in range(max_retries):
        try:
            eval_response = client.chat.completions.create(
                model=eval_model,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are an expert evaluator for a benchmark of extremely challenging questions. "
                            "Respond with ONLY a JSON object that exactly matches this schema and nothing else.\n"
                            "{\"Answer_Correctness\": int, \"Expert_Solve_Probability\": int, "
                            "\"Answerable\": \"Yes|No\", \"Clear\": \"Yes|No\", "
                            "\"Unambiguous_Answer\": \"Yes|No\", \"Explanation\": str}"
                        )
                    },
                    {"role": "user", "content": eval_prompt}
                ],
                reasoning_effort="high"
            )
            content = eval_response.choices[0].message.content.strip()
            
            # Use the robust JSON parser
            result = safe_json_loads(content)
            return result
            
        except Exception as e:
            if attempt < max_retries - 1:
                print(f"Error evaluating question (sample {sample_id}): {e}. Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
            else:
                print(f"Failed to evaluate question (sample {sample_id}) after {max_retries} attempts: {e}")
                return None

def process_question(question, answer_model, eval_model, num_samples=3, temperature=0.1):
    """
    Process a question using a two-model approach with multiple evaluation samples:
    1. One model attempts to answer the question
    2. Another model evaluates the difficulty and answer quality multiple times
    
    Returns a dictionary with aggregated evaluation results
    """
    client = get_client()

    # Step 1: Have GPT-4o attempt to answer the question
    answer_prompt = f"""
Please try your best to answer the following question:

QUESTION TITLE: {question.get('title', '')}
QUESTION BODY: {question.get('body', '')}
TAGS: {question.get('tags', '')}
SITE: {question.get('source', '')}

Provide your best and most accurate answer.
"""
    
    try:
        answer_response = client.chat.completions.create(
            model=answer_model,
            messages=[
                {"role": "system", "content": "You are an expert. Provide the most accurate answer possible."},
                {"role": "user", "content": answer_prompt}
            ],
            temperature=temperature
        )
        model_answer = answer_response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Error getting answer: {e}")
        model_answer = "Failed to generate answer"
    
    # Step 2: Evaluate multiple times
    results = []
    for i in range(num_samples):
        sample_result = evaluate_single_sample(
            question, 
            model_answer, 
            eval_model, 
            sample_id=i+1
        )
        if sample_result:
            results.append(sample_result)
        # Avoid rate limits
        time.sleep(1)
    
    # Step 3: Aggregate results
    aggregated_result = {
        "Model_Answer": model_answer
    }
    
    if not results:
        aggregated_result["Answer_Correctness"] = 100
        aggregated_result["Expert_Solve_Probability"] = 100
        aggregated_result["Answerable"] = "No"
        aggregated_result["Clear"] = "No"
        aggregated_result["Unambiguous_Answer"] = "No"
        aggregated_result["Explanation"] = "Failed to evaluate question"
        return aggregated_result

    # Calculate average for numerical values
    try:
        aggregated_result["Answer_Correctness"] = sum(float(r.get("Answer_Correctness", 0)) for r in results) / num_samples
    except (ValueError, TypeError):
        print("Error calculating Answer_Correctness average. Using default value.")
        aggregated_result["Answer_Correctness"] = 0  # Default to 0% if we can't calculate
        
    try:
        aggregated_result["Expert_Solve_Probability"] = sum(float(r.get("Expert_Solve_Probability", 50)) for r in results) / num_samples
    except (ValueError, TypeError):
        print("Error calculating Expert_Solve_Probability average. Using default value.")
        aggregated_result["Expert_Solve_Probability"] = 0  # Default to 50% if we can't calculate
    
    # For Yes/No questions, only "Yes" if all samples are "Yes"
    aggregated_result["Answerable"] = "Yes" if all(r.get("Answerable", "No") == "Yes" for r in results) else "No"
    aggregated_result["Clear"] = "Yes" if all(r.get("Clear", "No") == "Yes" for r in results) else "No"
    aggregated_result["Unambiguous_Answer"] = "Yes" if all(r.get("Unambiguous_Answer", "No") == "Yes" for r in results) else "No"
    
    # Combine explanations
    aggregated_result["Explanation"] = " | ".join([f"Sample {i+1}: {r.get('Explanation', 'No explanation')}" 
                                                   for i, r in enumerate(results)])
    
    return aggregated_result

def process_single_question_task(args):
    """Process a single question for multi-threading"""
    idx, question, answer_model, eval_model, num_samples, temperature = args
    
    # Skip questions that don't have required fields
    if not all(key in question for key in ['title', 'body']):
        print(f"Skipping question {idx} - missing required fields")
        return None
    
    result = process_question(
        question, 
        answer_model=answer_model, 
        eval_model=eval_model, 
        num_samples=num_samples,
        temperature=temperature
    )
    
    # Add evaluation results to the original data
    evaluated_question = question.copy()
    evaluated_question["Answer_Correctness"] = result.get("Answer_Correctness", 100)
    evaluated_question["Expert_Solve_Probability"] = result.get("Expert_Solve_Probability", 100)
    evaluated_question["Answerable"] = result.get("Answerable", "No")
    evaluated_question["Clear"] = result.get("Clear", "No")
    evaluated_question["Unambiguous_Answer"] = result.get("Unambiguous_Answer", "No")
    evaluated_question["Explanation"] = result.get("Explanation", "")
    evaluated_question["original_index"] = idx
    # evaluated_question["Model_Answer"] = result.get("Model_Answer", "")
    # evaluated_question["Individual_Samples"] = result.get("Individual_Samples", [])
    
    return evaluated_question

def read_jsonl(file_path):
    """Read questions from JSONL file"""
    questions = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                questions.append(json.loads(line.strip()))
            except json.JSONDecodeError as e:
                print(f"Error parsing line in {file_path}: {e}")
    return questions

def write_jsonl(data, file_path):
    """Write data to JSONL file"""
    with open(file_path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    print(f"Saved {len(data)} items to {file_path}")

def main():
    args = parse_args()
    
    # Load questions from JSONL
    questions = read_jsonl(args.input)
    print(f"Loaded {len(questions)} questions from {args.input}")
    
    # --- Resume Logic Start ---
    all_evaluated = []
    processed_indices = set()
    if os.path.exists(args.output_all):
        print(f"Found existing output file: {args.output_all}. Loading results...")
        try:
            all_evaluated = read_jsonl(args.output_all)
            # Get indices of already processed questions
            # Handle cases where 'original_index' might be missing (e.g., older files)
            processed_indices = {q['original_index'] for q in all_evaluated if 'original_index' in q}
            print(f"Loaded {len(all_evaluated)} previously evaluated questions. Found {len(processed_indices)} processed indices.")
        except Exception as e:
            print(f"Error loading or parsing existing results from {args.output_all}: {e}. Starting fresh.")
            all_evaluated = []
            processed_indices = set()
            # Optionally: backup the corrupted file
            # os.rename(args.output_all, args.output_all + ".bak")
    
    # Filter out questions that have already been processed
    questions_to_process = [(i, q) for i, q in enumerate(questions) if i not in processed_indices]
    
    if not questions_to_process:
        print("All questions have already been processed.")
        # Proceed to filtering and saving based on existing results
    else:
        print(f"Resuming processing. {len(questions_to_process)} questions remaining.")

    # --- Resume Logic End ---
    
    # Create thread-safe list for results (already initialized above)
    results_lock = threading.Lock()

    # Progress bar for completed tasks (only for remaining questions)
    pbar = tqdm(total=len(questions_to_process), desc="Processing questions")
    
    # Function to update progress and save incremental results
    def update_progress(future):
        result = future.result()
        if result:
            with results_lock:
                all_evaluated.append(result)
                if len(all_evaluated) % 10 == 0:
                    write_jsonl(all_evaluated, args.output_all) 
        pbar.update(1)
    
    # Prepare arguments for processing only the remaining questions
    question_args = [
        (idx, q, args.answer_model, args.eval_model, args.num_samples, args.temperature) 
        for idx, q in questions_to_process # Use the filtered list
    ]
    
    max_workers = min(args.max_workers, len(question_args))
    # Only run processing if there are questions left
    if questions_to_process:
        # Process questions in parallel
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(process_single_question_task, arg) for arg in question_args]
            for future in futures:
                future.add_done_callback(update_progress)
            
            # Wait for all futures to complete
            concurrent.futures.wait(futures)
        
        pbar.close()
    
    # Make sure all_evaluated is not empty before proceeding (could be empty if input was empty or all processed previously)
    if not all_evaluated:
        print("No questions were evaluated (either input empty, all processed previously, or errors occurred).")
        # Ensure output files are handled correctly even in this case
        # Write empty files if they don't exist? Or just skip? Let's skip filtering if no results.
        return # Exit early
    
    # Save all evaluated questions (including previously loaded and newly processed)
    # This overwrites the incremental saves with the final complete list
    write_jsonl(all_evaluated, args.output_all)
    print(f"Saved all {len(all_evaluated)} evaluated questions to {args.output_all}")
    
    # Filter extremely difficult questions (using the combined list)
    filtered_questions = [
        q for q in all_evaluated if (
            # Add checks for key existence before casting to float
            q.get("Answerable", "No") == "Yes" and
            q.get("Clear", "No") == "Yes" and
            q.get("Unambiguous_Answer", "No") == "Yes" and
            (float(q["Answer_Correctness"]) < 40 if "Answer_Correctness" in q else False) and
            (float(q["Expert_Solve_Probability"]) < 70 if "Expert_Solve_Probability" in q else False)
        )
    ]
    
    # Print statistics before saving filtered questions
    print("\nEvaluation Statistics:")
    print(f"Total questions evaluated (including resumed): {len(all_evaluated)}")
    
    # Safely calculate stats
    valid_ac_count = sum(1 for q in all_evaluated if "Answer_Correctness" in q)
    valid_esp_count = sum(1 for q in all_evaluated if "Expert_Solve_Probability" in q)
    valid_ans_count = sum(1 for q in all_evaluated if "Answerable" in q)
    valid_clr_count = sum(1 for q in all_evaluated if "Clear" in q)
    valid_una_count = sum(1 for q in all_evaluated if "Unambiguous_Answer" in q)

    if valid_ac_count > 0:
        try:
            answer_correctness_below_40 = sum(1 for q in all_evaluated if "Answer_Correctness" in q and float(q["Answer_Correctness"]) < 40)
            print(f"Questions with model correctness < 40%: {answer_correctness_below_40} ({answer_correctness_below_40/len(all_evaluated)*100:.1f}%)")
        except Exception as e:
            print(f"Error calculating answer correctness stats: {e}")
    else:
        print("Answer Correctness stats not available.")

    if valid_esp_count > 0:
        try:
            expert_prob_below_70 = sum(1 for q in all_evaluated if "Expert_Solve_Probability" in q and float(q["Expert_Solve_Probability"]) < 70)
            print(f"Questions with expert solve probability < 70%: {expert_prob_below_70} ({expert_prob_below_70/len(all_evaluated)*100:.1f}%)")
        except Exception as e:
            print(f"Error calculating expert probability stats: {e}")
    else:
        print("Expert Solve Probability stats not available.")

    if valid_ans_count > 0:
        answerable_yes = sum(1 for q in all_evaluated if q.get("Answerable", "No") == "Yes")
        print(f"Answerable: {answerable_yes} ({answerable_yes/len(all_evaluated)*100:.1f}%)")
    else:
        print("Answerable stats not available.")
        
    if valid_clr_count > 0:
        clear_yes = sum(1 for q in all_evaluated if q.get("Clear", "No") == "Yes")
        print(f"Clear: {clear_yes} ({clear_yes/len(all_evaluated)*100:.1f}%)")
    else:
        print("Clear stats not available.")

    if valid_una_count > 0:
        unambiguous_yes = sum(1 for q in all_evaluated if q.get("Unambiguous_Answer", "No") == "Yes")
        print(f"Unambiguous Answer: {unambiguous_yes} ({unambiguous_yes/len(all_evaluated)*100:.1f}%)")
    else:
        print("Unambiguous Answer stats not available.")
        
    # Combined filter stats
    print(f"Questions meeting all criteria: {len(filtered_questions)} ({len(filtered_questions)/len(all_evaluated)*100:.1f}% of evaluated)")
    
    # Save filtered questions if any found
    if filtered_questions:
        write_jsonl(filtered_questions, args.output_filtered)
        print(f"Saved {len(filtered_questions)} filtered questions to {args.output_filtered}")
    else:
        # Check if all_evaluated was not empty but filtering resulted in zero questions
        if all_evaluated: 
            print("Warning: No questions met all filtering criteria. Check your filtering thresholds.")
        # Save an empty file to avoid confusion if the file doesn't exist or if filtering yielded nothing
        if not os.path.exists(args.output_filtered) or all_evaluated:
             with open(args.output_filtered, 'w') as f:
                 f.write('')
             print(f"Created/Cleared file at {args.output_filtered}")
        
    # Print detailed breakdown of filtering results to help diagnose issues
    print("\nDetailed filter breakdown:")
    
    # Use safe checks for calculations
    answer_correct_fail = sum(1 for q in all_evaluated if not ("Answer_Correctness" in q and float(q["Answer_Correctness"]) < 40))
    expert_solve_fail = sum(1 for q in all_evaluated if not ("Expert_Solve_Probability" in q and float(q["Expert_Solve_Probability"]) < 70))
    answerable_fail = sum(1 for q in all_evaluated if q.get("Answerable", "No") != "Yes")
    clear_fail = sum(1 for q in all_evaluated if q.get("Clear", "No") != "Yes")
    unambiguous_fail = sum(1 for q in all_evaluated if q.get("Unambiguous_Answer", "No") != "Yes")
    
    print(f"Questions failing Answer_Correctness < 40 (or missing key): {answer_correct_fail}")
    print(f"Questions failing Expert_Solve_Probability < 70 (or missing key): {expert_solve_fail}")
    print(f"Questions failing Answerable == Yes: {answerable_fail}")
    print(f"Questions failing Clear == Yes: {clear_fail}")
    print(f"Questions failing Unambiguous_Answer == Yes: {unambiguous_fail}")

if __name__ == "__main__":
    main()