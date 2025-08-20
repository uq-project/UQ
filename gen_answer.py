import json
import argparse
import os
from typing import Dict, Any
from utils.utils import OPENAI_MODEL_LIST, ANTHROPIC_MODEL_LIST, GEMINI_MODEL_LIST, TOGETHER_MODEL_LIST
from utils.dataset_utils import load_uq_dataset, load_existing_answers, save_result_to_file
from utils.api_utils import (
    initialize_client, format_prompt, generate_openai_response,
    generate_anthropic_response, generate_gemini_response, generate_together_response
)
from dotenv import load_dotenv
import threading
import concurrent.futures
from tqdm import tqdm

load_dotenv("key.env")  

class APIClient:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.client = initialize_client(model_name)

    def generate_answer(self, question: Dict[str, Any]) -> str:
        """Generate answer using the appropriate API.
        
        Args:
            question: Dictionary containing question information
        """
        prompt = format_prompt(question)
        print(prompt)
        
        try:
            if self.model_name in OPENAI_MODEL_LIST:
                return generate_openai_response(self.client, self.model_name, prompt)
                
            elif self.model_name in ANTHROPIC_MODEL_LIST:
                return generate_anthropic_response(self.client, self.model_name, prompt)
                
            elif self.model_name in GEMINI_MODEL_LIST:
                return generate_gemini_response(self.client, prompt)
                
            elif self.model_name in TOGETHER_MODEL_LIST:
                return generate_together_response(self.client, self.model_name, prompt)
                
        except Exception as e:
            raise RuntimeError(f"Failed to generate answer: {str(e)}")


def process_question(client, question, output_file, lock):
    question_id = question['question_id']
    
    try:
        answer = client.generate_answer(question)
        
        # Create result dictionary with metadata
        result = {
            "question_id": question_id,
            "model_name": client.model_name,
            "answer": answer,
            "metadata": {
                "title": question.get("title", ""),
                "tags": question.get("tags", []),
                "site": question.get("site", ""),
                "category": question.get("category", "")
            }
        }
        
        # Write result to file with lock to prevent concurrent writes
        save_result_to_file(result, output_file, lock)
        
        return f"Successfully processed question {question_id}"
        
    except Exception as e:
        error_result = {
            "question_id": question_id,
            "model_name": client.model_name,
            "error": str(e),
            "metadata": {
                "title": question.get("title", ""),
                "tags": question.get("tags", []),
                "site": question.get("site", ""),
                "category": question.get("category", "")
            }
        }
        
        # Write error to file with lock
        save_result_to_file(error_result, output_file, lock)
        
        return f"Failed to process question {question_id}: {str(e)}"

def main():
    parser = argparse.ArgumentParser(description='Generate answers using API')
    parser.add_argument(
        '--model_name',
        default='o4-mini',
        help='Chosen model name'
    )
    parser.add_argument(
        "--max_workers",
        type=int,
        default=16,
        help="Maximum number of worker threads"
    )
    args = parser.parse_args()

    try:
        # Initialize client
        client = APIClient(args.model_name)
        
        # Load questions from Hugging Face dataset
        questions = load_uq_dataset()
        
        # Prepare output directory and set output file name based on model and dataset
        output_dir = "generated_answer"
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, f"{args.model_name.split('/')[-1]}_uq_dataset.jsonl")
        
        # Create file if it doesn't exist
        if not os.path.exists(output_file):
            with open(output_file, "w", encoding="utf-8") as f:
                pass
        
        # Load existing answers if file exists
        existing_answers = load_existing_answers(output_file)
        
        # Filter out questions that already have answers
        questions_to_process = [q for q in questions if q['question_id'] not in existing_answers]
        
        if not questions_to_process:
            print("All questions have already been processed.")
            return
        
        max_workers = min(args.max_workers, len(questions_to_process))
        print(f"Processing {len(questions_to_process)} questions with {max_workers} workers...")
        
        # Create a thread lock for file writing
        lock = threading.Lock()
        
        # Process questions in parallel using ThreadPoolExecutor
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_question = {
                executor.submit(
                    process_question, 
                    client, 
                    question, 
                    output_file, 
                    lock
                ): question for question in questions_to_process
            }
            
            # Process results as they complete with progress bar
            for future in tqdm(concurrent.futures.as_completed(future_to_question), total=len(questions_to_process)):
                question = future_to_question[future]
                try:
                    result = future.result()
                    # Uncomment to see detailed progress
                    # print(result)
                except Exception as exc:
                    print(f"Question {question['question_id']} generated an exception: {exc}")

        print(f"Successfully generated answers and saved to {output_file}")

    except Exception as e:
        print(f"An error occurred: {str(e)}")
        exit(1)

if __name__ == "__main__":
    main()